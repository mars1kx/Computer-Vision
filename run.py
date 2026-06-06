"""
Главный скрипт пайплайна People Counting & Flow Direction.

Запуск:
    python run.py                   # минимум: stats.json + summary-роза
    python run.py --save            # + 6 PNG на каждый кадр (требование brief)
    python run.py --live            # live-окно с детекциями (real-time демо)
    python run.py --video           # + output.mp4 со всеми оверлеями
    python run.py --save --live --video   # всё вместе

Полный цикл:
    1. Загрузка кадров из images/.
    2. Прогон №1 — прогрев MOG2 (построение фоновой модели).
    3. Прогон №2 — для каждого кадра:
         enhance → segment → clean → detect → tracker.update
         (опционально: save 6 PNG в outputs/per_frame/<frame>/)
         (опционально: показ live-окна)
         (опционально: запись output.mp4)
    4. После всех кадров — decide → stats.json + direction_rose.png.

Автор: Lead CV Engineer.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from src.enhance import enhance
from src.segment import Segmenter
from src.clean import clean, build_static_mask
from src.detect import detect
from src.tracker import CentroidTracker
from src.decide import decide
from src.viz import draw_frame, draw_summary_rose


IMAGES_DIR = Path("images")
OUTPUTS_DIR = Path("outputs")
PER_FRAME_DIR = OUTPUTS_DIR / "per_frame"
FINAL_DIR = OUTPUTS_DIR / "final"

# Временное сглаживание clean-маски: пиксель остаётся, если он был
# foreground хотя бы в SMOOTH_MIN_HITS из последних SMOOTH_WINDOW кадров.
# Это гасит одно-кадровое мерцание силуэтов, из-за которого детекции
# то появлялись, то пропадали, а трекер пересоздавал ID.
SMOOTH_WINDOW: int = 3
SMOOTH_MIN_HITS: int = 2


def temporal_smooth(mask_buffer: list[np.ndarray]) -> np.ndarray:
    """
    Возвращает сглаженную бинарную маску по «большинству» из буфера
    последних масок. Пока буфер короче окна — работает по тому, что есть
    (порог пропорционально снижается, чтобы не глушить первые кадры).
    """
    stack = np.stack([(m > 0).astype(np.uint8) for m in mask_buffer], axis=0)
    votes = stack.sum(axis=0)
    need = SMOOTH_MIN_HITS if len(mask_buffer) >= SMOOTH_WINDOW else 1
    return (votes >= need).astype(np.uint8) * 255


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="People Counting & Flow Direction — главный пайплайн"
    )
    p.add_argument("--save", action="store_true",
                   help="сохранить 6 PNG (original, enhanced, mask, clean, "
                        "detect, decision) на каждый кадр")
    p.add_argument("--live", action="store_true",
                   help="показать live-окно с результатами")
    p.add_argument("--video", action="store_true",
                   help="экспортировать выходное видео outputs/final/output.mp4")
    p.add_argument("--fps", type=int, default=10,
                   help="FPS для выходного видео (по умолчанию 10)")
    p.add_argument("--images", type=Path, default=IMAGES_DIR,
                   help="папка с кадрами (по умолчанию ./images)")
    return p.parse_args()


def load_frames(images_dir: Path) -> tuple[list[np.ndarray], list[Path]]:
    """Загружает все кадры из папки и возвращает (bgr_frames, paths)."""
    paths = sorted(images_dir.glob("frame_*.PNG"))
    if not paths:
        # Совместимость с другими расширениями
        paths = sorted(list(images_dir.glob("frame_*.png")) +
                       list(images_dir.glob("frame_*.jpg")))
    if not paths:
        print(f"Не найдено кадров в {images_dir}", file=sys.stderr)
        sys.exit(1)

    frames = []
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            print(f"Не удалось прочитать {p}", file=sys.stderr)
            sys.exit(1)
        
        # Автоматический ресайз до 1280 по большей стороне для корректной работы всех порогов
        h, w = img.shape[:2]
        long_side = max(h, w)
        if long_side != 1280:
            scale = 1280.0 / float(long_side)
            new_w = int(round(w * scale))
            new_h = int(round(h * scale))
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
        frames.append(img)
    return frames, paths


def save_per_frame_outputs(
    frame_dir: Path,
    original: np.ndarray,
    enhanced: np.ndarray,
    mask_segment: np.ndarray,
    mask_clean: np.ndarray,
    detect_overlay: np.ndarray,
    decision_overlay: np.ndarray,
) -> None:
    """
    Сохраняет 6 файлов на кадр — ровно как требует brief:
        Original / Enhanced / Mask / Clean / Detect / Decision.
    """
    frame_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(frame_dir / "01_original.png"), original)
    cv2.imwrite(str(frame_dir / "02_enhanced.png"), enhanced)
    cv2.imwrite(str(frame_dir / "03_mask.png"), mask_segment)
    cv2.imwrite(str(frame_dir / "04_clean.png"), mask_clean)
    cv2.imwrite(str(frame_dir / "05_detect.png"), detect_overlay)
    cv2.imwrite(str(frame_dir / "06_decision.png"), decision_overlay)


def main() -> int:
    args = parse_args()

    print("=" * 60)
    print("People Counting & Flow Direction — pipeline")
    print("=" * 60)

    # --- 1. Загрузка ----------------------------------------------------
    print(f"Загружаю кадры из {args.images}…")
    raw_frames, paths = load_frames(args.images)
    print(f"  кадров: {len(raw_frames)}  размер: {raw_frames[0].shape[1]}×{raw_frames[0].shape[0]}")

    # --- 2. Enhance (выполняем сразу для всех — нужно для warmup) -------
    print("Enhance…")
    enhanced_frames = [enhance(f) for f in raw_frames]

    # --- 3. Warmup MOG2 -------------------------------------------------
    print("Прогрев MOG2 (построение модели фона)…")
    segmenter = Segmenter()
    segmenter.warmup(enhanced_frames)

    # --- 3b. Строим статическую маску фоновых артефактов --------------------
    # Прогоняем все кадры через сегментатор для сбора combined масок,
    # из которых строим маску постоянных «не-людских» пикселей.
    print("Построение статической маски фона…")
    pre_segmenter = Segmenter()
    pre_segmenter.warmup(enhanced_frames)
    pre_combined_masks = []
    for f in enhanced_frames:
        _, _, combined = pre_segmenter.segment(f)
        pre_combined_masks.append(combined)
    static_mask = build_static_mask(pre_combined_masks, threshold=0.6)
    print(f"  статических пикселей: {int((static_mask > 0).sum())}")

    # --- 4. Основной прогон --------------------------------------------
    print("Основной прогон: segment → clean → detect → track…")
    tracker = CentroidTracker()

    video_writer = None
    if args.video:
        FINAL_DIR.mkdir(parents=True, exist_ok=True)
        h, w = raw_frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(
            str(FINAL_DIR / "output.mp4"), fourcc, args.fps, (w, h)
        )

    if args.save:
        PER_FRAME_DIR.mkdir(parents=True, exist_ok=True)

    if args.live:
        cv2.namedWindow("People Flow", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("People Flow", 1280, 720)

    quit_early = False
    smooth_buffer: list[np.ndarray] = []  # последние clean-маски для сглаживания

    for i, (raw, enhanced_img, src_path) in enumerate(
        zip(raw_frames, enhanced_frames, paths)
    ):
        # --- segment ---
        mog_mask, otsu_mask, mask_segment = segmenter.segment(enhanced_img)
        # --- clean ---
        mask_clean = clean(mask_segment, static_mask=static_mask)
        # --- temporal smoothing (гасит мерцание силуэтов) ---
        smooth_buffer.append(mask_clean)
        if len(smooth_buffer) > SMOOTH_WINDOW:
            smooth_buffer.pop(0)
        mask_clean = temporal_smooth(smooth_buffer)
        # --- detect + track ---
        bboxes = detect(mask_clean)
        active = tracker.update(bboxes)

        # Визуализация для этого кадра
        detect_vis = raw.copy()
        for (x, y, w, h) in bboxes:
            color = (0, 165, 255) if w * h > 6000 else (0, 255, 0)
            cv2.rectangle(detect_vis, (x, y), (x + w, y + h), color, 2)

        decision_vis = draw_frame(
            raw, active,
            frame_idx=i, total_frames=len(raw_frames),
            rose_counts=None, dominant=None,
        )

        if args.save:
            save_per_frame_outputs(
                PER_FRAME_DIR / src_path.stem,
                original=raw,
                enhanced=enhanced_img,
                mask_segment=mask_segment,
                mask_clean=mask_clean,
                detect_overlay=detect_vis,
                decision_overlay=decision_vis,
            )

        if video_writer is not None:
            video_writer.write(decision_vis)

        if args.live:
            cv2.imshow("People Flow", decision_vis)
            # 25 ms ~ 40 fps; q или Esc — выход
            key = cv2.waitKey(25) & 0xFF
            if key in (ord("q"), 27):
                quit_early = True
                break

        if (i + 1) % 10 == 0 or i == len(raw_frames) - 1:
            print(f"  кадр {i+1}/{len(raw_frames)}   active={len(active)}")

    if video_writer is not None:
        video_writer.release()
        print(f"  видео сохранено: {FINAL_DIR / 'output.mp4'}")

    if args.live:
        cv2.destroyAllWindows()
        if quit_early:
            print("  live-окно закрыто пользователем (q/Esc) — прерываюсь.")

    # --- 5. Decide ------------------------------------------------------
    print("Decide…")
    all_tracks = tracker.collect_finished_tracks()
    final = decide(all_tracks)

    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    stats_path = FINAL_DIR / "stats.json"
    with open(stats_path, "w", encoding="utf-8") as fp:
        json.dump(final.to_dict(), fp, ensure_ascii=False, indent=2)
    print(f"  stats.json → {stats_path}")

    rose = draw_summary_rose(final)
    rose_path = FINAL_DIR / "direction_rose.png"
    cv2.imwrite(str(rose_path), rose)
    print(f"  rose       → {rose_path}")

    # --- 6. Итог в консоль ---------------------------------------------
    print()
    print("=" * 60)
    print("FINAL")
    print("=" * 60)
    print(f"  total_people:        {final.total_people}")
    print(f"  dominant_direction:  {final.dominant_direction}")
    print( "  by_sector:")
    for s, n in final.by_sector.items():
        bar = "█" * n
        print(f"    {s:>3}: {bar} {n}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
