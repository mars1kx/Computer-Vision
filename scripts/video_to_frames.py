"""
Утилита: видео → набор кадров PNG в формате, который ждёт run.py.

Использование:
    python scripts/video_to_frames.py path/to/video.mp4
    python scripts/video_to_frames.py path/to/video.mp4 --out images_my
    python scripts/video_to_frames.py path/to/video.mp4 --every 5
    python scripts/video_to_frames.py path/to/video.mp4 --max 60 --resize 1280

Что делает:
    1. Открывает видеофайл через OpenCV (любой формат, который умеет ffmpeg/OpenCV).
    2. Идёт по кадрам, выбирает каждый --every-й.
    3. Опционально ресайзит до --resize по большей стороне (с сохранением пропорций).
    4. Сохраняет как images_dir/frame_NNNNNN.PNG — точно тот же формат, как
       в стартовом датасете, чтобы run.py их подхватил без правок.

Параметр --every нужен, чтобы:
    a) уменьшить число кадров (длинное видео → разумный размер датасета);
    b) увеличить смещение людей между соседними кадрами — это полезно
       для MOG2 (если каждый кадр почти идентичен, фон «впитывает» людей).

Рекомендация: для 30 fps видео используй --every 3..6 — это даст
эффективные 5..10 fps, чего достаточно для детекции и трекинга.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Нарезает видео на кадры PNG для run.py")
    p.add_argument("video", type=Path, help="Путь к видеофайлу (mp4/mov/avi/…)")
    p.add_argument("--out", type=Path, default=Path("images"),
                   help="Папка для кадров (по умолчанию ./images)")
    p.add_argument("--every", type=int, default=1,
                   help="Брать каждый N-й кадр (по умолчанию 1 — все)")
    p.add_argument("--max", type=int, default=0,
                   help="Сохранить не более N кадров (0 = без ограничения)")
    p.add_argument("--resize", type=int, default=0,
                   help="Ресайз: бо́льшая сторона → N px (0 = без ресайза)")
    p.add_argument("--start", type=float, default=0.0,
                   help="С какой секунды начать (по умолчанию 0)")
    p.add_argument("--clear", action="store_true",
                   help="Очистить папку --out перед сохранением")
    return p.parse_args()


def resize_keep_aspect(img, target: int):
    """Уменьшает изображение так, чтобы бо́льшая сторона стала target px."""
    h, w = img.shape[:2]
    if max(h, w) <= target:
        return img
    if w >= h:
        new_w = target
        new_h = int(h * target / w)
    else:
        new_h = target
        new_w = int(w * target / h)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def main() -> int:
    args = parse_args()

    if not args.video.exists():
        print(f"Файл не найден: {args.video}", file=sys.stderr)
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    if args.clear:
        for old in args.out.glob("frame_*.PNG"):
            old.unlink()
        for old in args.out.glob("frame_*.png"):
            old.unlink()

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        print(f"OpenCV не смог открыть видео: {args.video}", file=sys.stderr)
        return 1

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    src_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    print(f"Видео: {args.video.name}")
    print(f"  fps={src_fps:.2f}   кадров={src_total}   размер={src_w}×{src_h}")

    # Промотать на --start секунд
    if args.start > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, args.start * 1000.0)
        print(f"  старт с {args.start:.2f}s")

    read_idx = -1
    saved = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        read_idx += 1
        if read_idx % args.every != 0:
            continue
        if args.resize > 0:
            frame = resize_keep_aspect(frame, args.resize)
        out_path = args.out / f"frame_{saved:06d}.PNG"
        cv2.imwrite(str(out_path), frame)
        saved += 1
        if saved % 20 == 0:
            print(f"  сохранено {saved}")
        if args.max and saved >= args.max:
            break

    cap.release()
    print(f"\nГотово: {saved} кадров в {args.out}/")
    print(f"Запусти пайплайн: python run.py --save --video")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
