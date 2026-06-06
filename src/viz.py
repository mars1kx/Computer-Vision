"""
Модуль визуализации: отрисовка bbox-ов, ID, траекторий, HUD и розы направлений.

Используется в run.py для live-окна и сохранения output.mp4.

Автор: Lead CV Engineer + Morphology & Report Lead.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from src.decide import SECTOR_NAMES, FinalDecision
from src.tracker import Track


# Цвета (BGR)
COLOR_BBOX = (0, 255, 0)             # зелёный — рамка человека
COLOR_BBOX_GROUP = (0, 165, 255)     # оранжевый — большая (группа)
COLOR_TRAIL = (255, 100, 0)          # синий — траектория
COLOR_ID = (0, 255, 255)             # жёлтый — ID
COLOR_HUD_BG = (0, 0, 0)
COLOR_HUD_FG = (255, 255, 255)
COLOR_ARROW = (0, 220, 220)          # светло-жёлтый — стрелка направления

# Минимальное число реальных сопоставлений (hits), чтобы РИСОВАТЬ трек.
# Свежие треки (hits 1–2) часто оказываются мерцанием/мусором и потом
# отсекаются в decide — чтобы они не «мигали» рамками на видео, не рисуем
# их, пока трек не подтвердится несколькими кадрами.
MIN_HITS_TO_DRAW: int = 3


def draw_frame(
    base_bgr: np.ndarray,
    active_tracks: list[Track],
    frame_idx: int,
    total_frames: int,
    rose_counts: dict[str, int] | None = None,
    dominant: str | None = None,
) -> np.ndarray:
    """
    Накладывает на base_bgr весь оверлей: bbox-ы, хвосты, ID, стрелки,
    HUD-полосу сверху, мини-розу направлений в углу.

    Параметры
    ---------
    base_bgr : кадр, на который рисуем.
    active_tracks : треки, активные в этом кадре.
    frame_idx, total_frames : для HUD-счётчика.
    rose_counts : словарь {sector: count} для мини-розы. Если None — роза не рисуется.
    dominant : строка с преобладающим направлением для HUD.
    """
    img = base_bgr.copy()
    H, W = img.shape[:2]

    # Рисуем только подтверждённые треки — без свежих фантомов (hits<порога).
    drawn_tracks = [
        t for t in active_tracks
        if getattr(t, "hits", t.age) >= MIN_HITS_TO_DRAW
    ]

    # --- bbox + ID + хвост + стрелка ---
    for t in drawn_tracks:
        x, y, w, h = t.bbox
        is_group = w * h > 6000
        color = COLOR_BBOX_GROUP if is_group else COLOR_BBOX
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, f"#{t.track_id}", (x, max(y - 3, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_ID, 2)

        pts = list(t.history)
        for k in range(1, len(pts)):
            cv2.line(img,
                     (int(pts[k - 1][0]), int(pts[k - 1][1])),
                     (int(pts[k][0]), int(pts[k][1])),
                     COLOR_TRAIL, 1)

        # Стрелка направления — от первой точки истории к последней
        if len(pts) >= 3:
            p0 = pts[0]
            p1 = pts[-1]
            dx = p1[0] - p0[0]
            dy = p1[1] - p0[1]
            if math.hypot(dx, dy) >= 8:
                cv2.arrowedLine(
                    img,
                    (int(p1[0]), int(p1[1])),
                    (int(p1[0] + dx * 0.25), int(p1[1] + dy * 0.25)),
                    COLOR_ARROW, 2, tipLength=0.4,
                )

    # --- HUD сверху ---
    cv2.rectangle(img, (0, 0), (W, 32), COLOR_HUD_BG, -1)
    n_active = len(drawn_tracks)
    hud_text = f"frame {frame_idx}/{total_frames - 1}   active people: {n_active}"
    if dominant:
        hud_text += f"   dominant: {dominant}"
    cv2.putText(img, hud_text, (10, 22), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, COLOR_HUD_FG, 1)

    # --- Мини-роза направлений в правом-нижнем углу ---
    if rose_counts is not None:
        _draw_rose(img, rose_counts, position="bottom-right")

    return img


def _draw_rose(img: np.ndarray, counts: dict[str, int],
               position: str = "bottom-right",
               radius: int = 70) -> None:
    """
    Рисует мини-розу направлений: 8 лучей разной длины, пропорциональных
    числу людей в каждом секторе.

    OpenCV: y растёт вниз, поэтому "S" — это +y (вниз).
    Углы для секторов (в радианах, atan2-конвенция):
        E=0, SE=π/4, S=π/2, SW=3π/4, W=π, NW=-3π/4, N=-π/2, NE=-π/4
    """
    H, W = img.shape[:2]
    margin = 20
    if position == "bottom-right":
        cx, cy = W - radius - margin, H - radius - margin
    else:
        cx, cy = radius + margin, H - radius - margin

    # Фон-кружок
    cv2.circle(img, (cx, cy), radius + 8, (40, 40, 40), -1)
    cv2.circle(img, (cx, cy), radius + 8, (180, 180, 180), 1)

    max_count = max(counts.values()) if counts and max(counts.values()) > 0 else 1

    sector_angles = {
        "E":  0.0,
        "SE": math.pi / 4,
        "S":  math.pi / 2,
        "SW": 3 * math.pi / 4,
        "W":  math.pi,
        "NW": -3 * math.pi / 4,
        "N":  -math.pi / 2,
        "NE": -math.pi / 4,
    }

    for sector in SECTOR_NAMES:
        n = counts.get(sector, 0)
        if n == 0:
            length = 6
        else:
            length = int(radius * (n / max_count))
        ang = sector_angles[sector]
        ex = int(cx + length * math.cos(ang))
        ey = int(cy + length * math.sin(ang))
        cv2.line(img, (cx, cy), (ex, ey), COLOR_ARROW, 2)

        # Подпись сектора чуть дальше луча
        lx = int(cx + (radius + 14) * math.cos(ang))
        ly = int(cy + (radius + 14) * math.sin(ang)) + 4
        cv2.putText(img, sector, (lx - 8, ly), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (220, 220, 220), 1)

    # Центральная точка
    cv2.circle(img, (cx, cy), 3, (255, 255, 255), -1)


def draw_summary_rose(decision: FinalDecision, size: int = 480) -> np.ndarray:
    """
    Большая отдельная роза направлений для итогового отчёта.
    Возвращает картинку size×size.
    """
    img = np.full((size, size, 3), 30, dtype=np.uint8)
    cx, cy = size // 2, size // 2
    radius = size // 2 - 50

    # Концентрические круги
    for r in (radius // 3, 2 * radius // 3, radius):
        cv2.circle(img, (cx, cy), r, (60, 60, 60), 1)

    max_count = max(decision.by_sector.values()) or 1
    sector_angles = {
        "E":  0.0, "SE": math.pi / 4, "S":  math.pi / 2,
        "SW": 3 * math.pi / 4, "W":  math.pi,
        "NW": -3 * math.pi / 4, "N":  -math.pi / 2, "NE": -math.pi / 4,
    }

    for sector in SECTOR_NAMES:
        n = decision.by_sector.get(sector, 0)
        ang = sector_angles[sector]
        length = int(radius * (n / max_count)) if n > 0 else 8
        ex = int(cx + length * math.cos(ang))
        ey = int(cy + length * math.sin(ang))

        color = (0, 240, 240) if sector == decision.dominant_direction else (180, 180, 180)
        cv2.line(img, (cx, cy), (ex, ey), color, 3)

        lx = int(cx + (radius + 30) * math.cos(ang))
        ly = int(cy + (radius + 30) * math.sin(ang)) + 6
        cv2.putText(img, f"{sector}: {n}", (lx - 30, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.circle(img, (cx, cy), 5, (255, 255, 255), -1)

    # Заголовок
    title = f"People = {decision.total_people}   dominant = {decision.dominant_direction}"
    cv2.putText(img, title, (20, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (255, 255, 255), 1)

    return img
