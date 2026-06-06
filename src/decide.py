"""
Стадия 5 пайплайна: DECIDE — финальное решение системы.

Что приходит на вход:
    Все треки за весь ролик (CentroidTracker.collect_finished_tracks()).
    Каждый трек = один человек со своей историей центроидов.

Что должно получиться:
    Автоматическое, интерпретируемое решение, как требует brief:
      • total_people   — итоговое количество уникальных людей в ролике
      • dominant       — преобладающее направление потока (N/NE/E/.../NW)
      • by_sector      — распределение людей по 8 секторам (роза)
      • per_track      — направление, скорость, длина пути для каждого трека

Используемые методы:
    - math.atan2 для угла между двумя точками
    - простая статистика (Counter, mean)

Никакого OpenCV здесь нет специально — это «чистая» стадия принятия
решения, которая работает поверх результатов всех предыдущих стадий.
Так требует brief: «Decide produces an automatic, interpretable final output».

Автор стадии: Lead CV Engineer.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

from src.tracker import Track


# -------------------------------------------------------------------------
# Параметры стадии.
# -------------------------------------------------------------------------
# Минимальное число РЕАЛЬНЫХ сопоставлений (hits), чтобы засчитать трек
# как человека. Считаем именно по hits, а не по age: age растёт даже на
# пропущенных кадрах, поэтому короткие фантомы успевали «дожить» до порога.
# hits — это сколько кадров трек реально совпал с детекцией.
MIN_HITS_FOR_COUNT: int = 10

# Минимальное смещение (px) между первой и последней точкой трека,
# чтобы считать его «движущимся». Ниже — человек стоял на месте.
MIN_DISPLACEMENT_PX: float = 80.0

# Имена 8 секторов. Углы — стандартные:
#   0°    = East   (+x)
#   90°   = South  (+y — y растёт вниз в координатах OpenCV!)
#   180°  = West
#   -90°  = North
SECTOR_NAMES: tuple[str, ...] = (
    "E", "SE", "S", "SW", "W", "NW", "N", "NE"
)


@dataclass
class TrackDecision:
    """Решение по одному треку."""
    track_id: int
    age: int                       # сколько кадров жил
    displacement_px: float         # длина пути (от первой до последней точки)
    angle_deg: float | None        # угол движения в градусах (или None если стоял)
    direction: str                 # "E"/"SE"/.../"NW" или "stationary"


@dataclass
class FinalDecision:
    """Итоговое решение пайплайна. Сериализуется в stats.json."""
    total_people: int
    dominant_direction: str
    by_sector: dict[str, int]
    per_track: list[TrackDecision]

    def to_dict(self) -> dict:
        return {
            "total_people": self.total_people,
            "dominant_direction": self.dominant_direction,
            "by_sector": self.by_sector,
            "per_track": [
                {
                    "track_id": d.track_id,
                    "age": d.age,
                    "displacement_px": round(d.displacement_px, 1),
                    "angle_deg": (
                        None if d.angle_deg is None else round(d.angle_deg, 1)
                    ),
                    "direction": d.direction,
                }
                for d in self.per_track
            ],
        }


# -------------------------------------------------------------------------
# Решающая функция.
# -------------------------------------------------------------------------
def decide(tracks: list[Track]) -> FinalDecision:
    """
    Считает финальные метрики по всем трекам.
    """
    per_track: list[TrackDecision] = []
    sector_counter: Counter[str] = Counter()
    valid = 0

    for t in tracks:
        # Считаем по реальным сопоставлениям (hits). Старое поле age росло
        # и на пропущенных кадрах, из-за чего фантомы перешагивали порог.
        hits = getattr(t, "hits", t.age)
        if hits < MIN_HITS_FOR_COUNT:
            continue  # трека реально видели слишком мало кадров — не человек
        valid += 1

        first = t.history[0]
        last = t.history[-1]
        dx = last[0] - first[0]
        dy = last[1] - first[1]
        disp = math.hypot(dx, dy)

        if disp < MIN_DISPLACEMENT_PX:
            decision = TrackDecision(
                track_id=t.track_id,
                age=t.age,
                displacement_px=disp,
                angle_deg=None,
                direction="stationary",
            )
        else:
            # atan2 возвращает угол в радианах в диапазоне [-π, π].
            # В OpenCV ось y направлена вниз, поэтому +y = South.
            angle_rad = math.atan2(dy, dx)
            angle_deg = math.degrees(angle_rad)
            sector = _angle_to_sector(angle_deg)
            decision = TrackDecision(
                track_id=t.track_id,
                age=t.age,
                displacement_px=disp,
                angle_deg=angle_deg,
                direction=sector,
            )
            sector_counter[sector] += 1

        per_track.append(decision)

    # Заполним нулями отсутствующие секторы — удобно для отчёта/розы
    by_sector = {s: int(sector_counter.get(s, 0)) for s in SECTOR_NAMES}

    dominant = (
        sector_counter.most_common(1)[0][0]
        if sector_counter
        else "stationary"
    )

    return FinalDecision(
        total_people=valid,
        dominant_direction=dominant,
        by_sector=by_sector,
        per_track=per_track,
    )


def _angle_to_sector(angle_deg: float) -> str:
    """
    Переводит угол в градусах (atan2-конвенция) в имя одного из 8 секторов.
    OpenCV ось y направлена вниз → положительные углы = South-half.
    """
    # Нормализуем угол в [0, 360)
    a = angle_deg % 360.0
    # Делим круг на 8 секторов по 45°, сдвинутых на 22.5° так, чтобы
    # центр сектора "E" был на 0°.
    idx = int((a + 22.5) // 45) % 8
    return SECTOR_NAMES[idx]


# -------------------------------------------------------------------------
# Smoke-тест. Запуск:
#     python -m src.decide
# -------------------------------------------------------------------------
if __name__ == "__main__":
    from collections import deque
    from src.tracker import Track

    # Искусственные треки на 4 направления
    tracks = [
        # Идёт на восток (+x)
        Track(1, (0, 0, 30, 80), (300, 200),
              deque([(100, 200), (300, 200)], maxlen=32), age=10),
        # Идёт на юг (+y)
        Track(2, (0, 0, 30, 80), (500, 400),
              deque([(500, 200), (500, 400)], maxlen=32), age=10),
        # Идёт на северо-запад
        Track(3, (0, 0, 30, 80), (200, 100),
              deque([(400, 300), (200, 100)], maxlen=32), age=10),
        # Стоял на месте
        Track(4, (0, 0, 30, 80), (505, 205),
              deque([(500, 200), (505, 205)], maxlen=32), age=10),
        # Слишком короткий трек — должен отлететь
        Track(5, (0, 0, 30, 80), (0, 0),
              deque([(0, 0)], maxlen=32), age=1),
    ]

    decision = decide(tracks)
    print(f"total_people:       {decision.total_people}")
    print(f"dominant_direction: {decision.dominant_direction}")
    print(f"by_sector:          {decision.by_sector}")
    print("per_track:")
    for d in decision.per_track:
        print(f"  id={d.track_id}  age={d.age}  disp={d.displacement_px:.1f}  "
              f"angle={d.angle_deg}  dir={d.direction}")
