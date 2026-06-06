"""
CentroidTracker — простой трекер по центроидам bounding-box'ов.

Идея:
    Для каждого нового кадра у нас есть N bbox-кандидатов. Для каждого
    существующего трека мы ищем ближайший bbox в радиусе R пикселей.
    Если нашли — обновляем трек. Если нет — трек «тухнет», и после
    нескольких пропусков удаляется. Несоотнесённые bbox становятся
    новыми треками с новым ID.

Зачем нужен:
    Без трекера мы умеем только подсчитывать «детекции в кадре», а нам
    нужно «уникальных людей в ролике» и «направление движения каждого».
    Трекер связывает детекции через время.

Алгоритм соотнесения — жадное по расстоянию (Hungarian-like, но проще):
    1. Считаем матрицу расстояний centroid_track ↔ centroid_detection.
    2. Идём по парам в порядке возрастания расстояния.
    3. Берём пару, если оба ещё не заняты и расстояние < MAX_DISTANCE.
    4. Остальные треки — пропуск, остальные детекции — новые треки.

Поле history каждого трека хранит последние центроиды — на их основе
стадия Decide посчитает направление и расстояние движения.

Автор стадии: Lead CV Engineer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from collections import deque

import numpy as np


# -------------------------------------------------------------------------
# Параметры трекера.
# -------------------------------------------------------------------------
# Видео нарезано каждый ~4-й кадр (≈7.5 fps), поэтому человек смещается
# между соседними кадрами заметно — на 80 px связь рвётся и трекер плодит
# новые ID. Поднимаем порог и добавляем предсказание позиции по скорости.
MAX_DISTANCE: float = 140.0     # максимальное расстояние для сопоставления (px)
MAX_MISSED: int = 4             # сколько кадров терпеть «пропавший» трек
                                # (умеренно: больше плодит «долгоживущих» фантомов)
HISTORY_LEN: int = 32           # длина истории центроидов (для направления)

# Доля «инерции» при предсказании: следующая позиция ≈ last + VELOCITY_GAIN*v,
# где v — средняя скорость по последним кадрам. 1.0 = полное предсказание.
VELOCITY_GAIN: float = 1.0
# По скольким последним шагам усредняем скорость.
VELOCITY_WINDOW: int = 3


@dataclass
class Track:
    """Один трек = один человек на видео."""
    track_id: int
    bbox: tuple[int, int, int, int]                    # (x, y, w, h) в последнем кадре
    centroid: tuple[float, float]                      # (cx, cy) последний
    history: deque = field(default_factory=lambda: deque(maxlen=HISTORY_LEN))
    missed: int = 0                                    # сколько кадров не сопоставлен
    age: int = 0                                       # сколько кадров живёт (вкл. пропуски)
    hits: int = 0                                      # сколько кадров реально сопоставлен с детекцией

    def predicted_centroid(self) -> tuple[float, float]:
        """
        Предсказывает, где трек окажется в СЛЕДУЮЩЕМ кадре, исходя из
        средней скорости по последним VELOCITY_WINDOW шагам. Сопоставление
        с детекциями идёт по этой точке, а не по последней — это удерживает
        ID на быстрых людях (важно при низком эффективном fps).

        Если пропусков было несколько (missed>0), экстраполируем дальше,
        умножая скорость на (missed+1).
        """
        h = list(self.history)
        if len(h) < 2:
            return self.centroid
        win = min(VELOCITY_WINDOW, len(h) - 1)
        # средняя скорость за последние win шагов
        dx = (h[-1][0] - h[-1 - win][0]) / win
        dy = (h[-1][1] - h[-1 - win][1]) / win
        steps = VELOCITY_GAIN * (self.missed + 1)
        return (self.centroid[0] + dx * steps, self.centroid[1] + dy * steps)


class CentroidTracker:
    """
    Минималистичный многообъектный трекер по центроидам.

    Применение в пайплайне:
        tracker = CentroidTracker()
        for frame in frames:
            bboxes = detect(...)         # список (x,y,w,h)
            active_tracks = tracker.update(bboxes)
            # active_tracks — это список Track для отрисовки/анализа
    """

    def __init__(
        self,
        max_distance: float = MAX_DISTANCE,
        max_missed: int = MAX_MISSED,
    ) -> None:
        self._max_distance = max_distance
        self._max_missed = max_missed
        self._next_id = 1
        self._tracks: dict[int, Track] = {}
        self._archive: list[Track] = []  # «протухшие» треки сохраняем сюда

    # ----------------------------------------------------------------- public
    def update(self, bboxes: list[tuple[int, int, int, int]]) -> list[Track]:
        """
        Обновляет треки новыми детекциями. Возвращает список «живых» треков.

        bboxes : список (x, y, w, h) детекций в текущем кадре.
        """
        # Центроиды детекций
        det_centroids = [
            (x + w / 2.0, y + h / 2.0) for (x, y, w, h) in bboxes
        ]

        if not self._tracks:
            # Стартовый кадр: каждый bbox → новый трек
            for b, c in zip(bboxes, det_centroids):
                self._spawn(b, c)
            return self._snapshot()

        if not bboxes:
            # Нет детекций — всем трекам ставим +1 пропуск
            self._age_and_kill()
            return self._snapshot()

        # Жадное сопоставление.
        # Сопоставляем детекции не с ПОСЛЕДНЕЙ позицией трека, а с
        # ПРЕДСКАЗАННОЙ (last + скорость) — так трек «догоняет» быстро
        # идущего человека и не рвётся в новый ID.
        track_ids = list(self._tracks.keys())
        track_centroids = np.array(
            [self._tracks[i].predicted_centroid() for i in track_ids]
        )
        det_arr = np.array(det_centroids)

        # Матрица расстояний (T × D)
        dists = np.linalg.norm(
            track_centroids[:, None, :] - det_arr[None, :, :], axis=2
        )

        # Список (расстояние, t_idx, d_idx)
        pairs = sorted(
            (
                (dists[t, d], t, d)
                for t in range(len(track_ids))
                for d in range(len(bboxes))
            ),
            key=lambda p: p[0],
        )

        used_tracks: set[int] = set()
        used_dets: set[int] = set()

        for dist, t_idx, d_idx in pairs:
            if dist > self._max_distance:
                break
            if t_idx in used_tracks or d_idx in used_dets:
                continue
            used_tracks.add(t_idx)
            used_dets.add(d_idx)
            # Обновляем трек
            track = self._tracks[track_ids[t_idx]]
            track.bbox = bboxes[d_idx]
            track.centroid = det_centroids[d_idx]
            track.history.append(det_centroids[d_idx])
            track.missed = 0
            track.age += 1
            track.hits += 1

        # Несопоставленные треки → стареют
        for t_idx, tid in enumerate(track_ids):
            if t_idx not in used_tracks:
                self._tracks[tid].missed += 1
                self._tracks[tid].age += 1

        # Несопоставленные детекции → новые треки
        for d_idx, (b, c) in enumerate(zip(bboxes, det_centroids)):
            if d_idx not in used_dets:
                self._spawn(b, c)

        # Удаляем «протухшие»
        self._kill_dead()

        return self._snapshot()

    # ----------------------------------------------------------------- internals
    def _spawn(self, bbox: tuple[int, int, int, int], centroid: tuple[float, float]) -> None:
        track = Track(track_id=self._next_id, bbox=bbox, centroid=centroid)
        track.history.append(centroid)
        track.age = 1
        track.hits = 1
        self._tracks[self._next_id] = track
        self._next_id += 1

    def _age_and_kill(self) -> None:
        for tid in list(self._tracks.keys()):
            self._tracks[tid].missed += 1
            self._tracks[tid].age += 1
        self._kill_dead()

    def _kill_dead(self) -> None:
        for tid in list(self._tracks.keys()):
            if self._tracks[tid].missed > self._max_missed:
                self._archive.append(self._tracks[tid])
                del self._tracks[tid]

    def _snapshot(self) -> list[Track]:
        return [t for t in self._tracks.values() if t.missed == 0]

    def collect_finished_tracks(self) -> list[Track]:
        """
        Возвращает все треки за всё время — живые + архивированные.
        Вызывать в конце прогона, перед стадией Decide.
        """
        return list(self._archive) + list(self._tracks.values())


# -------------------------------------------------------------------------
# Smoke-тест: создаём искусственные траектории и проверяем,
# что ID-ы сохраняются.
# -------------------------------------------------------------------------
if __name__ == "__main__":
    tr = CentroidTracker()

    # Два человека идут параллельно
    frame1 = [(100, 100, 30, 80), (500, 100, 30, 80)]
    frame2 = [(105, 110, 30, 80), (495, 110, 30, 80)]
    frame3 = [(110, 120, 30, 80), (490, 120, 30, 80)]

    for i, fr in enumerate([frame1, frame2, frame3]):
        active = tr.update(fr)
        print(f"кадр {i+1}: " + ", ".join(f"id={t.track_id}@{t.centroid}" for t in active))

    all_tracks = tr.collect_finished_tracks()
    print(f"\nвсего треков: {len(all_tracks)}")
    for t in all_tracks:
        print(f"  id={t.track_id}  age={t.age}  history={list(t.history)}")
