"""
Стадия 4 пайплайна: DETECT — превращение очищенной маски в bounding-box'ы людей.

Что приходит на вход:
    Чистая бинарная маска от стадии Clean. В ней:
      — большинство компонент = ровно один человек (одна → один bbox);
      — некоторые компоненты = группа из 2–4 человек, которые слиплись
        в одно пятно (одна → несколько bbox через watershed-сплит).

Что должно получиться:
    Список bbox-ов (x, y, w, h) для текущего кадра. Этот список
    идёт на вход CentroidTracker, который присваивает каждому bbox
    стабильный ID.

Используемые методы (классический OpenCV):
    - cv2.connectedComponentsWithStats
    - cv2.distanceTransform       — для нахождения «центров» людей в группе
    - cv2.connectedComponents     — на пиках distance transform
    - cv2.watershed               — разделение группы на отдельных людей

Watershed-сплит — стандартный приём, когда в одной маске сидят несколько
объектов одного класса (классический пример из OpenCV — «разделение
монет»). Здесь — разделение «слипшихся» людей.

Автор стадии: Lead CV Engineer.
"""

from __future__ import annotations

import cv2
import numpy as np


# -------------------------------------------------------------------------
# Параметры стадии.
# -------------------------------------------------------------------------
# Компонента площадью больше SPLIT_AREA_THRESHOLD считается «группой»
# и идёт через watershed-сплит. Меньше — сразу bbox без сплита.
# Поднято: один человек сверху ≈ 2000–8000 px², поэтому пятна до 22000
# чаще всего ещё ОДИН человек (раздутый морфологией), а не группа.
# Слишком низкий порог приводил к ложному дроблению (det>clean).
SPLIT_AREA_THRESHOLD: int = 22000

# Параметры distance transform / поиска пиков для watershed.
# Пик ищется как «локальный максимум distance transform >= коэф * глобального».
# Поднято с 0.45 до 0.55: более высокий порог → меньше ложных «зёрен»,
# watershed дробит только реально слипшихся людей с двумя явными ядрами.
PEAK_REL_THRESHOLD: float = 0.55

# Минимальная площадь bbox после сплита. Меньше — считаем шумом.
# Поднято: осколки watershed площадью <800 px² — это не отдельные люди.
MIN_BBOX_AREA_AFTER_SPLIT: int = 800


def detect(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    """
    Превращает очищенную маску в список bbox-ов отдельных людей.

    Параметры
    ---------
    mask : np.ndarray (H, W), uint8
        Бинарная маска от стадии Clean.

    Возвращает
    ----------
    list[(x, y, w, h)] — список bbox-ов отдельных людей.
    """
    if mask is None or mask.ndim != 2:
        raise ValueError("detect: ожидалась 2D бинарная маска")

    binary = (mask > 0).astype(np.uint8) * 255

    # Разбираем маску на связные компоненты
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )

    bboxes: list[tuple[int, int, int, int]] = []

    for label_id in range(1, num_labels):
        x, y, w, h, area = stats[label_id]

        # Компонента этого label_id
        component = (labels == label_id).astype(np.uint8) * 255

        if area <= SPLIT_AREA_THRESHOLD:
            # Одиночка — bbox без изменений
            bboxes.append((int(x), int(y), int(w), int(h)))
        else:
            # Группа — пытаемся разделить через watershed
            split_boxes = _split_by_watershed(component, (x, y, w, h))
            if not split_boxes:
                # Не получилось разделить — оставляем как один bbox
                bboxes.append((int(x), int(y), int(w), int(h)))
            else:
                bboxes.extend(split_boxes)

    return bboxes


def _split_by_watershed(
    component_mask: np.ndarray,
    bbox: tuple[int, int, int, int],
) -> list[tuple[int, int, int, int]]:
    """
    Разделяет одну большую компоненту на несколько bbox через watershed.

    Алгоритм:
        1) Distance transform — каждой точке внутри маски присваивается
           расстояние до ближайшего фона. У человека «пик» distance будет
           примерно в центре его тела.
        2) Порог по пикам → получаем «зёрна» (sure foreground).
        3) Метки зёрен + неизвестная область → watershed.
        4) Каждая итоговая метка → отдельный bbox.

    Если зёрен < 2 — функция возвращает [], и снаружи bbox остаётся одним.
    """
    x, y, w, h = bbox
    # Работаем в малой ROI, чтобы было быстро
    roi = component_mask[y:y + h, x:x + w].copy()

    # 1) Distance transform
    dist = cv2.distanceTransform(roi, cv2.DIST_L2, 5)
    if dist.max() <= 0:
        return []

    # 2) Зёрна — там, где dist выше порога
    _, sure_fg = cv2.threshold(
        dist, PEAK_REL_THRESHOLD * dist.max(), 255, cv2.THRESH_BINARY
    )
    sure_fg = sure_fg.astype(np.uint8)

    # Сколько отдельных «зёрен» получилось?
    n_seeds, seed_labels = cv2.connectedComponents(sure_fg)
    # n_seeds включает фон → реальных зёрен n_seeds - 1
    if n_seeds - 1 < 2:
        return []  # ровно одно зерно — это один человек

    # 3) Готовим маркеры для watershed:
    #    фон = 1, зёрна = 2..N+1, неизвестно = 0
    sure_bg = cv2.dilate(roi, np.ones((3, 3), np.uint8), iterations=2)
    unknown = cv2.subtract(sure_bg, sure_fg)

    markers = seed_labels.astype(np.int32) + 1
    markers[unknown > 0] = 0

    # watershed работает на BGR-картинке
    roi_bgr = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
    cv2.watershed(roi_bgr, markers)
    # markers теперь содержит: -1 = граница, 1 = фон, 2..N+1 = объекты

    # 4) Для каждой метки объекта — bbox в координатах исходного кадра
    result: list[tuple[int, int, int, int]] = []
    unique_labels = [
        ml for ml in np.unique(markers) if ml > 1
    ]  # пропускаем -1 (граница) и 1 (фон)

    for ml in unique_labels:
        obj_mask = (markers == ml).astype(np.uint8) * 255
        if obj_mask.sum() == 0:
            continue
        ys, xs = np.where(obj_mask > 0)
        if len(xs) == 0:
            continue
        bx, by = int(xs.min()), int(ys.min())
        bw, bh = int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)
        if bw * bh < MIN_BBOX_AREA_AFTER_SPLIT:
            continue
        # Сдвигаем в координаты всего кадра
        result.append((x + bx, y + by, bw, bh))

    return result


# -------------------------------------------------------------------------
# Smoke-тест. Запуск:
#     python -m src.detect
# -------------------------------------------------------------------------
if __name__ == "__main__":
    from pathlib import Path

    from src.enhance import enhance
    from src.segment import Segmenter
    from src.clean import clean

    paths = sorted(Path("images").glob("frame_*.PNG"))
    frames = [enhance(cv2.imread(str(p))) for p in paths]

    seg = Segmenter()
    seg.warmup(frames)

    masks = []
    for f in frames:
        _, _, c = seg.segment(f)
        masks.append(clean(c))

    mid = 20
    cleaned = masks[mid]
    bboxes = detect(cleaned)
    print(f"кадр {paths[mid].name}: {len(bboxes)} bbox после detect")

    # Сравним с количеством компонент ДО сплита
    n, _, _, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
    print(f"  компонент до сплита: {n - 1}")
    print(f"  bbox после сплита:   {len(bboxes)}")
    print(f"  добавлено сплитом:   {len(bboxes) - (n - 1)}")

    # Визуализация
    orig = cv2.imread(str(paths[mid]))
    vis = orig.copy()
    for (x, y, w, h) in bboxes:
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.rectangle(vis, (0, 0), (vis.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(vis, f"frame {mid}/{len(frames)-1}  people detected: {len(bboxes)}",
                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.imwrite("outputs/detect_demo.png", vis)
    print("Сохранено: outputs/detect_demo.png")
