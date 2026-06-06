"""
Стадия 3 пайплайна: CLEAN — морфологическая чистка маски сегментации.

Что приходит на вход:
    Маска от Segmenter — белые пятна там, где могут быть люди. В ней есть
    мусор: декоративные круги на полу, отдельные пиксели шума, тени,
    разорванные силуэты людей.

Что должно получиться:
    Маска, в которой остались ТОЛЬКО силуэты людей. Это самая важная
    стадия для итоговой точности подсчёта — здесь мы убираем ложные
    срабатывания и склеиваем разорванные силуэты.

Используемые методы (всё классический OpenCV):
    - cv2.morphologyEx (MORPH_OPEN / MORPH_CLOSE)  — морфология
    - cv2.connectedComponentsWithStats             — анализ компонент
    - геометрические фильтры:
        * по площади        — отсечь шум и слишком крупные пятна
        * по circularity    — отсечь декоративные круги на полу
        * по aspect ratio   — оставить вертикальные силуэты людей

Автор стадии: Morphology & Report Lead.
"""

from __future__ import annotations

import cv2
import numpy as np


# -------------------------------------------------------------------------
# Параметры стадии. Подобраны под кадры 1280×720 с верхней камерой:
# люди занимают примерно 20×40 … 60×120 пикселей.
# -------------------------------------------------------------------------
OPEN_KERNEL_SIZE: int = 3       # ядро для удаления соль-перец шума
CLOSE_KERNEL_SIZE: int = 15     # ядро для склейки разорванных силуэтов (увеличено для лучшего склеивания)

# Region of Interest — режем только узкую нижнюю полосу (артефакт компрессии).
# Верх НЕ режем: там стоят дальние люди (горизонт сцены).
ROI_TOP_CROP: int = 0
ROI_BOTTOM_CROP: int = 10

# Точечные «вырезы» — прямоугольники, которые гарантированно НЕ люди.
# Сейчас пусто: проверка показала, что почти все «постоянно тёмные»
# зоны в верхней части кадра — это дальние стоящие люди.
# Если на конкретной сцене появится статичный артефакт (логотип, бордюр),
# добавьте его сюда в формате (x1, y1, x2, y2).
DEAD_ZONES: tuple[tuple[int, int, int, int], ...] = ()

MIN_AREA: int = 400             # меньше — это шум; дальние люди ~400–800 px²
MAX_AREA: int = 30000           # пропускаем большие группы (split — задача detect)

# Минимальная высота — дальние «ноги» в кадре имеют ~25–30 px.
MIN_HEIGHT: int = 20

# Circularity = 4π·A / P² ∈ (0, 1]. Декоративные круги: > 0.78.
MAX_CIRCULARITY: float = 0.72

# Aspect ratio = max(h, w) / min(h, w). Включает дальних (почти квадратные).
MIN_ASPECT: float = 0.90
MAX_ASPECT: float = 7.0


# -------------------------------------------------------------------------
# Маска статичных артефактов. Строится по серии масок (один раз перед
# прогоном) и используется как «вечный фон» — то, что неподвижно сидит
# в маске Otsu во всех кадрах. Типичные примеры: бордюры, виньетирование,
# постоянно тёмные точки на полу.
# -------------------------------------------------------------------------
def build_static_mask(masks: list[np.ndarray], threshold: float = 0.6) -> np.ndarray:
    """
    Строит бинарную маску пикселей, которые присутствуют как foreground
    в более чем `threshold` доле кадров. Эти пиксели — статика, не люди.

    Параметры
    ---------
    masks : список бинарных масок (одного размера, uint8 0/255)
    threshold : доля кадров (0..1), при превышении которой пиксель считается статичным

    Возвращает
    ----------
    np.ndarray (H, W) uint8 — маска статики (0/255)
    """
    if not masks:
        raise ValueError("build_static_mask: пустой список масок")
    stack = np.stack([(m > 0).astype(np.uint8) for m in masks], axis=0)
    presence = stack.mean(axis=0)  # доля «foreground» в каждом пикселе
    static = (presence > threshold).astype(np.uint8) * 255
    # Слегка раздуем, чтобы захватить рамку артефакта целиком
    static = cv2.dilate(static, np.ones((5, 5), np.uint8), iterations=1)
    return static


def clean(mask: np.ndarray, static_mask: np.ndarray | None = None) -> np.ndarray:
    """
    Чистит бинарную маску от шума и мусора, оставляет только людей.

    Параметры
    ---------
    mask : np.ndarray (H, W), uint8
        Бинарная маска (0 / 255) от стадии Segment.

    Возвращает
    ----------
    np.ndarray (H, W), uint8
        Очищенная бинарная маска того же размера.

    Шаги
    -----
    1. MORPH_OPEN  — убирает одиночные белые пиксели (шум).
    2. MORPH_CLOSE — склеивает разрывы внутри силуэтов людей
       (например, тёмная куртка + светлая голова → один силуэт).
    3. connectedComponentsWithStats — нумерует все связные компоненты.
    4. Для каждой компоненты применяем геометрические фильтры:
       — площадь, форма, aspect ratio.
    5. Возвращаем маску только из «прошедших» компонент.
    """
    if mask is None or mask.ndim != 2:
        raise ValueError("clean: ожидалась 2D бинарная маска")

    binary = (mask > 0).astype(np.uint8) * 255

    # --- 0a) ROI: занулить верхнюю/нижнюю кромки кадра ---------------
    if ROI_TOP_CROP > 0:
        binary[:ROI_TOP_CROP, :] = 0
    if ROI_BOTTOM_CROP > 0:
        binary[-ROI_BOTTOM_CROP:, :] = 0

    # --- 0b) Точечные dead-zones -------------------------------------
    # Зануляем заранее известные «не-людские» прямоугольники
    # (постоянный артефакт виньетирования и т.п.).
    for (x1, y1, x2, y2) in DEAD_ZONES:
        binary[y1:y2, x1:x2] = 0

    # --- 0c) Опциональное вычитание динамической static_mask ---------
    # Оставлено как опция; в текущем пайплайне не используется.
    if static_mask is not None:
        if static_mask.shape != binary.shape:
            raise ValueError(
                f"clean: static_mask {static_mask.shape} ≠ mask {binary.shape}"
            )
        binary = cv2.bitwise_and(binary, cv2.bitwise_not(static_mask))

    # --- 1) Открытие — снять «соль» ----------------------------------
    open_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (OPEN_KERNEL_SIZE, OPEN_KERNEL_SIZE)
    )
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, open_kernel)

    # --- 2) Закрытие — склеить разрывы силуэтов ----------------------
    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (CLOSE_KERNEL_SIZE, CLOSE_KERNEL_SIZE)
    )
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, close_kernel)

    # --- 3) Анализ связных компонент ---------------------------------
    num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        closed, connectivity=8
    )

    # --- 4) Геометрические фильтры -----------------------------------
    # Соберём финальную маску — копируем только те компоненты, которые прошли.
    result = np.zeros_like(closed)

    # label 0 — это фон, его пропускаем.
    for label_id in range(1, num_labels):
        x, y, w, h, area = stats[label_id]

        # 4.1 площадь
        if area < MIN_AREA or area > MAX_AREA:
            continue

        # 4.1b минимальная высота — отсечь короткие пятна (тени, головы)
        if h < MIN_HEIGHT:
            continue

        # 4.2 circularity (нужен периметр → берём контур этой компоненты)
        component_mask = (labels == label_id).astype(np.uint8) * 255
        contours, _ = cv2.findContours(
            component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            continue
        perimeter = cv2.arcLength(contours[0], closed=True)
        if perimeter <= 0:
            continue
        circularity = 4.0 * np.pi * area / (perimeter * perimeter)
        if circularity > MAX_CIRCULARITY:
            continue  # это, скорее всего, декоративный круг

        # 4.3 aspect ratio
        if w == 0 or h == 0:
            continue
        aspect = max(w, h) / float(min(w, h))
        if aspect < MIN_ASPECT or aspect > MAX_ASPECT:
            continue

        # 4.4 solidity — отсечь «рваные» тени и мусор
        hull = cv2.convexHull(contours[0])
        hull_area = cv2.contourArea(hull)
        if hull_area > 0:
            solidity = area / hull_area
            if solidity < 0.35:
                continue  # слишком «рваная» форма — не человек

        # Все проверки пройдены — переносим компоненту в результат.
        result[labels == label_id] = 255

    return result


# -------------------------------------------------------------------------
# Smoke-тест. Запуск:
#     python -m src.clean
# Прогоняет enhance → segment → clean по всем кадрам и сохраняет картинку
# для среднего кадра: до и после очистки + статистика.
# -------------------------------------------------------------------------
if __name__ == "__main__":
    from pathlib import Path

    from src.enhance import enhance
    from src.segment import Segmenter

    img_dir = Path("images")
    frames_paths = sorted(img_dir.glob("frame_*.PNG"))
    frames = [enhance(cv2.imread(str(p))) for p in frames_paths]

    seg = Segmenter()
    seg.warmup(frames)

    # Прогон по всем кадрам и чистка (DEAD_ZONES сработают автоматически)
    masks_before = []
    masks_after = []
    for f in frames:
        _, _, combined = seg.segment(f)
        masks_before.append(combined)
        masks_after.append(clean(combined))

    mid = len(frames) // 2

    # Сохраним отдельно «грязную» и «чистую» маску
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    cv2.imwrite(str(out_dir / "clean_demo_before.png"), masks_before[mid])
    cv2.imwrite(str(out_dir / "clean_demo_after.png"), masks_after[mid])

    # И «бок о бок» в одном файле — удобно для отчёта
    side = np.hstack([masks_before[mid], masks_after[mid]])
    cv2.imwrite(str(out_dir / "clean_demo_side_by_side.png"), side)

    def count_components(m):
        n, _, _, _ = cv2.connectedComponentsWithStats((m > 0).astype(np.uint8))
        return n - 1  # минус фон

    before_n = count_components(masks_before[mid])
    after_n = count_components(masks_after[mid])
    before_px = int((masks_before[mid] > 0).sum())
    after_px = int((masks_after[mid] > 0).sum())

    print(f"Кадр: {frames_paths[mid].name}")
    print(f"  ДО  очистки: компонент={before_n:>4}  пикселей={before_px:>7}")
    print(f"  ПОСЛЕ      : компонент={after_n:>4}  пикселей={after_px:>7}")
    print(f"  отсеяно компонент: {before_n - after_n}  "
          f"({(1 - after_n / before_n) * 100:.1f}%)")
    print("Сохранены: outputs/clean_demo_{before,after,side_by_side}.png")
