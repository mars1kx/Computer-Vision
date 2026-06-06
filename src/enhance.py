"""
Стадия 1 пайплайна: ENHANCE — улучшение качества кадра.

Зачем эта стадия:
    Исходный кадр почти чёрно-белый, на светлом полу видны декоративные круги,
    блики и слабые тени. Чтобы последующая сегментация надёжно отделила тёмных
    людей от светлого пола, нужно:
      1) увеличить локальный контраст (CLAHE),
      2) подавить мелкий шум плитки, не разрушая силуэты людей (bilateral).

Используемые методы — только классический OpenCV (по требованию brief):
    - cv2.cvtColor (BGR ↔ LAB)
    - cv2.createCLAHE                — адаптивное выравнивание гистограммы
    - cv2.bilateralFilter            — сглаживание, сохраняющее края
    - cv2.LUT                        — гамма-коррекция

Автор стадии: Image Processing Specialist.
"""

from __future__ import annotations

import cv2
import numpy as np


# -------------------------------------------------------------------------
# Параметры стадии. Вынесены в константы, чтобы их было удобно подбирать
# и упоминать в отчёте. Подобраны под сцену с верхней камерой и светлым полом.
# -------------------------------------------------------------------------
CLAHE_CLIP_LIMIT: float = 2.0          # сила усиления контраста (2.0 — мягко)
CLAHE_TILE_GRID: tuple[int, int] = (8, 8)  # размер плиток для CLAHE
BILATERAL_DIAMETER: int = 5            # радиус соседей для bilateral
BILATERAL_SIGMA_COLOR: float = 35.0    # допуск по цвету
BILATERAL_SIGMA_SPACE: float = 35.0    # допуск по пространству
GAMMA: float = 1.10                    # >1 затемняет светлые области (фон-плитку)


# Предвычисленная LUT для гамма-коррекции — быстрее, чем cv2.pow на каждый кадр.
_GAMMA_LUT = np.array(
    [((i / 255.0) ** (1.0 / GAMMA)) * 255 for i in range(256)],
    dtype=np.uint8,
)


def enhance(frame_bgr: np.ndarray) -> np.ndarray:
    """
    Улучшает один кадр.

    Параметры
    ---------
    frame_bgr : np.ndarray (H, W, 3), uint8
        Исходный кадр в формате BGR (как читает cv2.imread).

    Возвращает
    ----------
    np.ndarray (H, W, 3), uint8
        Улучшенный кадр в формате BGR. Размер не меняется.

    Шаги
    -----
    1. BGR → LAB. Канал L хранит яркость, A/B — цвет. Работаем только с L,
       чтобы не сдвинуть цвет.
    2. CLAHE по каналу L. Адаптивное выравнивание гистограммы локально
       поднимает контраст в зонах с близкими яркостями (а у нас как раз
       почти белый пол).
    3. Сборка обратно LAB → BGR.
    4. Bilateral filter — удаляет шум плитки, сохраняя границы людей.
    5. Гамма > 1 — слегка затемняет очень светлый фон, чтобы силуэты
       людей стали относительно ярче по контрасту.
    """
    if frame_bgr is None or frame_bgr.size == 0:
        raise ValueError("enhance: получен пустой кадр")
    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError(
            f"enhance: ожидался BGR-кадр (H,W,3), получено {frame_bgr.shape}"
        )

    # --- 1) BGR → LAB ----------------------------------------------------
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    # --- 2) CLAHE по L ---------------------------------------------------
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID)
    l_eq = clahe.apply(l_channel)

    # --- 3) обратно в BGR -----------------------------------------------
    lab_eq = cv2.merge((l_eq, a_channel, b_channel))
    bgr_eq = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

    # --- 4) bilateral filter --------------------------------------------
    denoised = cv2.bilateralFilter(
        bgr_eq,
        d=BILATERAL_DIAMETER,
        sigmaColor=BILATERAL_SIGMA_COLOR,
        sigmaSpace=BILATERAL_SIGMA_SPACE,
    )

    # --- 5) гамма-коррекция --------------------------------------------
    result = cv2.LUT(denoised, _GAMMA_LUT)

    return result


# -------------------------------------------------------------------------
# Локальный smoke-тест. Запуск:
#     python -m src.enhance images/frame_000000.PNG
# -------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    from pathlib import Path

    if len(sys.argv) < 2:
        print("Использование: python -m src.enhance <путь_к_кадру>")
        sys.exit(1)

    src_path = Path(sys.argv[1])
    img = cv2.imread(str(src_path))
    if img is None:
        print(f"Не удалось прочитать {src_path}")
        sys.exit(1)

    out = enhance(img)
    side_by_side = np.hstack([img, out])

    out_path = Path("outputs") / f"enhance_demo_{src_path.stem}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), side_by_side)
    print(f"Сохранено: {out_path}  (слева — оригинал, справа — enhanced)")
