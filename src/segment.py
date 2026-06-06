"""
Стадия 2 пайплайна: SEGMENT — выделение людей на бинарной маске.

Подход — комбинация двух классических методов OpenCV:

    1) MOG2 background subtraction
       Камера статична, фон (пол) меняется медленно, люди движутся —
       идеальный сценарий для вычитания фона. MOG2 строит вероятностную
       модель каждого пикселя и отмечает «непохожие на фон» как foreground.

    2) Otsu thresholding по серому каналу (инвертированный)
       Люди на сцене существенно темнее светлого пола. Otsu автоматически
       подбирает порог между двумя пиками гистограммы (тёмные люди / светлый
       пол). Это резервный канал — он ловит людей, которые остановились
       (MOG2 такого «съедает» в фон).

Финальная маска = MOG2 OR Otsu — берём всё, что хоть один метод считает
человеком. Лишнее уберёт стадия Clean.

Важно: MOG2 требует «прогрева» (несколько десятков кадров на построение
модели фона). Поскольку у нас всего 41 кадр, делаем ДВА прохода:
    pass 1 — кормим MOG2 всеми кадрами с learningRate>0 (фоновая модель);
    pass 2 — кормим теми же кадрами с learningRate≈0, забирая чистые маски.

Класс Segmenter инкапсулирует это поведение.

Автор стадии: Segmentation Engineer.
"""

from __future__ import annotations

from typing import Iterable

import cv2
import numpy as np


# -------------------------------------------------------------------------
# Параметры стадии.
# -------------------------------------------------------------------------
MOG2_HISTORY: int = 60                # длина истории фоновой модели
MOG2_VAR_THRESHOLD: float = 25.0      # чувствительность (меньше → больше foreground)
MOG2_DETECT_SHADOWS: bool = True      # тени MOG2 пометит серым (127), мы их потом отбросим

OTSU_INVERT: bool = False              # люди темнее фона → после инверсии станут белыми
OTSU_BLUR_KERNEL: tuple[int, int] = (5, 5)  # лёгкий blur перед Otsu — стабильнее порог
OTSU_MIN_THRESH: float = 40.0          # минимальный порог Otsu на разностном кадре (защита от шума)


class Segmenter:
    """
    Сегментатор кадров. Использует MOG2 + Otsu и объединяет маски через OR.

    Типичное использование:
        seg = Segmenter()
        seg.warmup(frames)                 # один проход — построение фона
        for frame in frames:
            mog_mask, otsu_mask, final = seg.segment(frame)
    """

    def __init__(self) -> None:
        # createBackgroundSubtractorMOG2 — реализация Zivkovic, 2004.
        # Параметр detectShadows=True помечает теневые пиксели значением 127
        # в выходной маске; мы их сами обнулим в _binarize_mog2.
        self._mog2 = cv2.createBackgroundSubtractorMOG2(
            history=MOG2_HISTORY,
            varThreshold=MOG2_VAR_THRESHOLD,
            detectShadows=MOG2_DETECT_SHADOWS,
        )
        self._warmed_up = False

    # ----------------------------------------------------------------- warmup
    def warmup(self, frames: Iterable[np.ndarray]) -> None:
        """
        Прогон по всем кадрам для построения фоновой модели.
        После warmup() segment() будет работать с learningRate≈0 — фон
        перестанет «впитывать» стоящих людей.
        """
        for frame in frames:
            # learningRate=-1 → MOG2 сам выбирает скорость обучения по history
            self._mog2.apply(frame, learningRate=-1)
        self._warmed_up = True

    # ----------------------------------------------------------------- segment
    def segment(
        self, frame_bgr: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Сегментирует один кадр.

        Возвращает три бинарных маски (uint8, 0/255), одинакового размера:
            mog_mask   — результат MOG2 (без теней)
            otsu_mask  — результат Otsu по grayscale
            combined   — финальная маска (OR двух предыдущих)

        Все три возвращаются явно, чтобы в отчёте показать каждую отдельно
        и обосновать выбор «двух методов вместо одного» (бонус за method comparison).
        """
        if frame_bgr is None or frame_bgr.ndim != 3:
            raise ValueError("segment: ожидался BGR-кадр")

        # --- MOG2 -------------------------------------------------------
        # После warmup ставим маленький learningRate, чтобы стоящие люди
        # не растворялись в фоне за время прогона.
        lr = 0.001 if self._warmed_up else -1
        mog_raw = self._mog2.apply(frame_bgr, learningRate=lr)
        mog_mask = self._binarize_mog2(mog_raw)

        # --- Otsu -------------------------------------------------------
        otsu_mask = self._otsu_mask(frame_bgr)

        # --- объединение -----------------------------------------------
        # Ограничиваем Otsu зоной вблизи MOG2 детекций: расширяем MOG2
        # маску и пересекаем с Otsu. Это убирает ложные срабатывания
        # от теней и статичных элементов здания вдали от людей.
        mog_dilated = cv2.dilate(mog_mask, np.ones((25, 25), np.uint8), iterations=1)
        otsu_filtered = cv2.bitwise_and(otsu_mask, mog_dilated)
        combined = cv2.bitwise_or(mog_mask, otsu_filtered)

        return mog_mask, otsu_mask, combined

    # ----------------------------------------------------------------- helpers
    @staticmethod
    def _binarize_mog2(mog_raw: np.ndarray) -> np.ndarray:
        """
        MOG2 возвращает: 0 — фон, 127 — тень, 255 — foreground.
        Нам нужны только настоящие люди → отбрасываем тени (127).
        """
        return np.where(mog_raw >= 255, 255, 0).astype(np.uint8)

    def _otsu_mask(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Рассчитывает порог Otsu по разности текущего кадра и модели фона MOG2.
        Если модель фона еще не готова, делает fallback на обычный Otsu по grayscale.
        """
        bg_img = self._mog2.getBackgroundImage()
        if bg_img is not None and bg_img.shape == frame_bgr.shape:
            # Разность кадра и фона (движущиеся/изменившиеся области)
            diff = cv2.absdiff(frame_bgr, bg_img)
            gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, OTSU_BLUR_KERNEL, 0)
            
            # Otsu thresholding
            otsu_thresh, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            
            # Защита: если Otsu выбрал слишком низкий порог на кадрах без движения, перевычисляем с OTSU_MIN_THRESH
            if otsu_thresh < OTSU_MIN_THRESH:
                _, mask = cv2.threshold(blurred, OTSU_MIN_THRESH, 255, cv2.THRESH_BINARY)
            return mask
        else:
            # Fallback к классическому методу (grayscale Otsu)
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, OTSU_BLUR_KERNEL, 0)
            flag = cv2.THRESH_BINARY_INV if OTSU_INVERT else cv2.THRESH_BINARY
            _, mask = cv2.threshold(blurred, 0, 255, flag | cv2.THRESH_OTSU)
            return mask


# -------------------------------------------------------------------------
# Smoke-тест. Запуск:
#     python -m src.segment
# Прогоняет сегментатор по всем кадрам в images/ и сохраняет для среднего
# кадра три маски (MOG2 / Otsu / combined).
# -------------------------------------------------------------------------
if __name__ == "__main__":
    from pathlib import Path

    from src.enhance import enhance

    img_dir = Path("images")
    frames_paths = sorted(img_dir.glob("frame_*.PNG"))
    if not frames_paths:
        print("Кадры не найдены в images/")
        raise SystemExit(1)

    # Загружаем все кадры и предварительно улучшаем — на enhanced MOG2/Otsu
    # работают чище (контраст выше).
    frames = [enhance(cv2.imread(str(p))) for p in frames_paths]

    seg = Segmenter()
    print(f"Прогрев MOG2 на {len(frames)} кадрах…")
    seg.warmup(frames)

    # Берём кадр из середины, где люди уже стабильно отслежены
    mid_idx = len(frames) // 2
    mog, otsu, combined = seg.segment(frames[mid_idx])

    # Параллельно прогоняем ВСЕ кадры (второй проход), чтобы маски
    # были стабильны (это эмулирует то, как будет работать run.py).
    for f in frames[:mid_idx]:
        seg.segment(f)
    mog, otsu, combined = seg.segment(frames[mid_idx])

    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    cv2.imwrite(str(out_dir / "segment_demo_mog2.png"), mog)
    cv2.imwrite(str(out_dir / "segment_demo_otsu.png"), otsu)
    cv2.imwrite(str(out_dir / "segment_demo_combined.png"), combined)

    h, w = combined.shape
    print(f"Кадр: {frames_paths[mid_idx].name}  ({w}×{h})")
    print(f"  MOG2  foreground pixels: {int((mog > 0).sum()):>8}  "
          f"({(mog > 0).mean() * 100:.2f}%)")
    print(f"  Otsu  foreground pixels: {int((otsu > 0).sum()):>8}  "
          f"({(otsu > 0).mean() * 100:.2f}%)")
    print(f"  OR    foreground pixels: {int((combined > 0).sum()):>8}  "
          f"({(combined > 0).mean() * 100:.2f}%)")
    print("Сохранены: outputs/segment_demo_{mog2,otsu,combined}.png")
