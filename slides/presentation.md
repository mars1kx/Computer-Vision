# People Counting & Flow Direction — Презентация
### 10 минут (по 2 мин на участника) + 5 минут Q&A

---

## Слайд 1 — Титул (15 сек, ведёт Lead)
- **People Counting & Flow Direction**
- Тема №14 · Computer Vision Team Project
- Команда из 5 человек:
  - **Lead CV Engineer (Team Lead):** Maksim Ahafonau
  - **Image Processing Specialist:** Aron Shapialevich
  - **Morphology & Report Lead:** Aleksei Karpukovich
  - **Decision & Tracking Engineer:** Nick Sinazhenski
  - **Data & Testing Engineer:** Ulada Malets
- «Считаем людей сверху и направление потока. Только классический OpenCV».

---

## БЛОК 1 — Lead CV Engineer (Maksim Ahafonau, 2 мин)

### Слайд 2 — Motivation & задача
- Зачем: подсчёт посетителей, анализ потоков (ТЦ, площади, транспорт).
- Вход: видео сверху (статичная камера), 65 кадров, 1280×1242.
- Выход: число людей + преобладающее направление (роза 8 секторов).
- Ограничение: **без нейросетей** — только OpenCV.

### Слайд 3 — Архитектура пайплайна
- Схема: `image → enhance → segment → clean → detect → decide`.
- 5 модулей в `src/`, собраны in `run.py`, запуск одной командой.
- Кто за какую стадию отвечает (таблица ролей).

---

## БЛОК 2 — Image Processing Specialist (Aron Shapialevich, 2 мин)

### Слайд 4 — Enhance
- CLAHE по L-каналу LAB + bilateral denoise.
- Картинка: `01_original.png` vs `02_enhanced.png`.

### Слайд 5 — Segment (method comparison — бонус)
- MOG2 (вычитание фона) **+** Otsu по разности с фоном.
- Почему два метода: MOG2 теряет стоящих → Otsu их добирает.
- Картинка: `03_mask.png` + сравнение MOG2 / Otsu / combined.

---

## БЛОК 3 — Morphology & Report Lead (Aleksei Karpukovich, 2 мин)

### Слайд 6 — Clean
- open/close + фильтры: площадь, высота, circularity, aspect, solidity.
- Картинка «до/после»: `outputs/clean_demo_side_by_side.png`.
- Эффект: 80–550 шумовых компонент → стабильно 5–11 силуэтов.

### Слайд 7 — Visualization
- bbox + ID + хвост траектории + стрелка направления.
- Фильтр отрисовки `MIN_HITS_TO_DRAW` — убирает мигающие ложные рамки.
- Картинка: `06_decision.png`.

---

## БЛОК 4 — Decision & Tracking Engineer (Nick Sinazhenski, 2 мин)

### Слайд 8 — Detect + Tracking
- Связные компоненты → bbox; watershed разделяет слипшихся людей.
- CentroidTracker: сопоставление **с предсказанием по скорости**.
- Поле `hits` отличает человека от фантома.

### Слайд 9 — Decide
- Человек = `hits ≥ 10`. Направление = `atan2`(первая→последняя точка).
- **Результат: 15 людей, преобладающее направление SW.**
- Картинка: `outputs/final/direction_rose.png`.

---

## БЛОК 5 — Data & Testing Engineer (Ulada Malets, 2 мин)

### Слайд 10 — Dataset & прогоны (own dataset — бонус)
- Собственный ролик, нарезка `scripts/video_to_frames.py` (каждый 4-й кадр).
- 65 кадров, 6 PNG на кадр в `outputs/per_frame/`.

### Слайд 11 — Failure analysis
- Фантомные ID на низком fps → решено (предсказание + `hits`): 28 → 15.
- Долго стоящие люди → риск выпадения (MOG2 впитывает в фон).
- Слипшиеся группы 4+; тени/блики.

---

## Слайд 12 — LIVE DEMO (ведёт Lead, ~1 мин)
- `python run.py --live` — живое окно с детекцией и ID.
- Показать: рамки, ID, хвосты, стрелки, счётчик `active people`.
- Резерв: проиграть `outputs/final/output.mp4`, если демо не запустится.

---

## Слайд 13 — Итоги и бонусы
- Все 5 стадий ✅, автоматический результат ✅, 6 выходов на кадр ✅.
- Бонусы: method comparison (+5), real-time demo (+5), own dataset (+5).
- Дальше: Hungarian-сопоставление, Kalman, адаптивная заморозка фона.

---

## Тайминг (10 мин)
| Блок | Участник | Время |
|---|---|---|
| Титул + архитектура | Lead (Maksim Ahafonau) | 2:00 |
| Enhance + Segment | Image Processing (Aron Shapialevich) | 2:00 |
| Clean + Visualization | Morphology & Report (Aleksei Karpukovich) | 2:00 |
| Detect/Track + Decide | Decision & Tracking (Nick Sinazhenski) | 2:00 |
| Dataset + Failure + Demo | Data & Testing (Ulada Malets) | 2:00 |
| **Q&A** | Все | 5:00 |
