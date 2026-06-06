# People Counting & Flow Direction

Командный проект по курсу Computer Vision (тема №14).
Считаем людей на видео сверху и определяем направление их движения.
Используется **только классический OpenCV** — без нейросетей.

---

## 📂 Структура проекта

```
people-flow/
├── images/             # 41 кадр исходного видео (1280×720, PNG)
├── src/                # модули пайплайна
│   ├── enhance.py      # стадия 1 — улучшение изображения
│   ├── segment.py      # стадия 2 — сегментация людей
│   ├── clean.py        # стадия 3 — морфология и очистка маски
│   ├── detect.py       # стадия 4 — детекция и трекинг
│   ├── decide.py       # стадия 5 — подсчёт и направление
│   ├── tracker.py      # CentroidTracker для присвоения ID
│   └── viz.py          # отрисовка overlay для демо
├── outputs/
│   ├── per_frame/      # 6 PNG для каждого кадра (требование brief)
│   └── final/          # output.mp4, роза направлений, stats.json
├── report/             # PDF-отчёт команды (5–8 страниц)
├── slides/             # слайды презентации (10 минут)
└── run.py              # главный скрипт
```

---

## 🚀 Установка и запуск

```bash
# 1. Создать виртуальное окружение
python3 -m venv .venv
source .venv/bin/activate   # macOS / Linux
# .venv\Scripts\activate    # Windows

# 2. Поставить зависимости
pip install -r requirements.txt

# 3. Запустить пайплайн на всех кадрах
python run.py --save              # 6 PNG на кадр в outputs/per_frame/
python run.py --live              # живое окно с детекцией (q — выход)
python run.py --video             # экспорт outputs/final/output.mp4
python run.py --save --live --video   # всё сразу
```

### 🎞 Свой ролик → кадры

Если хочешь прогнать пайплайн на **своём видео**, сначала нарежь его
на кадры (любой формат: mp4/mov/avi/…):

```bash
# простой случай: всё видео в папку images/
python scripts/video_to_frames.py path/to/video.mp4

# или: брать каждый 5-й кадр, не больше 60, ресайз до 1280 по большей стороне
python scripts/video_to_frames.py path/to/video.mp4 \
    --every 5 --max 60 --resize 1280 --clear

# или: в отдельную папку и стартовать с 30-й секунды
python scripts/video_to_frames.py path/to/video.mp4 \
    --out images_my --start 30 --max 80
```

Полезные параметры:
- `--every N` — брать каждый N-й кадр. Для 30-fps видео `--every 3..6`
  даёт эффективные 5–10 fps, чего хватает для MOG2 и трекинга.
  Если брать **каждый кадр**, люди двигаются между кадрами на 1–2 px,
  MOG2 может «впитать» их в фон.
- `--max N` — ограничить число кадров (быстрее прогон, меньше места).
- `--resize 1280` — ресайз. Все наши параметры в `src/clean.py` подобраны
  под кадр **1280 px по большей стороне**. На сильно других разрешениях
  фильтры по площади / высоте могут потребовать подкрутки.
- `--clear` — очищает папку `--out` перед сохранением (полезно при повторных прогонах).

После нарезки запускаешь обычно: `python run.py --save --video`.

Финальные числа лежат в `outputs/final/stats.json`:
```json
{
  "total_people": 23,
  "dominant_direction": "S",
  "by_sector": {"N": 2, "S": 8, "E": 3, "W": 4, ...}
}
```

---

## 👥 Команда и роли (5 человек)

| # | Роль | Кто отвечает | Стадии |
|---|---|---|---|
| 1 | Lead CV Engineer | _имя_ | Detect + Decide + сборка `run.py` |
| 2 | Image Processing Specialist | _имя_ | Enhance (`src/enhance.py`) |
| 3 | Segmentation Engineer | _имя_ | Segment (`src/segment.py`) |
| 4 | Morphology & Report Lead | _имя_ | Clean + финальный PDF-отчёт |
| 5 | Data & Testing Engineer | _имя_ | Датасет, прогоны, failure cases, слайды |

> Преподавателем согласован состав из 5 человек (стандартный brief — 3–4).
> Каждый участник отвечает за свой модуль и свою часть презентации.

---

## 🧪 Пайплайн (5 стадий по требованию brief)

```
image → enhance → segment → clean → detect → decide
```

| Стадия | Метод | Файл |
|---|---|---|
| Enhance | CLAHE по L-каналу LAB + bilateral denoise | `src/enhance.py` |
| Segment | MOG2 background subtraction + Otsu fallback | `src/segment.py` |
| Clean | Morph open/close, фильтр по площади/форме | `src/clean.py` |
| Detect | Контуры + CentroidTracker (IoU + расстояние) | `src/detect.py` |
| Decide | Уникальный счёт + 8-секторная роза направлений | `src/decide.py` |

---

## 📄 Отчёт и презентация

- **Отчёт** — `report/report.pdf`, 5–8 страниц на русском, по 1 разделу от каждого участника.
- **Презентация** — `slides/presentation.md`, 10 минут (по 2 минуты на участника) + 5 минут Q&A.
- **Contribution Statement** — `report/contribution.md`, подписи всех 5 участников.

---

## ✅ Что покрыто из рубрики (100 баллов)

| Критерий | Баллы | Где |
|---|---|---|
| Pipeline Completeness | 20 | все 5 стадий подключены в `run.py` |
| Technical Implementation | 25 | чистый OpenCV, модули разделены |
| Detection Quality | 15 | трекер по ID, фильтр по площади/форме |
| Decision Logic | 10 | count + dominant direction в `stats.json` |
| Visualization & Output | 10 | 6 PNG на кадр + `output.mp4` + роза |
| Report Quality | 10 | `report/report.pdf` |
| Team Presentation | 5 | `slides/presentation.md` |
| Team Organization | 5 | `report/contribution.md` |
| **Бонус:** real-time demo | +5 | `python run.py --live` |
| **Бонус:** method comparison | +5 | MOG2 vs Otsu в одном пайплайне, см. отчёт |
