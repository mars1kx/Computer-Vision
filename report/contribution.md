# Contribution Statement
### People Counting & Flow Direction · Тема №14 · Computer Vision Team Project

---

## 1. Роли и выполненные задачи

| # | Имя | Роль | Конкретные выполненные задачи | Файлы |
|---|---|---|---|---|
| 1 | Maksim Ahafonau | **Lead CV Engineer** | Архитектура пайплайна, стадия Detect, watershed-сплит групп, сборка `run.py` end-to-end, интеграция всех модулей | `src/detect.py`, `run.py` |
| 2 | Aron Shapialevich | **Image Processing Specialist** | Стадия Enhance (CLAHE+bilateral), стадия Segment (MOG2+Otsu), подбор параметров вычитания фона | `src/enhance.py`, `src/segment.py` |
| 3 | Aleksei Karpukovich | **Morphology & Report Lead** | Стадия Clean (морфология, геометрические фильтры), визуализация (`viz.py`), финальный PDF-отчёт, подготовка демо | `src/clean.py`, `src/viz.py`, `report/` |
| 4 | Nick Sinazhenski | **Decision & Tracking Engineer** | CentroidTracker с предсказанием скорости, поле `hits`, стадия Decide, роза направлений | `src/tracker.py`, `src/decide.py` |
| 5 | Ulada Malets | **Data & Testing Engineer** | Сбор и нарезка датасета, прогоны, документирование результатов и failure cases, QA всех стадий | `scripts/video_to_frames.py`, тесты, отчёт §5–6 |

---

## 2. Распределение баллов (Pool)

По правилам брифа:
`Pool = P × N`, где **P** = оценка проекта по рубрике (0–100), **N = 5** (число участников).
Сумма индивидуальных баллов **должна точно равняться Pool**. Каждый получает 0–100.
Итоговая оценка участника = его баллы ÷ 10.

| # | Имя | Индивидуальные баллы (0–100) | Оценка (баллы ÷ 10) |
|---|---|---|---|
| 1 | Maksim Ahafonau | P | P / 10 |
| 2 | Aron Shapialevich | P | P / 10 |
| 3 | Aleksei Karpukovich | P | P / 10 |
| 4 | Nick Sinazhenski | P | P / 10 |
| 5 | Ulada Malets | P | P / 10 |
| | **Сумма** | **= P × 5** | |

---

## 3. Подписи

Подписывая, каждый участник подтверждает, что указанные задачи выполнены им
лично и согласен с распределением баллов выше.

| # | Имя | Подпись | Дата |
|---|---|---|---|
| 1 | Maksim Ahafonau | M. Ahafonau | 20.06.2026 |
| 2 | Aron Shapialevich | A. Shapialevich | 20.06.2026 |
| 3 | Aleksei Karpukovich | A. Karpukovich | 20.06.2026 |
| 4 | Nick Sinazhenski | N. Sinazhenski | 20.06.2026 |
| 5 | Ulada Malets | U. Malets | 20.06.2026 |
