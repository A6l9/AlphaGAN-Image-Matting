[English](README.md) | **Русский**

# AlphaGAN for Image Matting

Неофициальная реализация тренировочного пайплайна **AlphaGAN** для задачи **image matting** (восстановления альфа канала по изображению и trimap) по статье:  
[AlphaGAN: Generative adversarial networks for natural image matting](https://arxiv.org/pdf/1807.10088)

## Содержание
- [Коротко о проекте](#коротко-о-проекте)
- [Структура репозитория](#структура-репозитория)
- [Установка зависимостей](#установка-зависимостей)
- [Подготовка данных](#подготовка-данных)
- [Configs](#configs)
  - [Пример конфига](#пример-конфига)
  - [Описание полей](#описание-полей-конфига)
- [Запуск обучения](#запуск-обучения)
- [Чекпоинты и логирование](#чекпоинты-и-логирование)
- [References](#references)
- [License](#license)

---

## Коротко о проекте

**AlphaGAN** решает задачу восстановления альфа канала (прозрачности) в неизвестной области trimap, используя GAN компоненты и специализированные лоссы для matting.

В этом репозитории реализованы:
- архитектуры моделей (generator, discriminator и вспомогательные блоки)
- лоссы и метрики
- тренировочный пайплайн (train и test)
- датасет пайплайн и трансформации
- логирование (TensorBoard) и чекпоинты

---

## Структура репозитория

```
.
├── configs/                 # YAML конфиги для тренировки и тестирования
├── losses/                  # Реализации всех лоссов и метрик проекта
├── models/                  # Архитектуры моделей и их компоненты
├── train_pipeline/          # Train и test шаги, цикл по эпохам
├── transforms/
│   ├── models/              # Кастомные трансформации
│   └── trans_pipeline.py    # Пайплайн трансформаций для train/test данных
├── cfg_loader.py            # Загрузка конфигов из папки configs
├── dataset.py               # Датасет, распаковка архивов, генерация CSV с метками
├── main.py                  # Точка входа: инициализация и запуск обучения
├── schemas.py               # Датаклассы для batch структур, лоссов, метрик и компонентов обучения
└── utils.py                 # Seed, логирование, helpers, создание и загрузка чекпоинтов
```

---

## Установка зависимостей

Проект написан на `python 3.13` и фреймворке `pytorch`

Пример установки при использовании пакетного менеджера `uv`:

Если `uv` еще не установлен, найдите инструкции по установке [здесь](https://docs.astral.sh/uv/)

```bash
uv sync
```
---
Пример установки через `pip`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Подготовка данных

Для обучения использовались датасеты:
- [AIM-500](https://github.com/JizhiziLi/AIM) (foreground, маски и trimap)
- [BG20K](https://www.kaggle.com/datasets/nguyenquocdungk16hl/bg-20o) (background)

`dataset.py` содержит:
- класс датасета для train/test
- функции распаковки архивов
- генерацию `dataset_labels.csv` с путями к `original`, `trimap`, `mask`, `background` и полем `split` (train/test)

### Ожидаемая структура данных

```text
<root>/
  dataset/
    AIM-500/              # fg_path
      mask/
        <id>.(png|jpg|jpeg)
      trimap/
        <id>.(png|jpg|jpeg)
      original/
        <id>.(png|jpg|jpeg)

    BG20K/                # bg_path
      **/*.(png|jpg|jpeg) # вложенные папки допустимы
```

Требования:
- в `AIM-500` должны быть папки `mask`, `trimap`, `original`
- для каждого `<id>` должны существовать файлы с одинаковым именем (stem) во всех трех папках
- в `BG20K` допустима любая вложенность, изображения ищутся рекурсивно по `png/jpg/jpeg`

### Пример использования

1) Распаковка архивов:

```python
from pathlib import Path
from dataset import unpack_archives

fg_zip_path = Path(__file__).parent / "dataset" / "AIM-500-20251030T115928Z-1-001.zip"
bg_zip_path = Path(__file__).parent / "dataset" / "archive.zip"
dst_path = Path(__file__).parent / "dataset"

unpack_archives(fg_zip_path, dst_path)
unpack_archives(bg_zip_path, dst_path)
```

2) Генерация `dataset_labels.csv`:

```python
from pathlib import Path
from dataset import prepare_labels

fg_path = Path(__file__).parent / "dataset" / "AIM-500"
bg_path = Path(__file__).parent / "dataset" / "BG20K"
output_path = Path(__file__).parent / "dataset" / "dataset_labels.csv"

prepare_labels(fg_path, bg_path, output_path)
```

---

## Configs

Конфиги имеют расширение `.yaml`.

### Пример конфига

```yaml
general:
  random_seed: 1669
  mean: [0.485, 0.456, 0.406]
  std: [0.229, 0.224, 0.225]
  batch_size: 5
  checkpoints_dir: checkpoints/
  log_dir: tb_logger/
  colab:
    use_colab: 0
    best_chkp_name: best_chkp
    last_chkp_name: last_chkp

train:
  use_gan_loss: 1
  save_chkp_n_epoches: 1
  crop_size: 256
  D:
    update_n_batches: 1
    scheduler:
      start_lr: 1e-5
      end_lr: 1.5e-4
      step_size_up: 4000
  G:
    scheduler:
      start_lr: 1e-3
      end_lr: 3e-4
      step_size_up: 4000
  optimizer:
    weight_decay: 5e-4
  logging:
    log_io_n_batches: 10
    log_lr_n_batches: 50
    log_curr_loss_n_batches: 1
  epoches: 5000
  losses:
    lambda_gan_g: 0.15
    lambda_alpha_g: 1.0
    lambda_comp_g: 1.0
  amp:
    use_amp: 1
    dtype: bf16
    use_grad_scaler: 0

test:
  resize_size: 256
  logging:
    log_curr_mets_n_batches: 1
    log_io_n_batches: 10
```

### Описание полей конфига

#### `general`
- `random_seed`  
  Seed для воспроизводимости.
- `mean`, `std`  
  Нормализация входного изображения (по умолчанию значения ImageNet).
- `batch_size`  
  Размер батча.
- `checkpoints_dir`  
  Директория для сохранения чекпоинтов.
- `colab.use_colab`  
  Включает режим, удобный для Colab и Google Drive (полезно при ограничениях по диску).
- `colab.best_chkp_name`, `colab.last_chkp_name`  
  Базовые имена файлов для best и last чекпоинтов.

#### `train`
- `use_gan_loss`  
  Включить GAN часть лосса. Если `0`, adversarial компонента отключена.
- `save_chkp_n_epoches`  
  Сохранять чекпоинт каждые `N` эпох.
- `crop_size`  
  Размер кропа вокруг неизвестной (серой) области trimap.
- `D.update_n_batches`  
  Обновлять дискриминатор раз в `N` батчей.
- `D.scheduler`, `G.scheduler`  
  Параметры scheduler: `start_lr`, `end_lr`, `step_size_up`.
- `optimizer.weight_decay`  
  Weight decay для оптимизатора.
- `logging`  
  Настройки логгирования TensorBoard.
  - `log_dir`  
    Папка, куда пишутся логи.
  - `log_io_n_batches`  
    Логгировать примеры входов/выходов каждые `N` батчей.
  - `log_lr_n_batches`  
    Логгировать learning rate каждые `N` батчей.
  - `log_curr_loss_n_batches`  
    Логгировать текущие значения лоссов каждые `N` батчей.
- `epoches`  
  Количество эпох обучения.
- `losses`  
  Веса компонентов взвешенной суммы лоссов генератора:
  - `lambda_gan_g` вес GAN компоненты для G
  - `lambda_alpha_g` вес alpha loss (ошибка между gt и предсказанной альфой)
  - `lambda_comp_g` вес composition loss (ошибка между gt композитом и композитом сделаным с использованием предсказанной альфы)
- `amp`  
  Настройки mixed precision (автоматическая смешанная точность).
  - `use_amp`  
    Включить AMP. Если `0`, обучение идет в fp32.
  - `dtype`  
    Тип для autocast, поддерживается `bf16` или `fp16`.
  - `use_grad_scaler`  
    Использовать gradient scaling. Обычно нужен для `fp16`, для `bf16` часто можно отключать.

#### `test`
- `crop_size`  
  Размер кропа вокруг неизвестной (серой) области trimap(аналогично train.crop_size).
- `logging.log_curr_mets_n_batches`  
  Частота логгирования метрик во время теста: раз в `N` батчей.
- `logging.log_io_n_batches`  
  Логгировать примеры входов/выходов каждые `N` батчей.

Примечания:
- Для обучения используются `AdamW` и `CyclicLR`.
- Путь до конфига задается в `cfg_loader.py`.

---

## Запуск обучения

В `main.py` укажите путь до файла с метками:

```python
csv_path = Path(__file__).parent / "dataset" / "dataset_labels.csv"
main(csv_path)
```

Запуск:

```bash
python main.py
```

---

## Чекпоинты и логирование

- чекпоинты сохраняются в `general.checkpoints_dir`
- логи TensorBoard пишутся в `train.logging.log_dir`

Запуск TensorBoard:

```bash
tensorboard --logdir=./tb_logger --bind_all --samples_per_plugin "images=1000, scalars=100000"
```

---

## References
- Sebastian Lutz, Konstantinos Amplianitis, Aljosa Smolic. ''AlphaGAN: Generative adversarial networks for natural image matting.'' arXiv:1807.10088, 2018.

---

## License
Этот проект находится под лицензией MIT. См. файл `LICENSE`.
