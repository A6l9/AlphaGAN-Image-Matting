[English](README.md) | **Русский**

# AlphaGAN для image matting

<p align="center">
  <img src="assets/alpha_gan_demo.gif" alt="Демонстрация AlphaGAN" width="500">
</p>

Неофициальная реализация пайплайна обучения **AlphaGAN** для **image matting** (восстановление alpha matte по изображению и trimap), основанная на статье:  
[AlphaGAN: Generative adversarial networks for natural image matting](https://arxiv.org/pdf/1807.10088)

## Содержание
- [Обзор проекта](#обзор-проекта)
- [Структура репозитория](#структура-репозитория)
- [Установка зависимостей](#установка-зависимостей)
- [Подготовка данных](#подготовка-данных)
- [Конфиги](#конфиги)
  - [Пример конфига](#пример-конфига)
  - [Описание полей конфига](#описание-полей-конфига)
- [Обучение](#обучение)
- [Лоссы и валидация](#лоссы-и-валидация)
- [Чекпоинты и логирование](#чекпоинты-и-логирование)
- [Вход и выход модели](#вход-и-выход-модели)
- [Инференс](#инференс)
- [Ссылки](#ссылки)
- [Лицензия](#лицензия)

---

## Обзор проекта

**AlphaGAN** предназначен для восстановления alpha matte (прозрачности) в неизвестной области trimap с использованием GAN-компонентов и специализированных matting-лоссов.

Этот репозиторий включает:
- архитектуры моделей (генератор, дискриминатор и базовые блоки)
- лоссы и метрики
- пайплайн обучения (train и test)
- пайплайн датасета и трансформаций
- логирование в TensorBoard и чекпоинты

---

## Структура репозитория

```
.
├── assets/                  # Визуальные материалы и превью для README
├── configs/                 # YAML-конфиги для обучения и тестирования
├── losses/                  # Реализации лоссов
├── models/                  # Архитектуры моделей и компоненты
├── train_pipeline/          # Шаги train/test и цикл по эпохам
├── transforms/
│   ├── models/              # Базовые блоки трансформаций
│   └── trans_pipeline.py    # Пайплайн трансформаций для train/test данных
├── cfg_loader.py            # Загрузка конфига из configs/config.yaml
├── dataset.py               # Датасет, распаковка архивов, генерация CSV с разметкой
├── inference.py             # Скрипт инференса через ONNX
├── main.py                  # Точка входа: инициализация и запуск обучения
├── onnx_export.py           # Скрипт экспорта в ONNX
├── schemas.py               # Dataclass-структуры для состояния обучения, лоссов и метрик
└── utils/                   # Seed, логирование, чекпоинты и train-хелперы
```

---

## Установка зависимостей

Проект использует `Python >=3.13` и фреймворк `PyTorch`.

Пример установки через `uv`:

Если `uv` еще не установлен, следуй инструкциям [здесь](https://docs.astral.sh/uv/)

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

Датасеты, использованные для обучения:
- Для foreground-части я создал датасет содержащий 506 объектов.

- Для background-части использовался датасет [BG20K](https://www.kaggle.com/datasets/nguyenquocdungk16hl/bg-20o).

`dataset.py` содержит:
- класс датасета для train/test
- хелперы для распаковки архивов
- генерацию `dataset_labels.csv` с путями к `original`, `trimap`, `mask` и полем `split` (`train` / `test`)

### Ожидаемая структура директорий

Код генерации меток ожидает следующую tiled-структуру датасета:

```text
<root>/
  dataset/
    NAME_OF_DATASET/
      part01/
        sample_0001/
          composite_crops/
          alpha_crops/
          trimap_crops/
        sample_0002/
          composite_crops/
          alpha_crops/
          trimap_crops/
      part02/
        sample_0003/
          composite_crops/
          alpha_crops/
          trimap_crops/
      ...
```

Требования:
- каждая директория `partXX` должна содержать подпапки с сэмплами
- каждая папка сэмпла должна содержать `composite_crops`, `alpha_crops` и `trimap_crops`
- каждый crop должен существовать во всех трех папках с одинаковым именем файла

### Пример использования

1) Распаковка архивов:

```python
from pathlib import Path
from dataset import unpack_archives

dt_path = Path(__file__).parent / "dataset" / "IMDatasetTiled.zip"
dst_path = Path(__file__).parent / "dataset"

unpack_archives(dt_path, dst_path)
```

2) Генерация `dataset_labels.csv`:

```python
from pathlib import Path
from dataset import prepare_dataset_labels

dt_path = Path(__file__).parent / "dataset" / "IMDatasetTiled"
output_path = Path(__file__).parent / "dataset" / "dataset_labels.csv"

prepare_dataset_labels(dt_path, output_path)
```

---

## Конфиги

Конфиги используют формат `.yaml`. Активный конфиг загружается из `configs/config.yaml` через `cfg_loader.py`.

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
  resize_size: 256
  D:
    update_n_batches: 1
    scheduler:
      start_lr: 1e-5
      end_lr: 1.5e-4
      step_size_up: 4000
  G:
    scheduler:
      start_lr: 1e-4
      end_lr: 3e-3
      step_size_up: 4000
  optimizer:
    weight_decay: 5e-4
  logging:
    log_io_n_batches: 10
    log_lr_n_batches: 50
    log_curr_loss_n_batches: 10
    log_grad_n_batches: 10
    log_random_weights_n_batches: 10
  epoches: 5000
  losses:
    lambda_gan_g: 0.15
    alpha_loss:
      lambda_alpha_g: 1.0
      use_weighted_option: 1
      unknown_weight: 4
      bg_weight: 4
      fg_weight: 4
    compos_loss:
      lambda_comp_g: 1.0
      use_weighted_option: 1
      unknown_weight: 4
      fg_weight: 4
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
  Параметры нормализации ImageNet, используемые генератором и VGG-экстрактором признаков.
- `batch_size`  
  Размер батча.
- `checkpoints_dir`  
  Директория для сохранения чекпоинтов.
- `log_dir`  
  Директория для логов TensorBoard.
- `colab.use_colab`  
  Включает режим, удобный для Colab и Google Drive, что полезно при ограниченном дисковом пространстве.
- `colab.best_chkp_name`, `colab.last_chkp_name`  
  Базовые имена файлов для лучшего и последнего чекпоинтов.

#### `train`
- `use_gan_loss`  
  Включает GAN loss. Если `0`, adversarial-компонент отключается.
- `save_chkp_n_epoches`  
  Сохранять чекпоинт каждые `N` эпох.
- `resize_size`  
  Размер стороны при resize в train pipeline трансформаций.
- `D.update_n_batches`  
  Обновлять дискриминатор один раз на каждые `N` батчей.
- `D.scheduler`, `G.scheduler`  
  Параметры scheduler: `start_lr`, `end_lr`, `step_size_up`.
- `optimizer.weight_decay`  
  Weight decay для оптимизатора.
- `logging`  
  Настройки логирования TensorBoard.
  - `log_io_n_batches`  
    Логировать примеры входов и выходов каждые `N` батчей.
  - `log_lr_n_batches`  
    Логировать learning rate каждые `N` батчей.
  - `log_curr_loss_n_batches`  
    Логировать текущие loss-значения каждые `N` батчей.
  - `log_grad_n_batches`  
    Логировать градиенты каждые `N` батчей.
  - `log_random_weights_n_batches`  
    Логировать случайно выбранные веса модели каждые `N` батчей.
- `epoches`  
  Количество эпох обучения.
- `losses`  
  Веса для компонентов generator loss:
  - `lambda_gan_g` вес GAN-компонента для G
  - `lambda_alpha_g` вес alpha loss (ошибка между GT и предсказанной alpha)
  - `alpha_loss.use_weighted_option` включает взвешенный alpha loss
  - `alpha_loss.unknown_weight` вес unknown-области в alpha loss
  - `alpha_loss.bg_weight` вес background-области в alpha loss
  - `alpha_loss.fg_weight` вес foreground-области в alpha loss
  - `lambda_comp_g` вес composition loss (ошибка между GT composite и composite, построенным по предсказанной alpha)
  - `compos_loss.use_weighted_option` включает взвешенный composition loss
  - `compos_loss.unknown_weight` вес unknown-области в composition loss
  - `compos_loss.fg_weight` вес foreground-области в composition loss
- `amp`  
  Настройки mixed precision (automatic mixed precision).
  - `use_amp`  
    Включает AMP. Если `0`, обучение идет в fp32.
  - `dtype`  
    Тип autocast, `bf16` или `fp16`.
  - `use_grad_scaler`  
    Использовать gradient scaling. Обычно нужен для `fp16`, для `bf16` часто не обязателен.

#### `test`
- `resize_size`  
  Размер resize при тестировании.
- `logging.log_curr_mets_n_batches`  
  Частота логирования метрик во время тестирования: один раз на каждые `N` батчей.
- `logging.log_io_n_batches`  
  Логировать примеры входов и выходов каждые `N` батчей.

Примечания:
- Для обучения используются `AdamW` и `CyclicLR`.
- Путь к конфигу задается в `cfg_loader.py`.

---

## Обучение

`main.py` сейчас использует жестко заданный путь к CSV с метками в блоке `__main__`:

```python
if __name__ == "__main__":
    csv_path = Path(__file__).parent / "dataset" / "dataset_labels.csv"
    main(csv_path)
```

Обнови этот путь, если файл с метками находится в другом месте.

Запуск через `uv`:

```bash
uv run python main.py
```

Или из активированного виртуального окружения:

```bash
python main.py
```

---

## Лоссы и валидация

Текущий код обучения использует следующие компоненты loss:

- `LAlphaLoss` для supervision alpha matte
- `LCompositeLoss` для compositional consistency
- `GANLoss` для adversarial-обучения с дискриминатором PatchGAN
- `PerceptualLoss` с VGG-экстрактором признаков для валидации и мониторинга

В текущей реализации дискриминатор получает:

- **real**: целевое composite RGB-изображение с trimap в качестве 4-го канала
- **fake**: сгенерированное composite RGB-изображение с trimap в качестве 4-го канала

Во время валидации сейчас логируются:

- alpha loss
- compositional loss
- perceptual loss

Лучший чекпоинт выбирается по validation perceptual loss в `train_pipeline/train_main.py`.

## Чекпоинты и логирование

- чекпоинты сохраняются в `general.checkpoints_dir`
- логи TensorBoard записываются в `general.log_dir`

Запуск TensorBoard:

```bash
tensorboard --logdir=./tb_logger --bind_all --samples_per_plugin "images=1000, scalars=100000"
```

## Вход и выход модели

Модель принимает **4-канальный входной тензор**:

- `RGB` изображение
- `trimap` как 4-й канал

Trimap выступает как явный spatial prior, который помечает известный background, известный foreground и unknown-области.

Модель предсказывает **одноканальную alpha matte** для входного изображения.

## Примеры выхода

<p align="center">
  <img src="assets/readme_preview.png" alt="" width="900">
</p>

---

## Инференс

ONNX-чекпоинт можно скачать [здесь](https://drive.google.com/file/d/1VEr-O4uWJzuRTkdZn7tTlNjSaNf3qrkq/view?usp=sharing).

Репозиторий также включает:

- [inference.py](./inference.py) для инференса через ONNX Runtime
- примеры входных данных в [inference_test](./inference_test/)

По умолчанию `inference.py`:

- загружает ONNX-модель из корня репозитория
- читает `inference_test/orig.png` и `inference_test/trimap.png`
- выполняет тайловый инференс по unknown-областям trimap
- записывает файлы `*-alpha.png` и `*-cutout.png` в `results/`

Запуск:

```bash
python inference.py
```

При необходимости отредактируй константы в блоке `__main__` файла `inference.py`, чтобы указать свою ONNX-модель и входные изображения.


## Ссылки
- Sebastian Lutz, Konstantinos Amplianitis, Aljosa Smolic. "AlphaGAN: Generative adversarial networks for natural image matting." arXiv:1807.10088, 2018.

---

## Лицензия
Этот проект распространяется под лицензией MIT. См. `LICENSE`.
