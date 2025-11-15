# CNN Voice Recognition — Proyecto 1 (IA)

Repositorio para el proyecto de reconocimiento de sonidos/voz usando espectrogramas y redes convolucionales (LeNet-5 adaptado y ResNet18 adaptado). Código, datos procesados y modelos están organizados para generar espectrogramas, entrenar redes y registrar experimentos con Weights & Biases (wandb).

## Contenido

- `proyecto.ipynb`  : Notebook principal con generación de espectrogramas, definición de dataset, modelos (LeNet-5 y ResNet18), bucle de entrenamiento y evaluación.
- `spectrograms/` y `spectrograms_augmented/` : Carpetas con imágenes PNG (mel-spectrogramas) por clase.
- `models/`        : Pesos de modelos entrenados - incluye LeNet5 (con/sin batch norm, con/sin augmentation) y ResNet18 adaptado para audio (.pth / .pt).
- `runs_lenet/`    : Resultados de entrenamiento, CSV resumen `lenet_summary.csv` y checkpoints guardados.
- `wandb/`         : Directorio con datos de ejecuciones locales (si se usa wandb en modo local/offline).

## Requisitos

- Python 3.8+
- Paquetes (recomendado crear un virtualenv):


```bash
pip install torch torchvision torchaudio librosa matplotlib pillow numpy scikit-learn wandb tqdm pandas
```

GPU: si tienes CUDA y versión de PyTorch compatible, las ejecuciones usarán `cuda` automáticamente.

## Generar espectrogramas

El notebook incluye una celda para convertir los archivos `.wav` del dataset ESC-50 en imágenes PNG (mel-spectrogramas en escala de grises). Asegúrate de tener los archivos y el CSV en:

- `data/ESC-50-master/audio/`
- `data/ESC-50-master/meta/esc50.csv`

Ejecuta la celda de "Generar espectrogramas" en `proyecto.ipynb`. También puedes ejecutar el notebook desde la terminal:

```zsh
jupyter nbconvert --to notebook --execute proyecto.ipynb --ExecutePreprocessor.timeout=600
```

Los PNG se guardarán por categoría dentro de `spectrograms/`.

## Entrenamiento

El notebook incluye dos arquitecturas principales:

### Modelo A - LeNet5
Red convolucional clásica adaptada para espectrogramas de 224x224 píxeles con 1 canal (grayscale).

### Modelo B - ResNet18
ResNet18 pre-entrenado adaptado para clasificación de audio a partir de espectrogramas.

El notebook define `run_experiment(dataset_name, data_dirs, hparams)` que:

- Carga el dataset desde carpetas por clase.
- Divide los datos (train/val/test, 80/10/10 aproximado).
- Entrena el modelo seleccionado con combinaciones de hiperparámetros.
- Registra métricas (train/val loss, accuracy, F1, precision, recall) y guarda checkpoints.

Para ejecutar los experimentos desde el notebook, abre `proyecto.ipynb` y ejecuta la celda principal (MAIN).

### Opciones de wandb (Weights & Biases)

Si no quieres usar wandb online (por autenticación o por ejecutar sin internet), tienes dos opciones:

- Modo offline (registra localmente):

```zsh
export WANDB_MODE=offline
# o
export WANDB_MODE=disabled   # para desactivar completamente
```

- En Python, desactivar/usar modo offline antes de inicializar wandb:

```python
import os
os.environ['WANDB_MODE'] = 'offline'
import wandb
```

El notebook llama a `wandb.init(...)`. Con `WANDB_MODE=offline` las ejecuciones quedarán registradas en el directorio `wandb/` y se podrán subir más tarde si se desea.

## Resultados y artefactos

- **Resumen de experimentos**: `runs_lenet/lenet_summary.csv` (cada fila: dataset, run, lr, wd, batch, epochs, métricas).
- **Pesos guardados**: 
  - LeNet5 (varios runs): `models/lenet5_best_run*.pth`, `models/lenet5_aug_best_run*.pth`, `models/lenet5_tanh_bn_best_run*.pth`
  - ResNet18: `models/MODEL_B_resnet18_audio_best_run*.pth`, `models/resnet18_audio_AUG_best_run*.pth`
- **Panel interactivo**: si usas wandb online, revisa el proyecto `Proyecto1_IA` en wandb.ai.

## Buenas prácticas y consejos

- Si observas overfitting al usar datos aumentados, prueba:
  - Cambiar optimizador a `AdamW` o `Adam` y ajustar lr/weight decay.
  - Aumentar dropout o añadir regularización adicional.
  - Revisar que la augmentación no esté cambiando etiquetas accidentalmente.
- Reproduce experimentos fijando la semilla (`SEED = 42` está en el notebook).

## Estructura de código (breve)

- `SpectrogramDataset`: carga imágenes organizadas por carpeta de clase y devuelve (tensor, label).
- **Modelo A - LeNet5**: implementación adaptada para entradas 1 canal (grayscale) con capas tanh y dropout. Variantes: LeNet5 básico, LeNet5 con Batch Normalization, LeNet5 con tanh+BN.
- **Modelo B - ResNet18**: arquitectura ResNet18 adaptada para clasificación de audio, entrenada tanto en datos raw como aumentados (AUG).
- `run_experiment`: bucle principal que entrena, valida y evalúa; el código de entrenamiento/validación/test está inline para mayor claridad (sin funciones auxiliares).

## Cómo contribuir / Contacto

Si quieres contribuir, abre un issue o PR con cambios concretos. Para preguntas rápidas, contacta al autor del repo.

---

**Autores**: Javier Alonso Rojas Rojas, Brandon Emmanuel Sanchez Araya, Julio Josue Varela Venegas
**Curso**: Inteligencia Artificial - TEC 2025 II
