# Instalación de dependencias (ejecutar en terminal antes)
# pip install transformers datasets torch torchvision torchaudio pillow scikit-learn pandas numpy matplotlib
# pip install albumentations seaborn timm  # Dependencias adicionales
# pip install wandb  # Opcional para logging avanzado

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset, concatenate_datasets, Features, Value, Image
from transformers import (
    ViTImageProcessor,
    ViTForImageClassification,
    TrainingArguments,
    Trainer,
    DefaultDataCollator,
    EarlyStoppingCallback
)
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from collections import Counter

from PIL import Image as PILImage
import warnings

warnings.filterwarnings('ignore')

# Importaciones opcionales
try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False

try:
    import seaborn as sns

    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Configuración global
USE_WANDB = False
device = None


def setup_global_config():
    global USE_WANDB, device
    USE_WANDB = WANDB_AVAILABLE and False  # Cambiar a True si quieres usar wandb
    if USE_WANDB:
        try:
            wandb.init(project="skin-cancer-vit-advanced", name="vit-advanced-v1")
            print("✓ Wandb inicializado correctamente")
        except Exception as e:
            print(f"⚠ Error inicializando wandb: {e}")
            USE_WANDB = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memoria total: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB")

    # Configurar torch para optimización
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    print("✓ Albumentations disponible" if ALBUMENTATIONS_AVAILABLE else "⚠ Albumentations no disponible")
    print("✓ Weights & Biases disponible" if WANDB_AVAILABLE else "⚠ Weights & Biases no disponible")
    print("✓ Seaborn disponible" if SEABORN_AVAILABLE else "⚠ Seaborn no disponible")


# Definir mapeo unificado de etiquetas (14 clases)
unified_labels = [
    "Actinic keratoses",
    "Basal cell carcinoma",
    "Benign keratosis-like-lesions",
    "Chickenpox",
    "Cowpox",
    "Dermatofibroma",
    "Healthy",
    "HFMD",
    "Measles",
    "Melanocytic nevi",
    "Melanoma",
    "Monkeypox",
    "Squamous cell carcinoma",
    "Vascular lesions"
]

label2id = {label: idx for idx, label in enumerate(unified_labels)}
id2label = {idx: label for idx, label in enumerate(unified_labels)}


# Función para cargar y procesar los datasets
def load_and_process_datasets():
    print("Cargando datasets...")

    try:
        # Cargar dataset original
        orig_train = load_dataset("marmal88/skin_cancer", split='train')
        orig_test = load_dataset("marmal88/skin_cancer", split='test')
        print(f"Dataset original cargado: {len(orig_train)} train, {len(orig_test)} test")
    except Exception as e:
        print(f"Error cargando dataset original: {e}")
        return None, None

    try:
        # Cargar dataset nuevo
        new_train = load_dataset("ahmed-ai/skin-lesions-classification-dataset", split='train')
        new_test = load_dataset("ahmed-ai/skin-lesions-classification-dataset", split='test')
        print(f"Dataset nuevo cargado: {len(new_train)} train, {len(new_test)} test")
    except Exception as e:
        print(f"Error cargando dataset nuevo: {e}")
        return None, None

    # Mapeo de etiquetas para dataset original
    dx_to_unified = {
        "actinic_keratoses": "Actinic keratoses",
        "basal_cell_carcinoma": "Basal cell carcinoma",
        "benign_keratosis-like_lesions": "Benign keratosis-like-lesions",
        "dermatofibroma": "Dermatofibroma",
        "melanocytic_Nevi": "Melanocytic nevi",  # Corregido: mayúscula en Nevi
        "melanoma": "Melanoma",
        "vascular_lesions": "Vascular lesions"
    }

    # Función para mapear etiquetas del dataset original
    def map_orig_labels(example):
        if example["dx"] in dx_to_unified:
            example["labels"] = label2id[dx_to_unified[example["dx"]]]
        else:
            # Si no encuentra la etiqueta, asignar una por defecto o saltar
            print(f"Etiqueta no encontrada: {example['dx']}")
            example["labels"] = 0  # o podrías filtrar estos ejemplos
        return example

    # Función para mapear etiquetas del dataset nuevo
    def map_new_labels(example):
        # Asegurar que el label esté en el rango correcto
        label = int(example["label"])
        if 0 <= label < len(unified_labels):
            example["labels"] = label
        else:
            print(f"Label fuera de rango: {label}")
            example["labels"] = 0
        return example

    # Aplicar mapeo de etiquetas
    orig_train = orig_train.map(map_orig_labels)
    orig_test = orig_test.map(map_orig_labels)
    new_train = new_train.map(map_new_labels)
    new_test = new_test.map(map_new_labels)

    # Limpiar columnas innecesarias
    columns_to_remove_orig = ["dx", "lesion_id", "image_id", "dx_type", "age", "sex", "localization"]
    columns_to_remove_new = ["label"]

    # Verificar qué columnas existen antes de eliminarlas
    for col in columns_to_remove_orig:
        if col in orig_train.column_names:
            orig_train = orig_train.remove_columns([col])
            orig_test = orig_test.remove_columns([col])

    for col in columns_to_remove_new:
        if col in new_train.column_names:
            new_train = new_train.remove_columns([col])
            new_test = new_test.remove_columns([col])

    # Definir características comunes
    common_features = Features({
        "image": Image(),
        "labels": Value("int64")
    })

    # Aplicar características comunes con manejo de errores
    try:
        orig_train = orig_train.cast(common_features)
        orig_test = orig_test.cast(common_features)
        new_train = new_train.cast(common_features)
        new_test = new_test.cast(common_features)
    except Exception as e:
        print(f"Error aplicando características comunes: {e}")
        return None, None

    # Combinar datasets
    combined_train = concatenate_datasets([orig_train, new_train])
    combined_test = concatenate_datasets([orig_test, new_test])

    print(f"Datasets combinados: {len(combined_train)} train, {len(combined_test)} test")
    return combined_train, combined_test


# Modelo ViT mejorado - SIMPLIFICADO y CORREGIDO
class ImprovedViTForImageClassification(nn.Module):
    def __init__(self, model_name="google/vit-base-patch16-224-in21k", num_labels=14, dropout_rate=0.3):
        super().__init__()
        # Usar el modelo base de ViT con clasificación
        self.vit = ViTForImageClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            ignore_mismatched_sizes=True
        )

        # Reemplazar el clasificador por uno mejorado
        hidden_size = self.vit.config.hidden_size
        self.vit.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(256, num_labels)
        )

    def forward(self, pixel_values, labels=None):
        return self.vit(pixel_values=pixel_values, labels=labels)


# Clase de entrenador avanzado - CORREGIDA
class AdvancedTrainer(Trainer):
    def __init__(self, class_weights=None, focal_loss_alpha=1.0, focal_loss_gamma=2.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.focal_loss_alpha = focal_loss_alpha
        self.focal_loss_gamma = focal_loss_gamma

    def focal_loss(self, inputs, targets, alpha=1.0, gamma=2.0):
        """Implementación de Focal Loss para manejar desbalance de clases"""
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.class_weights)
        pt = torch.exp(-ce_loss)
        focal_loss = alpha * (1 - pt) ** gamma * ce_loss
        return focal_loss.mean()

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Función de pérdida personalizada"""
        labels = inputs.get("labels")
        outputs = model(**inputs)

        if labels is not None:
            logits = outputs.get('logits')
            # Usar Focal Loss
            loss = self.focal_loss(
                logits,
                labels,
                alpha=self.focal_loss_alpha,
                gamma=self.focal_loss_gamma
            )
            outputs['loss'] = loss

        return (outputs['loss'], outputs) if return_outputs else outputs['loss']


# Función de métricas mejorada
def compute_metrics_advanced(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    accuracy = accuracy_score(labels, predictions)
    f1_weighted = f1_score(labels, predictions, average='weighted', zero_division=0)
    f1_macro = f1_score(labels, predictions, average='macro', zero_division=0)
    f1_micro = f1_score(labels, predictions, average='micro', zero_division=0)

    # Métricas por clase
    f1_per_class = f1_score(labels, predictions, average=None, zero_division=0)

    metrics = {
        'accuracy': accuracy,
        'f1_weighted': f1_weighted,
        'f1_macro': f1_macro,
        'f1_micro': f1_micro,
    }

    # Agregar F1 por clase de forma segura
    for i, class_name in enumerate(unified_labels):
        if i < len(f1_per_class):
            metrics[f'f1_{class_name.replace(" ", "_").replace("-", "_").lower()}'] = f1_per_class[i]

    return metrics


# Función de preprocessing simplificada
def preprocess_function(examples, processor):
    """Funcíon de preprocesamiento simplificada y robusta"""
    images = []
    for img in examples["image"]:
        try:
            # Asegurar que la imagen está en RGB
            if img.mode != "RGB":
                img = img.convert("RGB")
            images.append(img)
        except Exception as e:
            print(f"Error procesando imagen: {e}")
            # Crear una imagen en blanco como fallback
            images.append(PILImage.new('RGB', (224, 224), color='white'))

    # Procesar todas las imágenes de una vez
    try:
        processed = processor(images=images, return_tensors="pt")
        return processed
    except Exception as e:
        print(f"Error en processor: {e}")
        # Fallback: procesar una por una
        pixel_values = []
        for img in images:
            try:
                proc_img = processor(images=img, return_tensors="pt")
                pixel_values.append(proc_img["pixel_values"].squeeze(0))
            except:
                # Imagen en blanco como último recurso
                pixel_values.append(torch.zeros(3, 224, 224))
        return {"pixel_values": torch.stack(pixel_values)}


# Función principal de entrenamiento
def main():
    # Configurar entorno
    setup_global_config()

    # Cargar datasets
    train_ds, test_ds = load_and_process_datasets()
    if train_ds is None or test_ds is None:
        print("Error cargando datasets. Abortando...")
        return

    # Análisis de distribución de clases
    class_counts = Counter(train_ds["labels"])
    print("\nDistribución de clases en entrenamiento:")
    for i, count in class_counts.items():
        if i < len(unified_labels):
            print(f"{unified_labels[i]}: {count}")

    # Calcular pesos de clase para manejar desbalance
    total_samples = sum(class_counts.values())
    class_weights = torch.tensor([
        total_samples / (len(class_counts) * class_counts.get(i, 1))
        for i in range(len(unified_labels))
    ], dtype=torch.float).to(device)

    print(f"Pesos de clase calculados: {class_weights[:5]}...")  # Mostrar solo los primeros 5

    # Inicializar procesador
    processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

    # Preprocesar datasets
    print("Preprocesando datasets...")

    def preprocess_train(examples):
        return preprocess_function(examples, processor)

    def preprocess_val(examples):
        return preprocess_function(examples, processor)

    # Aplicar preprocesamiento con manejo de errores
    try:
        train_ds = train_ds.map(
            preprocess_train,
            batched=True,
            batch_size=16,
            remove_columns=["image"],
            desc="Preprocessing train"
        )
        test_ds = test_ds.map(
            preprocess_val,
            batched=True,
            batch_size=16,
            remove_columns=["image"],
            desc="Preprocessing test"
        )
        print("✓ Preprocesamiento completado")
    except Exception as e:
        print(f"Error en preprocesamiento: {e}")
        return

    # Crear modelo mejorado
    print("Inicializando modelo...")
    try:
        model = ImprovedViTForImageClassification(
            model_name="google/vit-base-patch16-224-in21k",
            num_labels=len(unified_labels),
            dropout_rate=0.3
        ).to(device)
        print("✓ Modelo inicializado correctamente")
    except Exception as e:
        print(f"Error inicializando modelo: {e}")
        return

    # Configuración de entrenamiento
    training_args = TrainingArguments(
        output_dir="./skin_cancer_vit_advanced",
        per_device_train_batch_size=8,  # Reducido para evitar OOM
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=4,
        num_train_epochs=10,  # Reducido para pruebas
        learning_rate=2e-5,  # Learning rate más conservador
        weight_decay=0.01,
        warmup_steps=500,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="f1_weighted",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),  # Solo si CUDA está disponible
        dataloader_pin_memory=False,  # Desactivado para evitar problemas
        dataloader_num_workers=0,  # Reducido para evitar conflictos
        remove_unused_columns=False,
        push_to_hub=False,
        report_to="wandb" if USE_WANDB else None,
        run_name="vit-advanced-skin-cancer",
        save_total_limit=2,
        seed=42,
    )

    # Crear entrenador
    trainer = AdvancedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        compute_metrics=compute_metrics_advanced,
        data_collator=DefaultDataCollator(return_tensors="pt"),
        class_weights=class_weights,
        focal_loss_alpha=1.0,
        focal_loss_gamma=2.0,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    # Entrenamiento
    try:
        print("Iniciando entrenamiento...")
        train_result = trainer.train()

        # Guardar modelo
        print("Guardando modelo...")
        trainer.save_model("./mejor_modelo_advanced")
        processor.save_pretrained("./mejor_modelo_advanced")

        # Evaluación final
        print("Realizando evaluación final...")
        eval_results = trainer.evaluate()

        # Predicciones para análisis
        predictions = trainer.predict(test_ds)
        y_pred = np.argmax(predictions.predictions, axis=1)
        y_true = predictions.label_ids

        # Reporte de clasificación
        print("\nReporte de clasificación:")
        print(classification_report(y_true, y_pred, target_names=unified_labels, zero_division=0))

        # Crear matriz de confusión
        plt.figure(figsize=(15, 12))
        cm = confusion_matrix(y_true, y_pred)

        if SEABORN_AVAILABLE:
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=[label[:15] for label in unified_labels],
                        yticklabels=[label[:15] for label in unified_labels])
        else:
            plt.imshow(cm, interpolation='nearest', cmap='Blues')
            plt.colorbar()
            tick_marks = np.arange(len(unified_labels))
            plt.xticks(tick_marks, [label[:15] for label in unified_labels], rotation=45)
            plt.yticks(tick_marks, [label[:15] for label in unified_labels])

        plt.title('Matriz de Confusión - Clasificación de Cáncer de Piel')
        plt.ylabel('Etiqueta Real')
        plt.xlabel('Etiqueta Predicha')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()

        # Guardar métricas
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.log_metrics("eval", eval_results)
        trainer.save_metrics("eval", eval_results)

        print(f"\n¡Entrenamiento completado exitosamente!")
        print(f"F1 Score Weighted: {eval_results['eval_f1_weighted']:.4f}")
        print(f"Accuracy: {eval_results['eval_accuracy']:.4f}")

        if USE_WANDB:
            wandb.log({
                "final_f1_weighted": eval_results['eval_f1_weighted'],
                "final_accuracy": eval_results['eval_accuracy']
            })

    except Exception as e:
        print(f"Error durante el entrenamiento: {str(e)}")
        import traceback
        traceback.print_exc()

    finally:
        # Limpiar memoria
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        if USE_WANDB:
            wandb.finish()


if __name__ == "__main__":
    main()