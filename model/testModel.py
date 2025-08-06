import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image as PILImage
from transformers import ViTImageProcessor, ViTForImageClassification
from datasets import load_dataset
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score
)
import seaborn as sns
from collections import Counter
import os
import warnings

warnings.filterwarnings('ignore')

# Configuración global
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# Definir las etiquetas unificadas (debe coincidir con el entrenamiento)
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


# Clase del modelo mejorado (debe coincidir con el entrenamiento)
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


# Función para cargar el modelo entrenado
def load_trained_model(model_path="./mejor_modelo_advanced"):
    """Cargar el modelo entrenado y el procesador"""
    try:
        # Cargar procesador
        processor = ViTImageProcessor.from_pretrained(model_path)

        # Cargar modelo personalizado
        model = ImprovedViTForImageClassification(
            num_labels=len(unified_labels),
            dropout_rate=0.3
        )

        # Cargar pesos del modelo
        model_state = torch.load(
            os.path.join(model_path, "pytorch_model.bin"),
            map_location=device
        )
        model.load_state_dict(model_state)
        model.to(device)
        model.eval()

        print(f"✓ Modelo cargado exitosamente desde {model_path}")
        return model, processor

    except Exception as e:
        print(f"Error cargando modelo: {e}")
        print("Intentando cargar modelo estándar...")

        # Fallback: cargar modelo estándar
        try:
            processor = ViTImageProcessor.from_pretrained(model_path)
            model = ViTForImageClassification.from_pretrained(model_path)
            model.to(device)
            model.eval()
            print("✓ Modelo estándar cargado exitosamente")
            return model, processor
        except Exception as e2:
            print(f"Error cargando modelo estándar: {e2}")
            return None, None


# Función de predicción mejorada
def predict_image(image, model, processor, return_probs=False):
    """Predecir clase de una imagen individual"""
    try:
        # Asegurar que la imagen esté en RGB
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Procesar imagen
        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Predicción
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
            predicted_class_id = logits.argmax(-1).item()

        predicted_label = unified_labels[predicted_class_id]

        if return_probs:
            probs_dict = {
                unified_labels[i]: prob.item()
                for i, prob in enumerate(probabilities.squeeze())
            }
            return predicted_label, predicted_class_id, probs_dict
        else:
            return predicted_label, predicted_class_id

    except Exception as e:
        print(f"Error en predicción: {e}")
        return "Error", -1


# Función para procesar datasets de prueba
def load_test_datasets():
    """Cargar y procesar datasets de prueba"""
    print("Cargando datasets de prueba...")

    datasets_info = []

    try:
        # Dataset original
        orig_test = load_dataset("marmal88/skin_cancer", split='test')

        # Mapeo de etiquetas para dataset original
        dx_to_unified = {
            "actinic_keratoses": "Actinic keratoses",
            "basal_cell_carcinoma": "Basal cell carcinoma",
            "benign_keratosis-like_lesions": "Benign keratosis-like-lesions",
            "dermatofibroma": "Dermatofibroma",
            "melanocytic_Nevi": "Melanocytic nevi",
            "melanoma": "Melanoma",
            "vascular_lesions": "Vascular lesions"
        }

        datasets_info.append({
            'dataset': orig_test,
            'name': 'skin_cancer_original',
            'label_key': 'dx',
            'label_mapping': dx_to_unified,
            'is_string_label': True
        })

        print(f"✓ Dataset original cargado: {len(orig_test)} muestras")

    except Exception as e:
        print(f"⚠ Error cargando dataset original: {e}")

    try:
        # Dataset nuevo
        new_test = load_dataset("ahmed-ai/skin-lesions-classification-dataset", split='test')

        datasets_info.append({
            'dataset': new_test,
            'name': 'skin_lesions_new',
            'label_key': 'label',
            'label_mapping': None,
            'is_string_label': False
        })

        print(f"✓ Dataset nuevo cargado: {len(new_test)} muestras")

    except Exception as e:
        print(f"⚠ Error cargando dataset nuevo: {e}")

    return datasets_info


# Función para evaluar un dataset
def evaluate_dataset(dataset_info, model, processor, max_samples=None):
    """Evaluar modelo en un dataset específico"""
    dataset = dataset_info['dataset']
    name = dataset_info['name']

    print(f"\nEvaluando {name}...")

    true_labels = []
    predicted_labels = []
    predicted_ids = []
    true_ids = []
    all_probabilities = []

    # Limitar número de muestras si se especifica
    samples_to_process = min(len(dataset), max_samples) if max_samples else len(dataset)

    for i, example in enumerate(dataset):
        if i >= samples_to_process:
            break

        try:
            # Obtener etiqueta verdadera
            if dataset_info['is_string_label']:
                true_label_str = example[dataset_info['label_key']]
                if dataset_info['label_mapping'] and true_label_str in dataset_info['label_mapping']:
                    true_label_unified = dataset_info['label_mapping'][true_label_str]
                    true_id = label2id[true_label_unified]
                else:
                    continue  # Saltar si no hay mapeo
            else:
                true_id = example[dataset_info['label_key']]
                if 0 <= true_id < len(unified_labels):
                    true_label_unified = unified_labels[true_id]
                else:
                    continue  # Saltar si está fuera de rango

            # Realizar predicción
            pred_label, pred_id, probs = predict_image(
                example['image'], model, processor, return_probs=True
            )

            # Almacenar resultados
            true_labels.append(true_label_unified)
            predicted_labels.append(pred_label)
            true_ids.append(true_id)
            predicted_ids.append(pred_id)
            all_probabilities.append(list(probs.values()))

            # Progreso
            if (i + 1) % 50 == 0:
                print(f"Procesadas {i + 1}/{samples_to_process} muestras...")

        except Exception as e:
            print(f"Error procesando muestra {i}: {e}")
            continue

    print(f"✓ Evaluación completada: {len(true_labels)} muestras procesadas")

    return {
        'true_labels': true_labels,
        'predicted_labels': predicted_labels,
        'true_ids': true_ids,
        'predicted_ids': predicted_ids,
        'probabilities': np.array(all_probabilities),
        'dataset_name': name
    }


# Función para generar reporte detallado - FIXED VERSION
def generate_detailed_report(results):
    """Generar reporte completo de evaluación"""
    true_ids = results['true_ids']
    pred_ids = results['predicted_ids']
    true_labels = results['true_labels']
    pred_labels = results['predicted_labels']
    dataset_name = results['dataset_name']
    probabilities = results['probabilities']

    print(f"\n{'=' * 60}")
    print(f"REPORTE DETALLADO - {dataset_name.upper()}")
    print(f"{'=' * 60}")

    # Métricas generales
    accuracy = accuracy_score(true_ids, pred_ids)
    f1_weighted = f1_score(true_ids, pred_ids, average='weighted', zero_division=0)
    f1_macro = f1_score(true_ids, pred_ids, average='macro', zero_division=0)
    f1_micro = f1_score(true_ids, pred_ids, average='micro', zero_division=0)

    print(f"\nMÉTRICAS GENERALES:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1-Score (Weighted): {f1_weighted:.4f}")
    print(f"F1-Score (Macro): {f1_macro:.4f}")
    print(f"F1-Score (Micro): {f1_micro:.4f}")

    # Distribución de clases
    print(f"\nDISTRIBUCIÓN DE CLASES:")
    true_counter = Counter(true_labels)
    pred_counter = Counter(pred_labels)

    for label in unified_labels:
        true_count = true_counter.get(label, 0)
        pred_count = pred_counter.get(label, 0)
        print(f"{label:30} | Real: {true_count:3d} | Pred: {pred_count:3d}")

    # FIX: Obtener solo las clases que realmente aparecen en los datos
    unique_true_ids = sorted(set(true_ids))
    unique_pred_ids = sorted(set(pred_ids))
    all_unique_ids = sorted(set(unique_true_ids + unique_pred_ids))

    # Crear labels solo para las clases que aparecen
    present_labels = [unified_labels[i] for i in all_unique_ids]

    # Reporte de clasificación detallado - FIXED
    print(f"\nREPORTE DE CLASIFICACIÓN:")
    try:
        class_report = classification_report(
            true_ids, pred_ids,
            labels=all_unique_ids,  # Especificar solo las clases presentes
            target_names=present_labels,  # Usar solo los nombres de clases presentes
            zero_division=0,
            digits=4
        )
        print(class_report)
    except Exception as e:
        print(f"Error generando reporte de clasificación: {e}")
        class_report = "No se pudo generar el reporte de clasificación"

    # Matriz de confusión - FIXED
    try:
        cm = confusion_matrix(true_ids, pred_ids, labels=all_unique_ids)
    except Exception as e:
        print(f"Error generando matriz de confusión: {e}")
        cm = np.array([[0]])

    # Visualización
    plt.figure(figsize=(16, 14))

    # Matriz de confusión
    plt.subplot(2, 2, 1)
    if cm.size > 1:
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=[label[:15] for label in present_labels],
            yticklabels=[label[:15] for label in present_labels],
            cbar_kws={'shrink': 0.8}
        )
        plt.title(f'Matriz de Confusión - {dataset_name}')
        plt.xlabel('Predicho')
        plt.ylabel('Real')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
    else:
        plt.text(0.5, 0.5, 'No hay suficientes datos\npara la matriz de confusión',
                 ha='center', va='center', transform=plt.gca().transAxes)
        plt.title(f'Matriz de Confusión - {dataset_name}')

    # Distribución de confianza
    plt.subplot(2, 2, 2)
    if len(probabilities) > 0:
        max_probs = np.max(probabilities, axis=1)
        plt.hist(max_probs, bins=30, alpha=0.7, edgecolor='black')
        plt.title('Distribución de Confianza de Predicciones')
        plt.xlabel('Probabilidad Máxima')
        plt.ylabel('Frecuencia')
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'No hay datos de probabilidad',
                 ha='center', va='center', transform=plt.gca().transAxes)

    # F1-Score por clase - FIXED
    plt.subplot(2, 2, 3)
    try:
        f1_per_class = f1_score(true_ids, pred_ids, labels=all_unique_ids, average=None, zero_division=0)
        present_labels_short = [label[:15] for label in present_labels]

        plt.barh(range(len(f1_per_class)), f1_per_class)
        plt.yticks(range(len(f1_per_class)), present_labels_short)
        plt.xlabel('F1-Score')
        plt.title('F1-Score por Clase')
        plt.grid(True, alpha=0.3)
    except Exception as e:
        plt.text(0.5, 0.5, f'Error en F1-Score:\n{str(e)}',
                 ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('F1-Score por Clase')

    # Precisión y Recall por clase - FIXED
    plt.subplot(2, 2, 4)
    try:
        precision, recall, _, _ = precision_recall_fscore_support(
            true_ids, pred_ids, labels=all_unique_ids, average=None, zero_division=0
        )

        x = np.arange(len(present_labels))
        width = 0.35

        plt.bar(x - width / 2, precision, width, label='Precisión', alpha=0.8)
        plt.bar(x + width / 2, recall, width, label='Recall', alpha=0.8)

        plt.xlabel('Clases')
        plt.ylabel('Score')
        plt.title('Precisión y Recall por Clase')
        present_labels_short = [label[:15] for label in present_labels]
        plt.xticks(x, present_labels_short, rotation=45, ha='right')
        plt.legend()
        plt.grid(True, alpha=0.3)
    except Exception as e:
        plt.text(0.5, 0.5, f'Error en Precisión/Recall:\n{str(e)}',
                 ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Precisión y Recall por Clase')

    plt.tight_layout()
    plt.savefig(f'evaluation_report_{dataset_name}.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Guardar reporte en archivo
    report_filename = f'detailed_report_{dataset_name}.txt'
    with open(report_filename, 'w', encoding='utf-8') as f:
        f.write(f"REPORTE DETALLADO - {dataset_name.upper()}\n")
        f.write("=" * 60 + "\n\n")

        f.write("MÉTRICAS GENERALES:\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"F1-Score (Weighted): {f1_weighted:.4f}\n")
        f.write(f"F1-Score (Macro): {f1_macro:.4f}\n")
        f.write(f"F1-Score (Micro): {f1_micro:.4f}\n\n")

        f.write("DISTRIBUCIÓN DE CLASES:\n")
        for label in unified_labels:
            true_count = true_counter.get(label, 0)
            pred_count = pred_counter.get(label, 0)
            f.write(f"{label:30} | Real: {true_count:3d} | Pred: {pred_count:3d}\n")

        f.write("\nCLASES PRESENTES EN LA EVALUACIÓN:\n")
        f.write(f"Clases reales únicas: {len(unique_true_ids)}\n")
        f.write(f"Clases predichas únicas: {len(unique_pred_ids)}\n")
        f.write(f"Clases totales presentes: {len(all_unique_ids)}\n")
        for i, class_id in enumerate(all_unique_ids):
            f.write(f"  {class_id}: {unified_labels[class_id]}\n")

        f.write("\nREPORTE DE CLASIFICACIÓN:\n")
        f.write(str(class_report))

        f.write("\nMATRIZ DE CONFUSIÓN:\n")
        f.write(str(cm))

    print(f"\n✓ Reporte guardado en: {report_filename}")
    print(f"✓ Gráficos guardados en: evaluation_report_{dataset_name}.png")

    return {
        'accuracy': accuracy,
        'f1_weighted': f1_weighted,
        'f1_macro': f1_macro,
        'confusion_matrix': cm,
        'classification_report': class_report,
        'present_classes': present_labels,
        'unique_class_ids': all_unique_ids
    }


# Función para probar una imagen individual
def test_single_image(image_path, model, processor):
    """Probar el modelo con una imagen individual"""
    try:
        # Cargar imagen
        image = PILImage.open(image_path)

        # Realizar predicción con probabilidades
        pred_label, pred_id, probs = predict_image(
            image, model, processor, return_probs=True
        )

        print(f"\nPredicción para {image_path}:")
        print(f"Clase predicha: {pred_label}")
        print(f"Confianza: {max(probs.values()):.4f}")

        print(f"\nTop 5 predicciones:")
        sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        for i, (label, prob) in enumerate(sorted_probs[:5]):
            print(f"{i + 1}. {label}: {prob:.4f}")

        # Mostrar imagen
        plt.figure(figsize=(8, 6))
        plt.imshow(image)
        plt.title(f'Predicción: {pred_label}\nConfianza: {max(probs.values()):.4f}')
        plt.axis('off')
        plt.show()

        return pred_label, probs

    except Exception as e:
        print(f"Error procesando imagen {image_path}: {e}")
        return None, None


# Función principal
def main():
    print("SISTEMA DE EVALUACIÓN AVANZADO - MODELO ViT")
    print("=" * 60)

    # Cargar modelo
    model, processor = load_trained_model()
    if model is None or processor is None:
        print("Error: No se pudo cargar el modelo. Abortando...")
        return

    # Cargar datasets de prueba
    datasets_info = load_test_datasets()
    if not datasets_info:
        print("Error: No se pudieron cargar datasets de prueba.")
        return

    # Evaluar cada dataset
    all_results = []
    for dataset_info in datasets_info:
        # Evaluar con límite de muestras para pruebas rápidas (remover max_samples para evaluación completa)
        results = evaluate_dataset(dataset_info, model, processor, max_samples=200)
        all_results.append(results)

        # Generar reporte detallado
        metrics = generate_detailed_report(results)

    print(f"\n{'=' * 60}")
    print("EVALUACIÓN COMPLETADA")
    print(f"{'=' * 60}")
    print("Archivos generados:")
    for results in all_results:
        dataset_name = results['dataset_name']
        print(f"- detailed_report_{dataset_name}.txt")
        print(f"- evaluation_report_{dataset_name}.png")

    # Ejemplo de uso para imagen individual (descomenta para usar)
    # print("\nEjemplo de predicción individual:")
    # test_single_image("path/to/your/image.jpg", model, processor)


if __name__ == "__main__":
    main()