import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os
import argparse


def load_and_process_data(csv_file):
    """Load the CSV data and convert it to format suitable for confusion matrix."""
    df = pd.read_csv(csv_file)

    # Mapping from English to Spanish labels
    label_mapping = {
        "Angry": "Enojo",
        "Disgust": "Disgusto",
        "Fear": "Miedo",
        "Happy": "Felicidad",
        "Neutral": "Neutral",
        "Sad": "Tristeza",
        "Surprise": "Sorpresa",
    }

    # Create lists for actual and predicted labels
    actual_labels = []
    predicted_labels = []

    # Expand the data based on nPredictions count
    for _, row in df.iterrows():
        actual = label_mapping.get(row["Actual"], row["Actual"])
        predicted = label_mapping.get(row["Predicted"], row["Predicted"])
        count = int(row["nPredictions"])

        # Add the labels count times
        actual_labels.extend([actual] * count)
        predicted_labels.extend([predicted] * count)

    return actual_labels, predicted_labels


def create_confusion_matrix(
    actual_labels,
    predicted_labels,
    output_path="confusion_matrix.png",
    use_percentages=False,
):
    """Create and save a confusion matrix visualization."""

    # Define the emotion labels in a consistent order
    emotion_labels = [
        "Enojo",
        "Disgusto",
        "Miedo",
        "Felicidad",
        "Tristeza",
        "Sorpresa",
        "Neutral",
    ]

    # Create confusion matrix
    cm = confusion_matrix(actual_labels, predicted_labels, labels=emotion_labels)

    # Create the plot
    plt.figure(figsize=(10, 8))

    # Prepare data for visualization
    if use_percentages:
        # Convert to percentages (normalize by row)
        cm_display = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] * 100.0

        fmt = ".2f"
        cbar_label = "Porcentaje (%)"
        title_suffix = "(Porcentajes)"
    else:
        cm_display = cm
        fmt = "d"
        cbar_label = "Número de predicciones"
        title_suffix = "(Cantidad)"

    # Create heatmap with custom colormap
    sns.heatmap(
        cm_display,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=emotion_labels,
        yticklabels=emotion_labels,
        cbar_kws={"label": cbar_label},
    )

    plt.title(
        f"Modelo Propuestos {title_suffix}",
        fontsize=16,
        fontweight="bold",
    )
    plt.xlabel("Emoción Predicha", fontsize=12)
    plt.ylabel("Emoción Real", fontsize=12)
    plt.tight_layout()

    # Save the plot
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()  # Close the figure to free memory

    # Print some statistics
    print(f"Matriz de confusión guardada en: {output_path}")
    print(f"Total de predicciones: {np.sum(cm)}")
    print(f"Precisión general: {np.trace(cm) / np.sum(cm):.3f}")

    # Print per-class accuracy
    print("\nPrecisión por clase:")
    for i, emotion in enumerate(emotion_labels):
        accuracy = cm[i, i] / np.sum(cm[i, :]) if np.sum(cm[i, :]) > 0 else 0
        print(f"{emotion}: {accuracy:.3f}")


def main():
    parser = argparse.ArgumentParser(
        description="Generar matriz de confusión a partir de datos CSV"
    )
    parser.add_argument("--csv", "-c", required=True, help="Ruta al archivo CSV")
    parser.add_argument(
        "--percentages",
        "--porcentajes",
        "-p",
        action="store_true",
        help="Mostrar porcentajes en lugar de cantidades en la matriz de confusión",
    )
    parser.add_argument(
        "--output-dir",
        "--directorio-salida",
        "-o",
        default=None,
        help="Directorio de salida. Si no se proporciona, guarda junto al archivo CSV.",
    )

    args = parser.parse_args()

    csv_file = args.csv

    # Determine output directory
    if args.output_dir:
        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = os.path.dirname(os.path.abspath(csv_file))

    # Generate output filename based on input CSV name
    csv_base_name = os.path.splitext(os.path.basename(csv_file))[0]
    suffix = "_porcentajes" if args.percentages else "_cantidades"
    output_name = f"{csv_base_name}{suffix}.png"
    output_path = os.path.join(output_dir, output_name)

    # Check if CSV file exists
    if not os.path.exists(csv_file):
        print(f"Error: Archivo CSV '{csv_file}' no encontrado.")
        return

    # Load and process the data
    print(f"Cargando datos desde CSV: {csv_file}")
    actual_labels, predicted_labels = load_and_process_data(csv_file)

    # Create and save confusion matrix
    print("Generando matriz de confusión...")
    create_confusion_matrix(
        actual_labels, predicted_labels, output_path, args.percentages
    )


if __name__ == "__main__":
    main()
