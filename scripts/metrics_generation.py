import os
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

os.makedirs(os.path.join(os.path.dirname(__file__), "metrics"), exist_ok=True)


predictions = os.listdir(os.path.join(os.path.dirname(__file__), "..", "metrics"))
predictions = [f for f in predictions if "ckplus" in f and f.endswith(".csv")]

for pred_file in predictions:
    output_path = os.path.join(os.path.dirname(__file__), "metrics", pred_file.replace(".csv", ""))
    os.makedirs(output_path, exist_ok=True)

    tags = pred_file.split(".")[0].split('_')
    model = tags[0]
    dataset = tags[1]

    emotions = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
    spanish_emotions = ["Enojo", "Disgusto", "Miedo", "Felicidad", "Neutral", "Tristeza", "Sorpresa"]
    if dataset == "gimefive":
        emotions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise"]
        spanish_emotions = ["Enojo", "Disgusto", "Miedo", "Felicidad", "Tristeza", "Sorpresa"]

    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "..", "metrics", pred_file))
    actual_emotions = df["actual"].values
    predicted_emotions = df["predicted"].values

    cr = classification_report(actual_emotions, predicted_emotions, target_names=emotions)
    with open(os.path.join(output_path, "report.txt"), "w") as f:
        f.write(f"{model} - {dataset} \n")
        f.write(cr)

    cm = confusion_matrix(actual_emotions, predicted_emotions, labels=emotions, normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=spanish_emotions)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = disp.plot(cmap=plt.cm.Blues, values_format='.2f', ax=ax, colorbar=False)

    if "cbam" in pred_file.lower():
        model_name = "propuesto"
    else:
        model_name = "base"
    plt.title(f"Matriz de confusión del modelo {model_name}")
    plt.xlabel("Predicción")
    plt.ylabel("Realidad")

    image_name = f"{dataset}_{model}_cm.png"
    plt.savefig(os.path.join(output_path, image_name))
