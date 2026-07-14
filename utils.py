import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(y_test, y_pred, model_name):
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')

    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.savefig(f"{model_name}_cm.png")
    plt.close()


def plot_metrics(results):
    models = list(results.keys())

    accuracy = [results[m]['accuracy'] for m in models]
    precision = [results[m]['precision'] for m in models]
    recall = [results[m]['recall'] for m in models]
    f1 = [results[m]['f1'] for m in models]

    x = range(len(models))

    plt.figure(figsize=(8, 5))
    plt.plot(x, accuracy, marker='o', label='Accuracy')
    plt.plot(x, precision, marker='o', label='Precision')
    plt.plot(x, recall, marker='o', label='Recall')
    plt.plot(x, f1, marker='o', label='F1 Score')

    plt.xticks(x, models)
    plt.title("Model Comparison")
    plt.legend()
    plt.show()