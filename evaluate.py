import os
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, ConfusionMatrixDisplay,
                             classification_report)


def evaluate_model(model, test_loader, device, output_dir):
    model.eval()
    actual, predicted = [], []
    with torch.no_grad():
        for batch in test_loader:
            images = batch['xray'].to(device)
            labels = batch['target'].to(device)
            outputs = model(images)
            actual.extend(labels.cpu().tolist())
            predicted.extend(outputs.argmax(1).cpu().tolist())

    results = {
        'Accuracy': accuracy_score(actual, predicted),
        'Precision': precision_score(actual, predicted, zero_division=0),
        'Recall': recall_score(actual, predicted, zero_division=0),
        'F1-score': f1_score(actual, predicted, zero_division=0)
    }
    report = classification_report(actual, predicted, labels=[0, 1],
                                   target_names=['Not Fractured', 'Fractured'], zero_division=0)
    with open(os.path.join(output_dir, 'test_results.txt'), 'w', encoding='utf-8') as file:
        for name, value in results.items():
            file.write(f'{name}: {value:.4f}\n')
        file.write('\n' + report)
    for name, value in results.items():
        print(f'{name}: {value:.4f}')
    print(report)

    matrix = confusion_matrix(actual, predicted, labels=[0, 1])
    display = ConfusionMatrixDisplay(matrix, display_labels=['Not Fractured', 'Fractured'])
    display.plot(cmap='Blues', values_format='d')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()
