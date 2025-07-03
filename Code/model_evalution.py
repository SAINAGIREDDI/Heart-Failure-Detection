import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc, accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVCss
import matplotlib.pyplot as plt
import numpy as np
import os

# Load the dataset
data = pd.read_csv('heart.csv')

# Preprocess the Data
X = data.drop('target', axis=1)
y = data['target']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize classifiers
classifiers = {
    "SVM": SVC(probability=True),
    "KNN": KNeighborsClassifier(),
    "Random Forest": RandomForestClassifier()
}

# Function to plot ROC curve
def plot_roc_curve(fpr, tpr, roc_auc, model_name):
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic - {model_name}')
    plt.legend(loc="lower right")
    plt.savefig(f'ev/roc_curve_{model_name}.png')
    plt.close()

# Dictionary to store performance metrics
performance_metrics = {
    "Algorithm": [],
    "Accuracy": [],
    "F1 Score": [],
    "Precision": [],
    "Recall": []
}

# Training and evaluating models
for name, classifier in classifiers.items():
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    y_pred_proba = classifier.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    plot_roc_curve(fpr, tpr, roc_auc, name)

    cm = confusion_matrix(y_test, y_pred)
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix - {name}')
    plt.colorbar()
    classes = ['No Heart Disease', 'Heart Disease']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(f'ev/confusion_matrix_{name}.png')
    plt.close()

    report = classification_report(y_test, y_pred, output_dict=True)
    performance_metrics["Algorithm"].append(name)
    performance_metrics["Accuracy"].append(report['accuracy'])
    performance_metrics["F1 Score"].append(report['weighted avg']['f1-score'])
    performance_metrics["Precision"].append(report['weighted avg']['precision'])
    performance_metrics["Recall"].append(report['weighted avg']['recall'])

# Saving performance metrics to CSV
performance_df = pd.DataFrame(performance_metrics)
performance_df.to_csv('res/performance_metrics.csv', index=False)

print("Model evaluation completed. Check the 'ev' and 'res' folders for results.")
