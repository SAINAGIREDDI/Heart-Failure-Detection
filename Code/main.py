import os
from tkinter import Tk, filedialog, Button, Label
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns

def upload_csv():
    filepath = filedialog.askopenfilename()
    if filepath:
        process_data(filepath)


def process_data(filepath):
    # Read the CSV
    df = pd.read_csv(filepath)

    # Generating and saving plots
    save_analysis_plots(df)

    # Applying SMOTE and Logistic Regression
    save_logistic_results(df)


def save_analysis_plots(df):
    # Create 'analysis' directory if not exists
    if not os.path.exists('analysis'):
        os.makedirs('analysis')

    # Defining the required variables
    features = df.columns[:-1]  # Exclude the target column
    subset_features = ['age', 'trestbps', 'chol', 'thalach', 'target']
    categorical_features = ['cp', 'fbs', 'restecg', 'exang']

    # Save Density plots
    for feature in features:
        plt.figure()
        sns.kdeplot(data=df, x=feature, hue="target", fill=True, common_norm=False, palette="coolwarm")
        plt.title(f'Density Plot of {feature}')
        plt.savefig(f'analysis/density_{feature}.jpg')
        plt.close()

    # Save Pair-plots
    sns.pairplot(df[subset_features], hue='target', palette='coolwarm', corner=True)
    plt.savefig('analysis/pair_plots.jpg')
    plt.close()

    # Save Count plots
    for feature in categorical_features:
        plt.figure()
        sns.countplot(data=df, x=feature, hue="target", palette="coolwarm")
        plt.title(f'Count Plot of {feature}')
        plt.savefig(f'analysis/count_{feature}.jpg')
        plt.close()


# This is the corrected version of the save_analysis_plots function.


def save_logistic_results(df):
    # Create 'results' directory if not exists
    if not os.path.exists('results'):
        os.makedirs('results')

    # Splitting data
    X = df.drop('target', axis=1)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # SMOTE
    smote = SMOTE()
    X_smote, y_smote = smote.fit_resample(X_train, y_train)

    # Standard Scaling
    scaler = StandardScaler()
    X_smote_scaled = scaler.fit_transform(X_smote)
    X_test_scaled = scaler.transform(X_test)

    # Logistic Regression
    model = LogisticRegression()
    model.fit(X_smote_scaled, y_smote)
    y_pred = model.predict(X_test_scaled)

    # Save performance metrics
    report = classification_report(y_test, y_pred, output_dict=True)
    metrics_df = pd.DataFrame(report).transpose()
    metrics_df.to_csv('results/logistic_performance.csv')

    # Save confusion matrix
    plt.figure()
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap="YlGnBu")
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('results/confusion_matrix.jpg')
    plt.close()

    # Save ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test_scaled)[:, 1])
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Curve')
    plt.legend(loc="lower right")
    plt.savefig('results/roc_curve.jpg')
    plt.close()


# GUI for uploading CSV file
root = Tk()
root.title("CSV Analysis Tool")
label = Label(root, text="Upload CSV file for Analysis", padx=20, pady=10)
label.pack()
button = Button(root, text="Upload CSV", command=upload_csv, padx=20, pady=10)
button.pack()
root.mainloop()
