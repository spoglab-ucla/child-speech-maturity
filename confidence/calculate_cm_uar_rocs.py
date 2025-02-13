import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc, confusion_matrix, recall_score

# File Paths
model_cleaned_path = "path.csv"
model_uncleaned_path = "path.csv"

# Corrected Label Mapping
label_mapping = {"Canonical": 2, "Non-Canonical": 1, "Cry": 4, "Junk": 0, "Laugh": 3,}
class_labels = ["Junk", "Non-Canonical", "Canonical", "Laugh", "Cry"]
num_classes = len(class_labels)

# Colorblind-Friendly Palette
colors = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#F0E442']

plt.rcParams.update({
    'font.size': 20,
    'axes.titlesize': 30,
    'axes.labelsize': 20,
    'legend.fontsize': 20,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
})

def calculate_uar(true_labels, predicted_labels, dataset_name):
    """Computes and prints Unweighted Average Recall (UAR) for each label."""
    unique_classes = np.unique(true_labels)
    recalls = []

    print(f"{dataset_name} UAR per class:")
    for class_label in unique_classes:
        # Compute recall for each class
        recall = recall_score(
            true_labels == class_label, predicted_labels == class_label, zero_division=0
        )
        recalls.append(recall)
        print(f"  {class_labels[class_label]}: {recall:.4f}")

    # Compute overall UAR
    uar = np.mean(recalls)
    print(f"{dataset_name} Overall UAR: {uar:.4f}")
    return uar

def generate_confusion_matrix(true_labels, predicted_labels, dataset_name, save_path):
    """Generates and saves a confusion matrix with improved figure size and text scaling."""
    # Desired order
    reordered_classes = ["Canonical", "Non-Canonical", "Cry", "Junk", "Laugh"]
    
    # Create a mapping from old class indices to new ones
    reordered_indices = [label_mapping[label] for label in reordered_classes]

    # Compute confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels, labels=range(num_classes))

    # Reorder the confusion matrix
    cm = cm[np.ix_(reordered_indices, reordered_indices)]

    # Normalize the confusion matrix
    cm_sums = cm.sum(axis=1, keepdims=True)
    cm_sums[cm_sums == 0] = 1  # Prevent division by zero
    cm_percentages = cm.astype('float') / cm_sums * 100

    # Plot the confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_percentages, annot=True, cmap='Blues', fmt='.2f', 
                xticklabels=reordered_classes, yticklabels=reordered_classes, annot_kws={"size": 20})
    
    plt.xlabel('Predicted', fontsize=20)
    plt.ylabel('True', fontsize=20)
    plt.xticks(rotation=0, ha='center', fontsize=17)
    plt.yticks(fontsize=17)

    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def plot_and_save_roc_curve(predictions_df, dataset_name, save_path):
    """Generates and saves a multi-class ROC curve using logits."""
    fig, ax = plt.subplots(figsize=(12, 10))  # Increased figure size

    # Extract true labels
    true_labels = predictions_df["true_label"].values

    # Convert stored string list (logits) to NumPy array safely
    try:
        logits = np.array(predictions_df["logits"].astype(str).apply(eval).tolist())  # Converts stored strings to lists
    except Exception as e:
        print(f"Error converting logits in {dataset_name}: {e}")
        return

    # Convert logits to probabilities using softmax
    probabilities = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)

    # Binarize the labels for one-vs-all ROC calculation
    y_test = label_binarize(true_labels, classes=range(num_classes))

    for i, color in zip(range(num_classes), colors):
        fpr, tpr, _ = roc_curve(y_test[:, i], probabilities[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, label=f'{class_labels[i]} (AUC = {roc_auc:.2f})', linewidth=3)

    # Plot diagonal reference line
    ax.plot([0, 1], [0, 1], 'k--', lw=2)

    # Labels with larger font sizes
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=30)
    ax.set_ylabel('True Positive Rate', fontsize=30)

    # Increase tick font size
    ax.tick_params(axis='both', which='major', labelsize=28)  # Bigger xticks and yticks

    # Increase legend font size and remove title
    ax.legend(loc="lower right", fontsize=30, frameon=True)

    ax.grid(True)

    # Save the plot
    plt.savefig(save_path, bbox_inches='tight')  # Ensure the plot is not cropped
    plt.close()


# Main Function
def main():
    """Processes model predictions and generates UAR, Confusion Matrices, and ROC curves."""
    # Load Predictions CSV Files
    df_model_unclean = pd.read_csv(model_uncleaned_path)
    df_model_clean = pd.read_csv(model_cleaned_path)

    # Convert stored string lists into proper lists (logits and probabilities)
    df_model_unclean["logits"] = df_model_unclean["logits"].astype(str).apply(eval)
    df_model_clean["logits"] = df_model_clean["logits"].astype(str).apply(eval)

    # Compute UAR and save confusion matrices
    for df, dataset_name, cm_path, roc_path in [
        (df_model_unclean, "SpeechMaturity-Uncleaned", "w2v2ll4300prosm_on_unclean_cm_new_labels.png", "roc_uncleaned_new_labels.png"),
        (df_model_clean, "SpeechMaturity-Cleaned", "w2v2ll4300prosm_on_clean_cm_new_labels.png", "roc_cleaned_new_labels.png")
    ]:
        # Calculate UAR
        calculate_uar(df["true_label"].values, df["predicted_label"].values, dataset_name)

        # Generate and save confusion matrix
        generate_confusion_matrix(df["true_label"].values, df["predicted_label"].values, dataset_name, cm_path)

        # Generate and save ROC curve
        plot_and_save_roc_curve(df, dataset_name, roc_path)

# Run the script
if __name__ == "__main__":
    main()
