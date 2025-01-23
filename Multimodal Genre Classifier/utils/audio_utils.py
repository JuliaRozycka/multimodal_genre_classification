import torch
from torch.utils.data import Dataset
import os
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, roc_curve


class MelSpectogramDataset(Dataset):
    def __init__(self, data_path, transform=None, max_samples_dict=None):
        """
        Initialize the MelSpectogramDataset class.

        Args:
            data_path (str): Path to the folder containing the mel spectrogram files.
            transform (callable): Optional transform to be applied to the mel spectrogram.
            max_samples_dict (dict): Dictionary specifying the maximum number of samples per genre.
                                     For example: {'rock': 1000, 'pop': 600}.
        """

        self.data_path = data_path
        self.transform = transform
        self.max_samples_dict = max_samples_dict

        # Sort genres alphabetically
        self.genres = sorted([d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))])

        self.file_paths = []
        self.labels = []

        for i, genre in enumerate(self.genres):
            genre_folder = os.path.join(data_path, genre)
            files = [f for f in os.listdir(genre_folder) if f.endswith('.npy')]

            # Check if this genre has a max_samples limit
            if max_samples_dict and genre in max_samples_dict:
                max_samples = max_samples_dict[genre]
                files = np.random.choice(files, size=min(max_samples, len(files)), replace=False).tolist()

            self.file_paths.extend([os.path.join(genre_folder, f) for f in files])
            self.labels.extend([i] * len(files))

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        mel_spectrogram = np.load(self.file_paths[idx])
        label = self.labels[idx]

        if self.transform:
            mel_spectrogram = self.transform(mel_spectrogram)

        return mel_spectrogram, label


def calculate_metrics(y_true, y_pred_probs, num_classes):
    """
    Calculate AUC, precision, recall, and F1 score for multiclass classification.

    Args:
        y_true (array-like): True labels.
        y_pred_probs (array-like): Predicted probabilities or logits.
        num_classes (int): Number of classes in the classification task.

    Returns:
        dict: A dictionary containing AUC, precision, recall, and F1 scores.
    """
    # Convert predicted probabilities to class predictions
    y_pred = y_pred_probs.argmax(axis=1)

    # Calculate metrics
    metrics = {
        "AUC": roc_auc_score(y_true, y_pred_probs, multi_class="ovr", average="macro"),
        "Precision": precision_score(y_true, y_pred, average="macro", zero_division=1),
        "Recall": recall_score(y_true, y_pred, average="macro"),
        "F1 Score": f1_score(y_true, y_pred, average="macro")
    }
    return metrics

def per_class_metrics(y_true, y_pred, num_classes):
    """
    Prints classification metrics for each class.

    Args:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.
        num_classes (int): Number of classes in the classification task.
    """
    report = classification_report(
        y_true,
        y_pred,
        zero_division=1,
        target_names=[f"Class {i}" for i in range(num_classes)]
    )
    print(report)







