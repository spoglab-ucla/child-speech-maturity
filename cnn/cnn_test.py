import numpy as np
import librosa
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import os
from tqdm import tqdm
from torch.utils.data import random_split, Dataset, DataLoader
from sklearn.metrics import recall_score
import matplotlib.pyplot as plt
import csv
from torchvision.models import resnet34, ResNet34_Weights

from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

label_mapping = {'Canonical': 0, 'Non-canonical': 1, 'Crying': 2, 'Junk': 3, 'Laughing': 4}

# --------------------FUNCTIONS TO SET UP AUDIO CLIPS-------------------
# Function to get mel spectrogram
def get_melspectrogram_db(file_path):
    wav, sr = librosa.load(file_path)

    # change to mono
    if len(wav.shape) > 1:
        wav = librosa.to_mono(wav)

    # resample to 16000
    wav_resampled = librosa.resample(wav, orig_sr=sr, target_sr=16000)

    if wav_resampled.size < 9217:
        leftover = 9217 - wav_resampled.size
        half = leftover // 2

        if leftover % 2 == 0:
            wav_resampled = np.pad(wav_resampled, (half, half), mode='edge')
        else:
            wav_resampled = np.pad(wav_resampled, (half + 1, half), mode='edge')

    mel_spec = librosa.feature.melspectrogram(y=wav_resampled, sr=16000)

    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec_db

# Function to convert spectrogram to image
def spec_to_image(spec, eps=1e-6):
    mean = spec.mean()
    std = spec.std()
    spec_norm = (spec - mean) / (std + eps)
    spec_min, spec_max = spec_norm.min(), spec_norm.max()
    spec_scaled = 255 * (spec_norm - spec_min) / (spec_max - spec_min)
    spec_scaled = spec_scaled.astype(np.uint8)
    return spec_scaled

# Define Dataset Class
class AudioArrays(Dataset):
    def __init__(self, base, df, in_col, out_col, frames_per_second=128, max_duration=0.5):
        self.df = df
        self.data = []
        self.labels = []
        self.frames_per_second = frames_per_second
        self.max_frames = int(frames_per_second * max_duration)

        self.label_mapping = label_mapping

        for ind in tqdm(range(len(df))):
            row = df.iloc[ind]
            file_path = os.path.join(base, row[in_col])
            spec = spec_to_image(get_melspectrogram_db(file_path))[np.newaxis, ...]

            # Pad or truncate the spectrogram to the target number of frames
            if spec.shape[2] < self.max_frames:
                leftover = self.max_frames - spec.shape[2]
                half = leftover // 2

                if leftover % 2 == 0:
                    spec = np.pad(spec, ((0, 0), (0, 0), (half, half)), mode='constant')
                else:
                    spec = np.pad(spec, ((0, 0), (0, 0), (half + 1, half)), mode='constant')

            label = self.label_mapping[row[out_col]]
            # Convert label to one-hot encoding
            one_hot_label = np.zeros(len(label_mapping))
            one_hot_label[label] = 1
            self.data.append(spec)
            self.labels.append(np.float32(one_hot_label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], torch.tensor(self.labels[idx], dtype=torch.float32)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ------------IMPORTING DATA FILES--------------------------------------------------------------
# Set prefix for file paths
prefix = '/your/file/path/prefix'

train = pd.read_csv('/training/data/path')
test = pd.read_csv('/test/data/path')


# Create datasets and data loaders
train_dataset = AudioArrays(prefix, train, 'audio_file', 'label', frames_per_second=128, max_duration=0.5)
test_dataset = AudioArrays(prefix, test, 'audio_file', 'label', frames_per_second=128, max_duration=0.5)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True, num_workers=4)

# ------------------------LOADING MODEL-------------------------------------------------------------
# Instantiate the model
resnet_model = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
resnet_model.fc = nn.Linear(512, 5)
resnet_model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

def remove_module_prefix(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('module.'):
            new_key = key[len('module.'):]
        else:
            new_key = key
        new_state_dict[new_key] = value
    return new_state_dict

# Load the saved state dict after removing the "module" prefix
checkpoint = torch.load('/path/to/checkpoint')
new_state_dict = remove_module_prefix(checkpoint['model_state_dict'])

# Load the modified state dict into the model
resnet_model.load_state_dict(new_state_dict)

resnet_model = resnet_model.to(device)

# ---------------------------------TESTING LOOP ON TRAIN SET-------------------------------
resnet_model.eval()
all_labels = []
all_predicted = []

with torch.no_grad():
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device, dtype=torch.float32), labels.to(device, dtype=torch.float32)
        outputs = resnet_model(inputs)
        probabilities = torch.softmax(outputs, dim=1)
        threshold = 0.5

        # Convert probabilities to binary predictions
        predicted = (probabilities >= threshold).float()

        all_labels.extend(labels.cpu().numpy())
        all_predicted.extend(predicted.cpu().numpy())

# Convert lists to numpy arrays
all_labels = np.array(all_labels)
all_predicted = np.array(all_predicted)

# Compute recall for each class
recall_per_class = recall_score(all_labels, all_predicted, average=None)

# Compute UAR
train_uar = np.mean(recall_per_class)

print(f"Unweighted Average Recall (UAR) on train set: {train_uar * 100}%")

# ---------------------------------TESTING LOOP ON TEST SET-------------------------------

all_labels = []
all_predicted = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device, dtype=torch.float32), labels.to(device, dtype=torch.float32)
        outputs = resnet_model(inputs)
        probabilities = torch.softmax(outputs, dim=1)
        threshold = 0.5

        # Convert probabilities to binary predictions
        predicted = (probabilities >= threshold).float()

        all_labels.extend(labels.cpu().numpy())
        all_predicted.extend(predicted.cpu().numpy())

# Convert lists to numpy arrays
all_labels = np.array(all_labels)
all_predicted = np.array(all_predicted)

# Compute recall for each class
recall_per_class = recall_score(all_labels, all_predicted, average=None)

# Compute UAR
test_uar = np.mean(recall_per_class)

print(f"Unweighted Average Recall (UAR) on test set: {test_uar * 100}%")

# ---------------------------PLOTTING CM FOR TEST SET ONLY-------------------------------------------
# Assuming 'true_labels' are the true labels and 'predicted_labels' are the predicted labels
# You need to convert them from tensors to numpy arrays for use with confusion_matrix
# true_labels = np.argmax(all_labels, axis=1)
# predicted_labels = np.argmax(all_predicted, axis = 1)

# # Calculate the confusion matrix
# cm = confusion_matrix(true_labels, predicted_labels)

# # Define class labels (replace with your actual class labels)
# class_labels = ['Canonical', 'Non-Canonical', 'Crying', 'Junk', 'Laughing']

# # Calculate percentages for each cell in the confusion matrix
# cm_percentages = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

# # Plot the confusion matrix with percentages
# plt.figure(figsize=(8, 6))
# sns.heatmap(cm_percentages, annot=True, cmap='Blues', fmt='.2f', xticklabels=class_labels, yticklabels=class_labels)
# plt.xlabel('Predicted')
# plt.ylabel('True')
# plt.title('CNN Accuracy(%)')

# plt.savefig('cm.png')