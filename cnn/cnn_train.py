import numpy as np
import librosa
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import os
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
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

# Update label mapping for 5 labels
label_mapping = {'Canonical': 0, 'Non-canonical': 1, 'Crying': 2, 'Junk': 3, 'Laughing': 4}

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
            wav_resampled = np.pad(wav_resampled, (half, half), mode='constant')
        else:
            wav_resampled = np.pad(wav_resampled, (half + 1, half), mode='constant')

    mel_spec = librosa.feature.melspectrogram(y=wav_resampled, sr=16000) #pad mode is here lol
    #n_fft = 512
    #step size = 256
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
                    spec = np.pad(spec, ((0, 0), (0, 0), (half, half)), mode='edge')
                else:
                    spec = np.pad(spec, ((0, 0), (0, 0), (half + 1, half)), mode='edge')

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

# Set prefix for file paths
prefix = '/your/file/path/prefix'

train = pd.read_csv('/training/data/path')
dev = pd.read_csv('/dev/data/path')

def train_model(rank, world_size):
    # Initialize process group
    dist.init_process_group(backend="gloo", init_method="tcp://localhost:12345", rank=rank, world_size=world_size)

    # Set random seed
    torch.manual_seed(42)

    # Load data
    train_dataset = AudioArrays(prefix, train, 'audio_file', 'label', frames_per_second=128, max_duration=0.5)
    dev_dataset = AudioArrays(prefix, dev, 'audio_file', 'label', frames_per_second=128, max_duration=0.5)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    dev_sampler = torch.utils.data.distributed.DistributedSampler(dev_dataset, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(train_dataset, batch_size=16, sampler=train_sampler, num_workers=4)
    dev_loader = DataLoader(dev_dataset, batch_size=16, sampler=dev_sampler, num_workers=4)

    # Load Pre-trained ResNet model
    resnet_model = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
    resnet_model.fc = nn.Linear(512,5)
    resnet_model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    # resnet_model.to(rank)

    # Wrap the model with DistributedDataParallel
    ddp_model = DDP(resnet_model)

    # Define loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(ddp_model.parameters(), lr=0.0001)
    epochs = 10

    # Training loop
    epoch_info = []

    for epoch in range(epochs):
        ddp_model.train()
        train_sampler.set_epoch(epoch)
        tqdm_train_loader = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}', leave=False)
        for inputs, labels in tqdm_train_loader:
            inputs, labels = inputs.float(), labels.float()

            optimizer.zero_grad()
            outputs = ddp_model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

        if rank == 0:
            ddp_model.eval()

            correct = 0
            total = 0
            all_labels = []
            all_predicted = []
            with torch.no_grad():
                for inputs, labels in dev_loader:
                    inputs, labels = inputs.float(), labels.float()
                    outputs = ddp_model(inputs)
                    probabilities = torch.softmax(outputs, dim=1)
                    threshold = 0.5

                    # Convert probabilities to binary predictions
                    predicted = (probabilities >= threshold).float()

                    #compute accuracy
                    total += labels.size(0)
                    correct += (predicted == labels).all(dim=1).sum().item()

                    #computer UAR
                    all_labels.extend(labels.cpu().numpy())
                    all_predicted.extend(predicted.cpu().numpy())

                accuracy = (correct / total) * 100
                print(f"Epoch {epoch + 1}: Accuracy on dev set: {accuracy}%")

                # Convert lists to numpy arrays
                all_labels = np.array(all_labels)
                all_predicted = np.array(all_predicted)

                # Compute recall for each class
                recall_per_class = recall_score(all_labels, all_predicted, average=None)

                # Compute UAR
                uar = np.mean(recall_per_class)

                print(f"Epoch {epoch + 1}: Unweighted Average Recall (UAR) on dev set: {uar * 100}%")

                # Save epoch information
                epoch_info.append({'Epoch': epoch + 1, 'Accuracy': accuracy, 'UAR': uar})

                true_labels = np.argmax(all_labels, axis=1)
                predicted_labels = np.argmax(all_predicted, axis = 1)

                # Calculate the confusion matrix
                cm = confusion_matrix(true_labels, predicted_labels)

                # Define class labels (replace with your actual class labels)
                class_labels = ['Canonical', 'Non-canonical', 'Crying', 'Junk', 'Laughing']

                # Calculate percentages for each cell in the confusion matrix
                cm_percentages = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

                # Plot the confusion matrix with percentages
                plt.figure(figsize=(8, 6))
                sns.heatmap(cm_percentages, annot=True, cmap='Blues', fmt='.2f', xticklabels=class_labels, yticklabels=class_labels)
                plt.xlabel('Predicted')
                plt.ylabel('True')
                plt.title(f'CNN Accuracy on ___ Dataset (%) for epoch {epoch+1}')

                plt.savefig(f'cm_{epoch+1}.png')


                # Save checkpoint
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': ddp_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch_info': epoch_info
                }
                torch.save(checkpoint, f'checkpoint_epoch_{epoch + 1}.pt')
            
            with open('epoch_information.csv', 'a', newline='') as csvfile:
                fieldnames = ['Epoch', 'Accuracy', 'UAR']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for info in epoch_info:
                    writer.writerow(info)
    # Cleanup
    dist.destroy_process_group()

def main():
    world_size = 8  # Number of processes (can be number of GPUs)
    mp.spawn(train_model, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()

    #will have a runtime error bc one process will terminate and then the other just hang for the last epoch
    #https://github.com/pytorch/pytorch/issues/30439