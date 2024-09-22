from fairseq_wav2vec import FairseqWav2Vec2
import os
import torch
import torchaudio
from torchaudio.transforms import Resample
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import librosa
from librosa.util import pad_center
from transformers import AutoModelForAudioClassification, TrainingArguments, Trainer, TrainerCallback, PrinterCallback, AutoFeatureExtractor, SequenceFeatureExtractor
import evaluate
from torch.utils.data import random_split
import copy
from accelerate import Accelerator
from sklearn.metrics import recall_score
import torch.nn as nn
import torchvision.datasets as dsets
import torch.optim as optim

# torch.backends.nnpack.enabled = False # 

class AudioArrays(Dataset):

    def __init__(self, csv_file, root_dir, transform=None):

        self.audios_df = csv_file # pd.read_csv(csv_file) 
        self.root_dir = root_dir
        self.transform = transform
        self.feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base") # load feature extractor

    def __len__(self):
        return len(self.audios_df)

    # __getitem__ to support the indexing such that dataset[i] can be used to get the ith sample.
    def __getitem__(self, row_index):

        get_clip = os.path.join(self.root_dir, self.audios_df.iloc[row_index, 0]) # get row_index row in column 0 (clip_id)

        y, sr = librosa.load(get_clip) # load with librosa using audioread for mp3 files
        y_mono = librosa.to_mono(y)  # convert to mono
        audio_array = torch.tensor(y_mono) # convert to torch tensor

        # Create a Resample transform
        resample_transform = Resample(orig_freq=sr, new_freq=16000)

        # Apply the transform to the audio waveform
        resampled_waveform = resample_transform(audio_array)

        # print((resampled_waveform.size()[0]).dtype)

        # pad 
        if resampled_waveform.size()[0] < 9217:
            # Pad it to 9217 if needed
            leftover = 9217 - resampled_waveform.size()[0]
            half = leftover // 2 
            
            if leftover % 2 == 0:
                resampled_waveform = torch.nn.functional.pad(resampled_waveform, (half, half), value=0)
            else:
                resampled_waveform = torch.nn.functional.pad(resampled_waveform, (half + 1, half), value=0)

        # featurize
        inputs = self.feature_extractor(resampled_waveform, sampling_rate=16000, max_length=9217, truncation=True, padding=True)['input_values'][0]
        #print(inputs.shape)

        # get label
        maturity_label = self.audios_df.iloc[row_index, 1]

        # sample = {'input_values': inputs, 'label': maturity_label} # feature and label

        if self.transform:
            sample = self.transform(sample)

        return inputs, maturity_label

class FeedforwardNeuralNetModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FeedforwardNeuralNetModel, self).__init__()
        # Linear function
        self.fc1 = nn.Linear(input_dim, hidden_dim) 
        # Non-linearity
        self.relu = nn.ReLU()
        # Linear function (readout)
        self.fc2 = nn.Linear(hidden_dim, output_dim)  
    
    def forward(self, x):
        # Linear function
        out = self.fc1(x)
        # Non-linearity
        out = self.relu(out)
        # Linear function (readout)
        out = self.fc2(out)
        return out


def return_custom_train():
    train = pd.read_csv('train.csv') # replace with your path to train csv
    label2id = {'Canonical': 0, 'Non-Canonical': 1, 'Crying': 2, 'Laughing': 3, 'Junk': 4}
    train['Label'] = train['Label'].replace(label2id) # this is compatible with both if Label column has numbers or word labelss

    transformed_dataset = AudioArrays(csv_file=train,
                                           root_dir='your/root/dir')


    return DataLoader(transformed_dataset, batch_size=32, shuffle=True)

def return_custom_test():
    dev = pd.read_csv('dev.csv') # replace with your path to dev csv
    label2id = {'Canonical': 0, 'Non-Canonical': 1, 'Crying': 2, 'Laughing': 3, 'Junk': 4}
    dev['Label'] = dev['Label'].replace(label2id) 

    transformed_dataset = AudioArrays(csv_file=dev,
                                           root_dir='your/root/dir')

    return DataLoader(transformed_dataset, batch_size=32, shuffle=True)

def permute_flatten(x): # input the tensor
    # Permute the tensor to bring batch size to the first dimension: [32, 11, 28, 768]
    x = x.permute(1, 0, 2, 3)

    # Flatten the tensor, keeping the batch size (32) separate
    x_flattened = x.contiguous().view(x.size(0), -1)  # This results in shape [32, 11x28x768]

    return x_flattened


def train_model(train_dataset, test_dataset):
    # load checkpoint
    save_path = "./checkpoint_best.pt"
    wav2vec_4300 = FairseqWav2Vec2(save_path)

    # instantiate FNN object
    input_dim = 11*28*768
    hidden_dim = 100 # can be anything
    output_dim = 5 # number of labels
    learning_rate = 0.001

    fnn = FeedforwardNeuralNetModel(input_dim, hidden_dim, output_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(fnn.parameters(), lr=learning_rate)

    # just see if we can iterate through dataset and get extracted features first
    # Training loop
    num_epochs = 10
    
    for epoch in range(num_epochs):
        """TRAINING LOOP"""
        fnn.train()
        running_loss = 0.0
        for inputs, labels in train_dataset:
            outputs_wav2vec = wav2vec_4300(inputs)
            # permute tensor
            outputs_wav2vec = permute_flatten(outputs_wav2vec)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs_fnn = fnn(outputs_wav2vec)
            loss = criterion(outputs_fnn, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_dataset):.4f}")

        """SAVE CHECKPOINT"""
        checkpoint_path = f"checkpoint_epoch_{epoch+1}.pt"
        torch.save({
            'epoch': epoch+1,
            'model_state_dict': fnn.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': running_loss/len(train_dataset),
        }, checkpoint_path)

        """TESTING LOOP - UAR AND ACCURACY"""
        fnn.eval()  # Set the model to evaluation mode
        correct = 0
        total = 0
        class_correct = torch.zeros(output_dim)
        class_total = torch.zeros(output_dim)
        with torch.no_grad():
            for inputs, labels in test_dataset:
                outputs_wav2vec = wav2vec_4300(inputs)
                # permute tensor
                outputs_wav2vec = permute_flatten(outputs_wav2vec)

                outputs_fnn = fnn(outputs_wav2vec)

                _, predicted = torch.max(outputs_fnn.data, 1)  # Get the predicted class
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                for label, prediction in zip(labels, predicted):
                    if label == prediction:
                        class_correct[label] += 1
                    class_total[label] += 1

        accuracy = 100 * correct / total  # Overall accuracy
        class_recall = class_correct / class_total
        uar = class_recall.mean().item() * 100  # Unweighted Average Recall

        print(f"Test Accuracy after epoch {epoch+1}: {accuracy:.2f}%")
        print(f"Test UAR after epoch {epoch+1}: {uar:.2f}%")

    print('done training')
    return


def main():
    print('get datasets')
    train_dataset = return_custom_train()
    test_dataset = return_custom_test()

    print('begin model training')
    train_model(train_dataset, test_dataset)



if __name__ == "__main__":
    main()




# print(outputs.shape) # torch.Size([11, 32, 28, 768]) ... permute it so that batch size is in the beginning
# print(labels)
