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
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.metrics import recall_score
import seaborn as sns
from fairseq_wav2vec import FairseqWav2Vec2
import torchvision.datasets as dsets
import torch.optim as optim
import torch.nn as nn
from Label5 import FeedforwardNeuralNetModel  

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

        # get label
        maturity_label = self.audios_df.iloc[row_index, 1]

        sample = {'input_values': inputs, 'label': maturity_label} # feature and label

        if self.transform:
            sample = self.transform(sample)

        return sample

def return_custom_set():
    test = pd.read_csv('test.csv') # replace with your path to test csv
    label2id = {'Canonical': 0, 'Non-Canonical': 1, 'Crying': 2, 'Laughing': 3, 'Junk': 4}
    test['Label'] = test['Label'].replace(label2id) 

    transformed_dataset = AudioArrays(csv_file=test,
                                           root_dir='/u/home/m/madurya/project-spoglab')

    return transformed_dataset

def permute_flatten(x): # input the tensor
    # Permute the tensor to bring batch size to the first dimension: [32, 11, 28, 768]
    x = x.permute(1, 0, 2, 3)

    # Flatten the tensor, keeping the batch size (32) separate
    x_flattened = x.contiguous().view(x.size(0), -1)  # This results in shape [32, 11x28x768]

    return x_flattened

def evaluate_model(model, dataloader):
    true_labels = []
    predicted_labels = []

    save_path = "./checkpoint_best.pt"
    wav2vec_4300 = FairseqWav2Vec2(save_path)

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['input_values']
            labels = batch['label']

            outputs_wav2vec = wav2vec_4300(inputs)
            print('before', outputs_wav2vec.shape)

            # permute tensor
            outputs_wav2vec = permute_flatten(outputs_wav2vec)
            print('after', outputs_wav2vec.shape)

            outputs = model(outputs_wav2vec)
            logits = outputs #.logits
            _, predictions = torch.max(logits, dim=1)

            true_labels.extend(labels.numpy())
            predicted_labels.extend(predictions.numpy())

    # calculate_uar(true_labels, predicted_labels)

    unique_labels = np.unique(np.concatenate([true_labels, predicted_labels]))
    print("unique_labels: ", unique_labels)

    # print("true_labels: ", true_labels)
    # print("predicted_labels: ", true_labels)

    return true_labels, predicted_labels

def calculate_uar(true_labels, predicted_labels):
    unique_classes = np.unique(true_labels)

    recalls = []
    for class_label in unique_classes:
        true_class = (true_labels == class_label)
        predicted_class = (predicted_labels == class_label)
        
        # Calculate recall for the current class
        recall = recall_score(true_class, predicted_class, zero_division=0)
        recalls.append(recall)

    # Calculate unweighted average recall
    uar = np.mean(recalls)

    print("UAR: ", uar)

    return

def main():
    print('started')
    #label2id = {'Canonical': 0, 'Non-Canonical': 1, 'Crying': 2, 'Laughing': 3, 'Junk': 4}
    order = [0, 1, 2, 4, 3]
    class_names = ['Canonical', 'Non-Canonical', 'Crying', 'Junk', 'Laughing']

    test_dataset = return_custom_set()
    test_dataloader=DataLoader(test_dataset, batch_size=32)

    input_dim = 236544
    hidden_dim = 100 # can be anything
    output_dim = 5 # number of labels
    learning_rate = 0.001

    model = FeedforwardNeuralNetModel(input_dim, hidden_dim, output_dim)
    PATH = '/best/checkpoint/path' # checkpoint epoch 8 was the best
    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(model.state_dict().keys())
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    model.eval()

    true_labels, predicted_labels = evaluate_model(model, test_dataloader)

    # STOP HERE if you just want to get UAR
    calculate_uar(true_labels, predicted_labels)
    return
    # return

    # Generate confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels, labels=order)
    cm_percentages = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    # Plot the confusion matrix with percentages
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_percentages, annot=True, cmap='Blues', fmt='.2f', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('LL4300 Accuracy on Maturity Dataset (%)')

    plt.savefig('cm.png')

    print('finished')

    # Display confusion matrix
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    # disp.plot() 

    # plt.show()  # Display the confusion matrix plot

if __name__ == "__main__":
    main()
