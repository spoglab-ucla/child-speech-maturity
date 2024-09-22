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
# MASTER CODE


class AudioArrays(Dataset):

    def __init__(self, csv_file, root_dir, transform=None):

        # self.audios_df = pd.read_csv(csv_file)
        self.audios_df = csv_file
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
                                           root_dir='/your/root/dir')

    return transformed_dataset

def evaluate_model(model, dataloader):
    true_labels = []
    predicted_labels = []

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['input_values']
            labels = batch['label']

            outputs = model(inputs)
            logits = outputs.logits
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

    print('balanced_bab.csv')
    print("UAR: ", uar)

    return

def main():
    print('started')
    #label2id = {'Canonical': 0, 'Non-Canonical': 1, 'Crying': 2, 'Laughing': 3, 'Junk': 4}
    order = [0, 1, 2, 4, 3]
    class_names = ['Canonical', 'Non-Canonical', 'Crying', 'Junk', 'Laughing']

    test_dataset = return_custom_set()
    test_dataloader=DataLoader(test_dataset, batch_size=32)

    model = AutoModelForAudioClassification.from_pretrained("/checkpoint/path")

    # Set the model to evaluation mode
    model.eval()

    true_labels, predicted_labels = evaluate_model(model, test_dataloader)

    # STOP HERE if you just want to get UAR
    calculate_uar(true_labels, predicted_labels)
    # return

    # Generate confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels, labels=order)
    cm_percentages = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    # Plot the confusion matrix with percentages
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_percentages, annot=True, cmap='Blues', fmt='.2f', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Wav2Vec Accuracy on Maturity Dataset (%)')

    plt.savefig('cm.png')

    print('finished')

    # Display confusion matrix
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    # disp.plot() 

    # plt.show()  # Display the confusion matrix plot

if __name__ == "__main__":
    main()
