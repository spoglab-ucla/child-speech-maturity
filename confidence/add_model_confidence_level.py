import os
import torch
import torchaudio
from torchaudio.transforms import Resample
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
import librosa
from tqdm import tqdm

label2id = {'Canonical': 0, 'Non-Canonical': 1, 'Crying': 2, 'Laughing': 3, 'Junk': 4}
id2label = {v: k for k, v in label2id.items()}

class AudioArrays(Dataset):

    def __init__(self, csv_file, root_dir, transform=None):
        self.audios_df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base") # load feature extractor

    def __len__(self):
        return len(self.audios_df)

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
        maturity_label = label2id[maturity_label]

        sample = {'input_values': inputs, 'label': maturity_label} # feature and label

        if self.transform:
            sample = self.transform(sample)

        return sample

def return_custom_set():
    clean_test = "clean_test.csv"
    unclean_test = "unclean_test.csv"
    transformed_dataset = AudioArrays(csv_file=weak_test,
                                      root_dir='/your/root/dir')

    return transformed_dataset

def evaluate_model(model, dataloader):
    true_labels = []
    predicted_labels = []
    confidences = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            inputs = batch['input_values']
            labels = batch['label']

            outputs = model(inputs)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=1)
            confidence, predictions = torch.max(probabilities, dim=1)

            true_labels.extend(labels.numpy())
            predicted_labels.extend(predictions.numpy())
            confidences.extend(confidence.numpy())

    return true_labels, predicted_labels, confidences

def main():
    print('started')

    test_dataset = return_custom_set()
    test_dataloader = DataLoader(test_dataset, batch_size=32)

    clean_checkpoint_path = "/path/to/checkpoint"
    unclean_checkpoint_path = "/path/to/checkpoint"

    model = AutoModelForAudioClassification.from_pretrained(unclean_checkpoint_path)

    # Set the model to evaluation mode
    model.eval()

    true_labels, predicted_labels, confidences = evaluate_model(model, test_dataloader)

    # Create a new dataframe to store the results
    results_df = pd.DataFrame({
        'audio_file': test_dataset.audios_df.iloc[:, 0],
        'label': [id2label[label] for label in predicted_labels],
        'confidence': confidences
    })

    clean_result_path = "model_clean.csv"
    unclean_result_path = "model_unclean.csv"
    # Save the dataframe to a new CSV file
    results_df.to_csv(xxxx_result_path, index=False)

    print("Results saved to " + xxxx_result_path)

if __name__ == "__main__":
    main()
