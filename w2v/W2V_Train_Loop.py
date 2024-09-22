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

class AudioArrays(Dataset):

    def __init__(self, csv_file, root_dir, transform=None):

        # self.audios_df = pd.read_csv(csv_file) # deprecated once we added mapping word label --> number before obj is constructed
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

        sample = {'input_values': inputs, 'label': maturity_label} # feature and label

        if self.transform:
            sample = self.transform(sample)

        return sample

def return_custom_train():
    train = pd.read_csv('train.csv') # replace with your path to train csv
    label2id = {'Canonical': 0, 'Non-Canonical': 1, 'Crying': 2, 'Laughing': 3, 'Junk': 4}
    train['Label'] = train['Label'].replace(label2id) # this is compatible with both if Label column has numbers or word labelss

    transformed_dataset = AudioArrays(csv_file=train,
                                           root_dir='/your/root/dir')

    return transformed_dataset

def return_custom_test():
    dev = pd.read_csv('dev.csv') # replace with your path to dev csv
    label2id = {'Canonical': 0, 'Non-Canonical': 1, 'Crying': 2, 'Laughing': 3, 'Junk': 4}
    dev['Label'] = dev['Label'].replace(label2id) 

    transformed_dataset = AudioArrays(csv_file=dev,
                                           root_dir='/your/root/dir')

    return transformed_dataset

def compute_metrics(eval_pred):
    accuracy = evaluate.load("accuracy")

    predictions = np.argmax(eval_pred.predictions, axis=1)
    print(accuracy.compute(predictions=predictions, references=eval_pred.label_ids)) # just to print and compare

    # Get the predicted labels by taking the argmax along axis 1
    # predictions = np.argmax(eval_pred.predictions, axis=1)
    
    # Get the true labels from eval_pred.label_ids
    true_labels = eval_pred.label_ids

    unique_true = np.unique(true_labels)
    unique_pred = np.unique(predictions)

    print("unique: ", unique_true, " ", unique_pred)
    
    # Calculate recall for each class
    recalls = recall_score(true_labels, predictions, average=None)
    
    # Calculate the Unweighted Average Recall (UAR)
    uar = np.mean(recalls)
    
    # You can print or return the UAR or any other relevant information
    #print("Recalls for each class:", recalls)
    #print("UAR:", uar)

    # You can return a dictionary with additional metrics if needed
    return {'uar': uar}


def train_model(train_dataset, test_dataset):
    id2label = {'0': 'Canonical', '1': 'Non-Canonical', '2': 'Crying', '3': 'Laughing', '4': 'Junk'} 
    label2id = {'Canonical': 0, 'Non-Canonical': 1, 'Crying': 2, 'Laughing': 3, 'Junk': 4}

    model = AutoModelForAudioClassification.from_pretrained(
    "facebook/wav2vec2-base", num_labels=5, label2id=label2id, id2label=id2label
    )

    feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")

    training_args = TrainingArguments(
    output_dir="/checkpoint/path",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-5, # reduce learning rate from 3e-5? 
    per_device_train_batch_size=32, # increase batch size from 32?
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=32, # increase batch size from 32?
    num_train_epochs=10,
    warmup_ratio=0.1,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="uar", #was "accuracy" ... will default to greater is better
    #dataloader_num_workers = ___

    #push_to_hub=True,
    )

    # accelerate launch --config_file {/Users/your_username/Library/Caches/default_config.yaml} build_parallel_trainer.py
    # /Users/maduryasuresh/.cache/huggingface/accelerate/default_config.yaml
    # accelerator = Accelerator()
    accelerator = Accelerator()
    trainer = accelerator.prepare(Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=feature_extractor, # not necessary, probably not causing the issue
        compute_metrics=compute_metrics,
    ))
    trainer.add_callback(PrinterCallback) 

    trainer.train()


def main():
    
    train_dataset = return_custom_train()
    test_dataset = return_custom_test()

    train_model(train_dataset, test_dataset)
    



if __name__ == "__main__":
    main()
