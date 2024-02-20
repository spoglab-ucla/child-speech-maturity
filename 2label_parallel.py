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
# MASTER CODE



class AudioArrays(Dataset):

    def __init__(self, csv_file, root_dir, transform=None):

        self.audios_df = pd.read_csv(csv_file)
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
        #print(audio_array.shape)

        # Create a Resample transform
        resample_transform = Resample(orig_freq=sr, new_freq=16000)

        # Apply the transform to the audio waveform
        resampled_waveform = resample_transform(audio_array)

        # pad 
        # if (resampled_waveform.size())[1] != 9216:
        if resampled_waveform.size()[0] < 12701:
            # Pad it to 9216 if needed
            leftover = 12701 - resampled_waveform.size()[0]
            half = leftover // 2 
            #left_half = half
            
            if leftover % 2 == 0:
                resampled_waveform = torch.nn.functional.pad(resampled_waveform, (half, half), value=0)
            else:
                resampled_waveform = torch.nn.functional.pad(resampled_waveform, (half + 1, half), value=0)

        # featurize
        inputs = self.feature_extractor(resampled_waveform, sampling_rate=16000, max_length=12701, truncation=True, padding=True)['input_values'][0]
        #print(inputs.shape)

        # get label
        maturity_label = self.audios_df.iloc[row_index, 1]
        # maturity_class = 0

        sample = {'input_values': inputs, 'label': maturity_label} # feature and label

        if self.transform:
            sample = self.transform(sample)

        return sample


def test_custom(transformed_dataset):

    for i, sample in enumerate(transformed_dataset):
        print(i, len(sample['input_values']), type(sample['input_values']), sample['label'])
        # shape of tensor is [channels, samples]
        print((sample['input_values'])[0]) # yup, prints out some non zero number!

        #if (sample['input_values'].size()[0] != 1):
        #    print(i, sample['input_values'].size(), type(sample['input_values']), sample['label'])

        if i == 3:
            break



def return_custom_set():
    transformed_dataset = AudioArrays(csv_file='./Maturity_1_2Class.csv', # '../Audio_Clips_ID_2Classes.csv'
                                           root_dir='/u/home/m/madurya/project-spoglab/speech_maturity_data') # '../BabbleCor_clips'

    return transformed_dataset

def compute_metrics(eval_pred):
    accuracy = evaluate.load("accuracy")

    predictions = np.argmax(eval_pred.predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=eval_pred.label_ids)


def train_model(train_dataset, test_dataset):
    """ id2label = {'0': 'Canonical', '1': 'Non-canonical', '2': 'Other'}
    label2id = {'Canonical': 0, 'Non-canonical': 1, 'Other': 2} """
    id2label = {'0': 'Non-canonical', '1': 'Canonical'} # FLIP THIS IN THE BABBLE COR DATASET ... was orginally switched and trained the model on it on 1/31
    label2id = {'Non-canonical': 0, 'Canonical': 1}

    model = AutoModelForAudioClassification.from_pretrained(
    "facebook/wav2vec2-base", num_labels=2, label2id=label2id, id2label=id2label
    )

    feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")

    training_args = TrainingArguments(
    output_dir="my_model_ver_1",
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
    metric_for_best_model="accuracy",
    #dataloader_num_workers = ___

    #push_to_hub=True,
    )

    # accelerate launch --config_file {/Users/your_username/Library/Caches/default_config.yaml} build_parallel_trainer.py
    # /Users/maduryasuresh/.cache/huggingface/accelerate/default_config.yaml
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
    # test_custom()

    custom_dataset = return_custom_set()
    #test_custom(custom_dataset)
    #return

    # split into train and test
    train_size = int(0.8 * len(custom_dataset))
    test_size = len(custom_dataset) - train_size
    torch.manual_seed(42) # set random seed
    train_dataset, test_dataset = random_split(custom_dataset, [train_size, test_size])

    train_model(train_dataset, test_dataset)
    



if __name__ == "__main__":
    main()