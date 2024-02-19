import pandas as pd
import os
import librosa
import torch
import torchaudio
from torchaudio.transforms import Resample
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


def check_class_balance(df):
    counter = [0, 0, 0, 0, 0] # Canonical, Non-canonical, Crying, Laughing, Junk

    for value in df['majority_label_vocalization_type']:
        if value == 'Canonical':
            counter[0] += 1
        if value == 'Non-Canonical':
            counter[1] += 1
        if value == 'Crying':
            counter[2] += 1
        if value == 'Laughing':
            counter[3] += 1
        if value == 'Junk':
            counter[4] += 1
    
    print(counter)
    # babble corpus: [1826, 5606, 823, 241, 4974] # Canonical, Non-canonical, Crying, Laughing, Junk
    # maturity1: [17531, 33098, 13276, 3565, 30576] # Canonical, Non-Canonical, Crying, Laughing, Junk

    # babble corpus
        # sample 1826 from non-canonical
        # upsample crying * 3, sample 1826
        # upsample laughing * 8, sample 1826
        # sample 1826 from junk
    # maturity1
        # sample 17531 from non-canonical
        # sample 4255 from crying, then add to crying
        # upsample laughing * 6, then sample 17531
        # sample 17531 from junk



def sampling(df):
    # babble corpus:
    """ df_canonical = df[df['Answer'] == 'Canonical']
    df_noncanonical = df[df['Answer'] == 'Non-canonical']
    df_crying = df[df['Answer'] == 'Crying']
    df_laughing = df[df['Answer'] == 'Laughing']
    df_junk = df[df['Answer'] == 'Junk']

    df_noncanonical_down = df_noncanonical.sample(n=1826, random_state=42)
    df_junk_down = df_junk.sample(n=1826, random_state=42)

    df_crying_up = pd.concat([df_crying, df_crying, df_crying], ignore_index=True)
    df_laughing_up = pd.concat([df_laughing, df_laughing, df_laughing, df_laughing, df_laughing, df_laughing, df_laughing, df_laughing], ignore_index=True)

    df_crying_down = df_crying_up.sample(n=1826, random_state=42)
    df_laughing_down = df_laughing_up.sample(n=1826, random_state=42)

    df_concat = pd.concat([df_canonical, df_noncanonical_down, df_junk_down, df_crying_down, df_laughing_down])
    print(df_concat.head)

    return df_concat """

    # maturity1:
    df_canonical = df[df['majority_label_vocalization_type'] == 'Canonical']
    df_noncanonical = df[df['majority_label_vocalization_type'] == 'Non-Canonical']
    df_crying = df[df['majority_label_vocalization_type'] == 'Crying']
    df_laughing = df[df['majority_label_vocalization_type'] == 'Laughing']
    df_junk = df[df['majority_label_vocalization_type'] == 'Junk']

    df_noncanonical_down = df_noncanonical.sample(n=17531, random_state=42)
    df_crying_down = df_crying.sample(n=4255, random_state=42)
    df_crying_up = pd.concat([df_crying, df_crying_down])
    df_laughing_up = pd.concat([df_laughing, df_laughing, df_laughing, df_laughing, df_laughing, df_laughing])
    df_laughing_down = df_laughing_up.sample(n=17531, random_state=42)
    df_junk_down = df_junk.sample(n=17531, random_state=42)

    df_concat = pd.concat([df_canonical, df_noncanonical_down, df_crying_up, df_laughing_down, df_junk_down])
    return df_concat
    



def remove_cols(df):
    # babble corpus:
    """ df_new = df[['clip_ID', 'Answer']]
    return df_new """

    # maturity1:
    df_new = df[['audio_file', 'majority_label_vocalization_type']]
    return df_new

def remove_no_label_entries(df):
    # Drop entries with "NO-LABEL" in the "Answer" column
    df_filtered = df[df['majority_label_vocalization_type'] != 'NO-LABEL']
    return df_filtered

def main():
    # df_bab = pd.read_csv('./private_metadata.csv')
    # print(df_bab.head)

    # df_mat = pd.read_csv('./maturity_data/maturity1.csv')
    # print(df_mat.head)

    # remove unecessary cols
    """ df_new = remove_cols(df_bab)
    print(df_new.head)
    df_new.to_csv('private_metadata_filtered.csv', index=False) """

    """ df_new = remove_cols(df_mat)
    print(df_new.head)
    df_new.to_csv('maturity1_filtered.csv', index=False) """

    # remove NO-LABEL sounds from maturity1
    """ df_mat = pd.read_csv('./maturity1_filtered.csv')
    df_filtered = remove_no_label_entries(df_mat)
    df_filtered.to_csv('./maturity1_filtered.csv', index=False) """

    # df_bab = pd.read_csv('./maturity1_filtered.csv')
    # check_class_balance(df_bab)

    """ df_bab = pd.read_csv('./private_metadata_filtered.csv')
    df_concat = sampling(df_bab)
    df_concat.to_csv('babble_corpus_balanced.csv', index=False) """

    """ df_mat = pd.read_csv('./maturity1_filtered.csv')
    df_concat = sampling(df_mat)
    df_concat.to_csv('maturity1_balanced.csv', index=False) """

    """ df_bab = pd.read_csv('./maturity1_balanced.csv')
    #df_bab.rename(columns={'clip_ID': 'audio_file'}, inplace=True)
    df_bab.rename(columns={'majority_label_vocalization_type': 'Label'}, inplace=True)

    print(df_bab.head)

    df_bab.to_csv('MAT.csv', index=False) """

    return

if __name__ == "__main__":
    main()