# Transformers-Based Model (Fine-Tuned Wav2Vec)

## Training Code

train_wav2vec.py - code used to fine-tune Wav2Vec on the training sets below, runs testing loop with corresponding testing set, outputs checkpoints to desired destination

How to run training code:

- Specify the folder/file paths where indicated in the file (for model checkpoint saving and datasets)
- Training code uses accelerate library to speed up training. To configure your accelerate, refer to `[this StackOverflow](https://stackoverflow.com/questions/76675018/how-does-one-use-accelerate-with-the-hugging-face-hf-trainer)`.
    - If you don't want to use the accelerate library, just don't wrap the Trainer object in the Accelerator object (trainer = Trainer(...))
- Install the packages imported at the top of the file if needed (pip3 install ...)
- If using CPU instead of GPU, run this in the command line: export PYTORCH_ENABLE_MPS_FALLBACK=1
- If using accelerate, run the following command to train the model: accelerate launch --config_file {/Users/your_username/Library/Caches/default_config.yaml} train_wav2vec.py
    - otherwise, run python3 train_wav2vec.py
    - ensure you are using python 3.9 or lower (model was trained using python 3.8)

## Datasets

The CSVs contain number IDs in place of strings for the labels. Here is the mapping of ID to Label, which applies to all datasets in this folder: {'0': 'Canonical', '1': 'Non-Canonical', '2': 'Crying', '3': 'Laughing', '4': 'Junk'}

./MAT_TRAIN.csv - large training set taken from speech maturity dataset (maturity1 and maturity2)
./MAT_TEST.csv - large test set taken from speech maturity dataset (maturity1 and maturity2)

./balanced_bab.csv - smaller training set taken from BabbleCorpus and maturity2 (slight class balancing for Crying, Laughing, and Canonical classes)
./babblecor_test_ID.csv - smaller test set taken from BabbleCorpus

## Models

./checkpoint-3330 - best UAR Wav2Vec checkpoint trained on MAT_TRAIN.csv, with testing loop generating UAR from MAT_TEST.csv (both speech maturity dataset)

./checkpoint-328 - best UAR Wav2Vec checkpoint trained on balanced_bab.csv (BabbleCorpus and some of maturity2 dataset), with testing loop generating UAR from babblecor_test_ID.csv (BabbleCorpus dataset)
