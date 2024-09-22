# child-speech-maturity

## This repository contains the code for creating and testing two different types of models, CNN and transformers based models, for classifying five different child speech vocalizations. Github has limits on file sizes, so rather than including the checkpoints for all the models, we have instead including the code for creating these models. Additionally, for privacy reasons, we have not included the dataset splits for either of the datasets we used ([BabbleCor](https://osf.io/rz4tx/) and [SpeechMaturity](https://osf.io/tf3hq/)). Please contact the authors of this paper if you would like access to the exact splits we used. The datasets for both models are in the format of [audio_file, label] where both columns are strings.

## Repository Layout
### There are three folders in this repository, and each contain the following:

### **cnn**
This folder contains the code to create and test the CNN. In the training loop, the performance of each epoch is saved and from that, the best performing checkpoint can be used in the test loop.

- Train & Dev UAR Loop: cnn_train.py
- Test UAR: cnn_test.py

### **w2v**
This folder contains the code to create and test the transformers models.

- Train & Dev UAR Loop (W2V Base): W2V_Train_Loop.py

- Test UAR & Confusion Matrix (W2V Base): W2V_Confusion.py

- Train & Dev UAR Loop (W2V-LL4300): W2V-LL4300_Train_Loop.py
(uses fairseq_wav2vec.py in code which is in this folder and was sourced from this [repository](https://huggingface.co/lijialudew/wav2vec_LittleBeats_LENA))

- Test UAR & Confusion Matrix (W2V-LL4300): W2V-LL4300_Confusion.py
(uses ./checkpoint_best.pt which can be found in this [repository](https://huggingface.co/lijialudew/wav2vec_LittleBeats_LENA) as LL4300/checkpoint_best.pt)

### **confidence**
This folder contains the code for adding the confidence levels to the existing datasets and calculating additional statistics from them.

Adding the confidence columns to the CSVs for humans and models:
- add_human_confidence_level.py
    - Dataset format for model outputs: [audio_file, label, confidence]
- add_model_confidence_level.py
    - Dataset format for human annotators: [audio_file, label, confidence, all_labels]

Calculating the Fleiss' and Cohens' Kappas:
- caculate_kappas.py
