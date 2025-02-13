# child-speech-maturity

## This repository contains the code for creating and testing a transformers based model, w2V2-base, for classifying five different child speech vocalizations. Github has limits on file sizes, so rather than including the checkpoints, we have instead including the code for creating these models. Additionally, for privacy reasons, we have not included the dataset splits for either of the datasets we used ([BabbleCor](https://osf.io/rz4tx/) and [SpeechMaturity](https://osf.io/tf3hq/)). Please contact the authors of this paper if you would like access to the exact splits we used. The datasets for both models are in the format of [audio_file, label] where both columns are strings.

## Repository Layout
### There are two folders in this repository, and each contain the following:

### **w2v**
This folder contains the code to create and test the transformers models for W2V2-base.

- Train & Dev UAR Loop (W2V Base): W2V_Train_Loop.py

- Test UAR & Confusion Matrix (W2V Base): W2V_Confusion.py

- Train & Dev UAR Loop (W2V-LL4300): W2V-LL4300_Train_Loop.py
(uses fairseq_wav2vec.py in code which is in this folder and was sourced from this [repository](https://huggingface.co/lijialudew/wav2vec_LittleBeats_LENA))

- Test UAR & Confusion Matrix (W2V-LL4300): W2V-LL4300_Confusion.py
(uses ./checkpoint_best.pt which can be found in this [repository](https://huggingface.co/lijialudew/wav2vec_LittleBeats_LENA) as LL4300/checkpoint_best.pt)

### **confidence**
This folder contains the code for calculating additional statistics from both the human annotator data and model outputs.

Data set format for calculating UAR over all samples and for each environment, ROC/AUC cuves, confusion matrices, Cohen's kappa, Fleiss' kappa:

[audio_file,label,child_id,child_age,corpus_id,annotators,model_predicted_label,environment,logits,probabilities]

- audio_file: unique audio file ID

- label: true label (canonical, non-canonical, laugh, cry, or junk)

- child_id: unique child ID for the child associated with the sample

- corpus_id: unique corpus ID for the corpus associated with the sample

- annotators: an array of what each annotator labeled the clip as. Number of annotators varies between samples.

- model_predicted_label: the label that the model predicts

- logits: the outputted logits from the models
    
- probabilities: using the logits, the probability that the model assigns to each label for each clip

Calculating the Fleiss' and Cohens' Kappas:
- caculate_kappas.py

Calculating ROC/AUC curves, confusion matrices, and UARs
- calculate_cm_uar_rocs.py
