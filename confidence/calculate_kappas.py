#fleiss' kappa between annotators
#cohens kappa between each annotator and model and then average

#micro weight averages roc
#de-aggregated rocs


#edit the csv so its sorted by number of annotators per label
#essentially subjects as rows, categories as columns
#remove the model id column

import numpy as np 
import pandas as pd 
from statsmodels.stats.inter_rater import fleiss_kappa
from statsmodels.stats import inter_rater as irr
import ast
from irrCAC.raw import CAC
from collections import Counter
from sklearn.metrics import cohen_kappa_score

human_strong_only_path = "/u/project/spoglab/theoz/confidence/result_csvs/mat_test_human_strong_only.csv"
human_weak_included_path = "/u/project/spoglab/theoz/confidence/result_csvs/mat_test_human_weak_included.csv"

model_strong_only_path = "/u/project/spoglab/theoz/confidence/result_csvs/mat_test_model_strong_only.csv"
model_weak_included_path = "/u/project/spoglab/theoz/confidence/result_csvs/mat_test_model_weak_included.csv" 

df_human_strong = pd.read_csv(human_strong_only_path)
df_human_weak = pd.read_csv(human_weak_included_path)

df_model_strong = pd.read_csv(model_strong_only_path)
df_model_weak = pd.read_csv(model_weak_included_path)

max_length = 3
mapping = {'Canonical': 0, 'Non-Canonical': 1, 'Crying': 2, 'Laughing': 3, 'Junk': 4, np.nan: 5}

'''
fleiss' kappa
'''
def calc_fleiss(df):
    annotators = df["all_labels"].to_numpy()
    ratings = [ast.literal_eval(item) for item in annotators]
    #make an array of the annotator labels, map them to the label ids
    #then turn it into categories as columns instead of raters as columns

    mapping = {'Canonical': 0, 'Non-Canonical': 1, 'Crying': 2, 'Laughing': 3, 'Junk': 4}

    mapped_ratings = []
    for lst in ratings:
        if len(lst) <= max_length:
            mapped_lst = [mapping[item] for item in lst]  # Map the categories 
            # Pad with NaN if the list is shorter than the max length
            padded_lst = mapped_lst + [np.nan] * (max_length - len(mapped_lst))
            mapped_ratings.append(padded_lst)
    # print(mapped_ratings[0:3])

    weight_matrix = [[ 1.0,0.0,0.3,0.3,0.3], [ 0.0,1.0,0.3,0.3,0.3],[ 0.3,0.3,1.0,0.7,0.7],[ 0.3,0.3,0.7,1.0,0.7],[ 0.3,0.3,0.7,0.7,1.0]]
    result = CAC(pd.DataFrame(mapped_ratings), weights= weight_matrix)
    return result.fleiss()

print("Fleiss' Kappa for Human Annotators (Strong Majority Only):")
print("Kappa: " + str(calc_fleiss(df_human_strong)['est']['coefficient_value']))
print("Confidence Interval: " + str(calc_fleiss(df_human_strong)['est']['confidence_interval']))
print("\n")
print("Fleiss' Kappa for Human Annotators (Weak Majority Included):")
print("Kappa: " + str(calc_fleiss(df_human_weak)['est']['coefficient_value']))
print("Confidence Interval: " + str(calc_fleiss(df_human_weak)['est']['confidence_interval']))
print("\n")

'''
    cohen's kappa
'''
def calc_cohens(df_human, df_model):
    dropped_indices = []

    for idx, row in df_human.iterrows():
        ratings = ast.literal_eval(row["all_labels"])
        if len(ratings) > max_length:
            dropped_indices.append(idx)
        if len(ratings) < max_length:
            # Pad with NaN if the list is shorter than the max length
            padded_ratings = ratings + [np.nan] * (5 - len(ratings))
            df_human.at[idx, "all_labels"] = padded_ratings
        if len(ratings) == max_length:
            df_human.at[idx, "all_labels"] = ratings

    for idx, row in df_model.iterrows():
        df_model.at[idx, "label"] = mapping[row["label"]]
    
    valid_human_ratings = {0:[], 1:[], 2:[], 3:[], 4:[]}
    valid_model_ratings= []

    for i in range(max_length):
        for idx_human, row_human in df_human.iterrows():
            if idx_human not in dropped_indices:
                valid_human_ratings[i].append(mapping[row_human["all_labels"][i]])

    for idx_model, row_model in df_model.iterrows():  
        if idx_model not in dropped_indices:         
            valid_model_ratings.append(row_model["label"])

    kappas = []
    for i in range(max_length):
        #if a human annotator says nan, that means there wasn't an ith annotator for that clip, so don't include that 
        ith_human = []
        ith_model = []
        for idx, (val_human, val_model) in enumerate(zip(valid_human_ratings[i], valid_model_ratings)):
            if val_human != 5:
                ith_human.append(val_human)
                ith_model.append(val_model)
        kappa = cohen_kappa_score(ith_human, ith_model)
        kappas.append(kappa)
    average_kappa = np.mean(kappas)
    mean_kappa = np.mean(kappas)
    std_kappa = np.std(kappas)
    
    # Calculate confidence interval
    # z = 1.96  # For 95% confidence interval
    # ci = z * (std_kappa / np.sqrt(len(kappas)))
    # ci_lower = mean_kappa - ci
    # ci_upper = mean_kappa + ci
    
    return kappas, mean_kappa, std_kappa

# Example usage
strong_kappas, average_strong_kappa, strong_std = calc_cohens(df_human_strong, df_model_strong)

print("Cohen's Kappa for each annotator vs model (strong only):", strong_kappas)
print("Average Cohen's Kappa (strong only):", average_strong_kappa)
# print("Confidence Interval for Average Cohen's Kappa (strong only):", strong_ci)
print("SD: ", strong_std)
print("\n")

weak_kappas, average_weak_kappa, weak_std = calc_cohens(df_human_weak, df_model_weak)

print("Cohen's Kappa for each annotator vs model (weak included):", weak_kappas)
print("Average Cohen's Kappa (weak included):", average_weak_kappa)
# print("Confidence Interval for Average Cohen's Kappa (weak included):", weak_ci)
print("SD: ", weak_std)