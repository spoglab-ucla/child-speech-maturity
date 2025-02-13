import numpy as np
import pandas as pd
import os
import ast
from statsmodels.stats.inter_rater import fleiss_kappa
from statsmodels.stats import inter_rater as irr
from irrCAC.raw import CAC
from sklearn.metrics import cohen_kappa_score

# Load CSV files
clean_test_path = "path.csv"
unclean_test_path = "path.csv"

# Mapping of labels
label_mapping = {"Non-Canonical": 1, "Canonical": 2, "Laughing": 3, "Crying": 4, "Junk": 0}

# Function to calculate Fleiss' kappa for different numbers of annotators
def calc_fleiss(df):
    results = {}
    max_annotators = min(5, df["annotators"].apply(lambda x: len(x.split(','))).max())
    for num_annotators in range(1, max_annotators + 1):
        subset_df = df[df["annotators"].apply(lambda x: len(x.split(','))) == num_annotators]
        if subset_df.empty:
            continue
        
        annotators = subset_df["annotators"].to_numpy()
        ratings = [labels.split(',')[:5] for labels in annotators]
        
        mapped_ratings = []
        for lst in ratings:
            mapped_lst = [label_mapping.get(item, np.nan) for item in lst]
            mapped_ratings.append(mapped_lst)
        
        weight_matrix = [[1.0, 0.0, 0.3, 0.3, 0.3], [0.0, 1.0, 0.3, 0.3, 0.3], [0.3, 0.3, 1.0, 0.7, 0.7], [0.3, 0.3, 0.7, 1.0, 0.7], [0.3, 0.3, 0.7, 0.7, 1.0]]
        result = CAC(pd.DataFrame(mapped_ratings), weights=weight_matrix)
        results[num_annotators] = result.fleiss()
    return results

# Function to calculate Cohen's kappa for different numbers of annotators
def calc_cohens(df):
    results = {}
    df = df[df["model_predicted_label"].notna()]
    max_annotators = min(5, df["annotators"].apply(lambda x: len(x.split(','))).max())
    for num_annotators in range(1, max_annotators + 1):
        subset_df = df[df["annotators"].apply(lambda x: len(x.split(','))) == num_annotators]
        if subset_df.empty:
            continue
        
        human_annotations = subset_df["annotators"].apply(lambda x: x.split(',')[:5])
        model_predictions = subset_df["model_predicted_label"].map(label_mapping).dropna()
        
        valid_indices = human_annotations.index.intersection(model_predictions.index)
        human_annotations = human_annotations.loc[valid_indices]
        model_predictions = model_predictions.loc[valid_indices]
        
        kappas = []
        for i in range(num_annotators):
            annotator_labels = human_annotations.apply(lambda x: label_mapping.get(x[i], np.nan) if len(x) > i else np.nan).dropna()
            model_labels = model_predictions.loc[annotator_labels.index]
            
            if not annotator_labels.empty and not model_labels.empty:
                kappa = cohen_kappa_score(annotator_labels, model_labels)
                kappas.append(kappa)
        
        if kappas:
            average_kappa = np.nanmean(kappas)
            std_kappa = np.nanstd(kappas)
            results[num_annotators] = (kappas, average_kappa, std_kappa)
    return results

# Process clean and unclean test sets
for test_path in [clean_test_path, unclean_test_path]:
    test_df = pd.read_csv(test_path)
    
    print(f"Processing {test_path}...\n")
    
    # Compute Fleiss' kappa for different annotator numbers
    fleiss_results = calc_fleiss(test_df)
    for num_annotators, result in fleiss_results.items():
        print(f"Fleiss' Kappa for {num_annotators} annotators:")
        print("Kappa: " + str(result['est']['coefficient_value']))
        print("Confidence Interval: " + str(result['est']['confidence_interval']))
        print("\n")
    
    # Compute Cohen's kappa for different annotator numbers
    cohens_results = calc_cohens(test_df)
    for num_annotators, (kappas, avg_kappa, std_kappa) in cohens_results.items():
        print(f"Cohen's Kappa for {num_annotators} annotators vs model:", kappas)
        print(f"Average Cohen's Kappa for {num_annotators} annotators:", avg_kappa)
        print(f"SD: {std_kappa}")
        print("\n")
