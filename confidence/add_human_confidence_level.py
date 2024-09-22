import pandas as pd

# Load CSV files
clean_test = "clean_test.csv"
unclean_test = "unclean_test.csv"
mat_test_df = pd.read_csv(xxxx_test) #replace with whichever dataset you want to use from above
mat1_df = pd.read_csv('mat1.csv')
mat2_df = pd.read_csv('mat2.csv')

# Concatenate mat1 and mat2 dataframes
mat_df = pd.concat([mat1_df, mat2_df])

# Function to extract the filename from the audio_file path
def extract_filename(audio_file):
    return audio_file.split('/')[-1]

# Add a new column to the dataframes with just the filename
mat_test_df['filename'] = mat_test_df['audio_file'].apply(extract_filename)
mat_df['filename'] = mat_df['audio_file'].apply(extract_filename)

# Function to calculate confidence
def calculate_confidence(row, mat_df):
    filename = row['filename']
    label = row['label']
    
    # Find the corresponding row in the mat dataframe
    mat_row = mat_df[mat_df['filename'] == filename]
    
    if not mat_row.empty:
        majority_label = mat_row['majority_label_vocalization_type'].values[0]
        labels_vocalization_type = mat_row['labels_vocalization_type'].values[0]
        labels_list = labels_vocalization_type.split(',')
        
        # Replace NA with Junk
        labels_list = ['Junk' if x == 'NA' else x for x in labels_list]
        
        # Calculate confidence
        confidence = labels_list.count(label) / len(labels_list)
        
        return confidence, labels_list
    else:
        return None, None

# Apply function to mat_test_df
mat_test_df['confidence'], mat_test_df['all_labels'] = zip(*mat_test_df.apply(calculate_confidence, mat_df=mat_df, axis=1))

mat_test_df.drop(columns=['filename'], inplace=True)

clean_result_path = "human_clean.csv"
unclean_result_path = "human_unclean.csv"
# Save the updated dataframe to a new CSV
mat_test_df.to_csv(result_path, index=False)

print(mat_test_df)
