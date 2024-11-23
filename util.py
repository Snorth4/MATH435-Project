from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import pandas as pd
import numpy as np
import random, csv, os

TRAIN_SPLIT = 70
VALID_SPLIT = 15
TEST_SPLIT  = 15

def load_data():
    """
    Loads the data.
    
    Returns:
        peptide_data: The dataset of peptides.
    """

    dataset_file = 'Dataset/Peptides.csv'
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_file_path = os.path.join(current_dir, dataset_file)

    data = []
    with open(dataset_file_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
            
    # Convert to DataFrame
    df = pd.DataFrame(data)

    return df

# Dictionary for amino acid to integer mapping
amino_acid_dict = {
    #------------------------------>  0, 1, 2 represent padding, start, and end values respectively
    'A':  3, 'C':  4, 'D':  5, 'E':  6, 'F':  7, 'G':  8, 'H':  9, 'I': 10, 'K': 11, 'L': 12, 
    'M': 13, 'N': 14, 'P': 15, 'Q': 16, 'R': 17, 'S': 18, 'T': 19, 'V': 20, 'W': 21, 'Y': 22
}

def normalize_data(data):
    """
    Normalizes MIC and sequence_length columns to the range [0, 1].

    Args:
        data (pd.DataFrame): Input data.

    Returns:
        pd.DataFrame: Normalized data.
        scaler: Fitted MinMaxScaler.
    """
    scaler = MinMaxScaler()
    data[['MIC', 'sequence_length']] = scaler.fit_transform(data[['MIC', 'sequence_length']])  #-->  Normalize the MICs and sequence lengths

    data = map_seq_to_int(data)  #-->  Normalize the sequences

    return data, scaler

def map_seq_to_int(data):
    """
    Takes a dataset (Pandas DataFrame) with a 'sequence' column,
    maps each amino acid to an integer, and replaces the 'sequence' column 
    with the corresponding integer values.

    Args:
        dataset (pd.DataFrame): The dataset containing a 'sequence' column.

    Returns:
        pd.DataFrame: The modified dataset with sequences replaced by integers.
    """

    def encode_sequence(sequence):
        """Function to convert a single sequence to integers."""
        return [amino_acid_dict[aa] for aa in sequence]
    
    if isinstance(data, pd.DataFrame):  #------------------------------->  Ensure 'sequence' is a Pandas Series
        data['sequence'] = data['sequence'].apply(encode_sequence)  #-->  Apply the encoding function to the 'sequence' column
    else:
        raise TypeError("Input data must be a pandas DataFrame")
    
    return data

def split_data(data):
    """
    Split data into variables X and y for training, validation, and testing.

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """

    X = data[['MIC', 'sequence_length']].astype(np.float32).to_numpy()  #-->  Input features (MIC and sequence_length)
    y = data['sequence'].apply(lambda x: np.array(x))  #------------------>  Target labels (encoded / padded sequences)
    y = pad_sequences(y, padding='post', value=0, dtype='int32')  #--------->  Pad sequences to the same length

    # Add start and end tokens to sequences
    start_token = 1
    end_token = 2
    y = np.pad(y, [(0, 0), (1, 1)], constant_values=(start_token, end_token))  #-->  Add tokens at start and end

    # Split data into 70-15-15 split for train, val, test
    randint = random.randint(0, 4294967295)
    X_train, X_temp, y_train, y_temp = train_test_split(     X,      y, train_size=(TRAIN_SPLIT/100), random_state=randint)
    X_val,   X_test, y_val,   y_test = train_test_split(X_temp, y_temp, train_size=((VALID_SPLIT + TEST_SPLIT) / (VALID_SPLIT*100)), random_state=randint)

    return [X_train[:, 0], X_train[:, 1]], [X_val  [:, 0], X_val  [:, 1]], [X_test [:, 0], X_test [:, 1]], y_train, y_val, y_test