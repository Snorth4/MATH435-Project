from sklearn.model_selection import train_test_split
import tensorflow as tf
import pandas as pd
import numpy as np
import random, csv, os

def load_data():
    """
    Loads the data.
    
    Returns:
        peptide_data: The dataset of peptides.
    """

    # dataset_file = 'my_dataset/dummy_data/[test_ds].csv'
    dataset_file = 'Dataset/Peptides.csv'
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_file_path = os.path.join(current_dir, dataset_file)

    data = []
    with open(dataset_file_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
            
    # Convert to DataFrame for easier handling
    df = pd.DataFrame(data)

    return df

# Dictionary for amino acid to integer mapping
amino_acid_dict = {
    # 0, 1, & 2 represent padding, start, and end values
    'A':  3, 'C':  4, 'D':  5, 'E':  6, 'F':  7, 'G':  8, 'H':  9, 'I': 10, 'K': 11, 'L': 12, 
    'M': 13, 'N': 14, 'P': 15, 'Q': 16, 'R': 17, 'S': 18, 'T': 19, 'V': 20, 'W': 21, 'Y': 22

    # The following are not recognized amino acids, but might be recognized in dummy data
    # They should be commented out by the end of this project
    # ,'B': 20, 'J': 21, 'O': 22, 'U': 23, 'X': 24, 'Z': 25
}

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
    
    if isinstance(data, pd.DataFrame):  # Ensure 'sequence' is a Pandas Series
        data['sequence'] = data['sequence'].apply(encode_sequence)  # Apply the encoding function to the 'sequence' column
    else:
        raise TypeError("Input data must be a pandas DataFrame")
    
    return data

def clean_data(data):
    """
    Cleans the dataset by handling missing or invalid values in numeric columns.

    Returns:
        data: 
    """
    # Replace empty strings or invalid values with NaN
    data['MIC'] = pd.to_numeric(data['MIC'], errors='coerce')
    data['sequence_length'] = pd.to_numeric(data['sequence_length'], errors='coerce')

    # Fill missing values with the mean (or any strategy you choose)
    data['MIC'].fillna(data['MIC'].mean(), inplace=True)
    data['sequence_length'].fillna(data['sequence_length'].mean(), inplace=True)

    return data

def split_data(data):
    """
    Split data into variables X and y for training, validation, and testing.

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """

    data = clean_data(data)

    X = data[['MIC', 'sequence_length']].astype(np.float32).to_numpy()                           # Input features (MIC and sequence_length)
    y = data['sequence'].apply(lambda x: np.array(x))                                            # Target labels (encoded / padded sequences)
    y = tf.keras.preprocessing.sequence.pad_sequences(y, padding='post', value=0, dtype='int32') # Pad sequences to the same length

    # Add start and end tokens to sequences
    start_token = 1
    end_token = 2
    y = np.pad(y, [(0, 0), (1, 1)], constant_values=(start_token, end_token))  # Add tokens at start and end

    # Split data into 70-15-15 split for train, val, test
    randint = random.randint(0, 4294967295)
    X_train, X_temp, y_train, y_temp = train_test_split(     X,      y, train_size=0.7, random_state=randint)
    X_val,   X_test, y_val,   y_test = train_test_split(X_temp, y_temp, train_size=0.5, random_state=randint)

    return X_train, X_val, X_test, y_train, y_val, y_test

def create_decoder_input(targets):
    """
    Create decoder input by shifting the target sequences and adding a start token.

    Args:
        targets (np.ndarray): Padded target sequences with start and end tokens.

    Returns:
        np.ndarray: Decoder input sequences.
    """
    # Initialize decoder input with padding
    decoder_input = np.full_like(targets, 0)
    decoder_input[:, 1:] = targets[:, :-1]  # Shift targets to the right
    decoder_input[:, 0] = 1  # Ensure start token at the beginning

    return decoder_input

def print_data(X_train = None, X_val = None, X_test = None, y_train = None, y_val = None, y_test = None):
    """
    Print the X and y variables for training, validation, and testing.
    """
    if (X_train is None or X_val is None or X_test is None or y_train is None or y_val is None or y_test is None):
        data = load_data()
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(data)
    print("X_train:")
    print(X_train)
    print()
    print("X_val:")
    print(X_val)
    print()
    print("X_test:")
    print(X_test)
    print()
    print("y_train:")
    print(y_train)
    print()
    print("y_val:")
    print(y_val)
    print()
    print("y_test:")
    print(y_test)
    print()