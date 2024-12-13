from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import tensorflow as tf, numpy as np, pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import random, csv, os


TRAIN_SPLIT = 70
VALID_SPLIT = 15
TEST_SPLIT  = 15

NUM_SYMBOLS = 23

# These represent the bounds of the lengths of the peptides in the dataset
MIN_PEPTIDE_LENGTH = 1
MAX_PEPTIDE_LENGTH = 51

# These represent this^^^ plus the start and end tokens
MIN_SEQUENCE_LENGTH = MIN_PEPTIDE_LENGTH + 2
MAX_SEQUENCE_LENGTH = MAX_PEPTIDE_LENGTH + 2


# Dictionary for amino acid to integer mapping
amino_acid_dict = {
    #------------------------>  0, 1, 2 represent padding, start, and end values respectively
    'PAD': 0, 'START': 1, 'END': 2,
    'A':  3, 'C':  4, 'D':  5, 'E':  6, 'F':  7, 'G':  8, 'H':  9, 'I': 10, 'K': 11, 'L': 12, 
    'M': 13, 'N': 14, 'P': 15, 'Q': 16, 'R': 17, 'S': 18, 'T': 19, 'V': 20, 'W': 21, 'Y': 22
}


def load_data():
    dataset_file = 'Dataset/Peptides.csv'
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_file_path = os.path.join(current_dir, dataset_file)
    if not os.path.exists(dataset_file_path):
        raise FileNotFoundError(f"Dataset file not found at {dataset_file_path}")

    data = []
    with open(dataset_file_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            sequence_length = int(row.get('sequence_length', ''))
            if MIN_PEPTIDE_LENGTH <= sequence_length and sequence_length <= MAX_PEPTIDE_LENGTH:
                data.append(row)
    
    return pd.DataFrame(data)  #->  Convert to DataFrame


def map_seq_to_int(sequence):
    return [amino_acid_dict.get(aa, 0) for aa in sequence]


def preprocess_data(data):
    # Normalize MIC and sequence_length
    scaler = MinMaxScaler()
    data[['MIC', 'sequence_length']] = scaler.fit_transform(data[['MIC', 'sequence_length']])

    # Normalize sequences using integer-encoding (0-22)
    # Tokenize each sequence (convert each amino acid into its corresponding integer)
    data['sequence'] = data['sequence'].apply(map_seq_to_int)
    
    padding_token, start_token, end_token = 0, 1, 2
    
    # Add the start and end tokens
    data['sequence'] = data['sequence'].apply(lambda seq: [start_token] + seq + [end_token])
    
    # Pad sequences to ensure they all have the same length
    padded_sequences = pad_sequences(
        data['sequence'], 
        maxlen=MAX_SEQUENCE_LENGTH, 
        padding='post', 
        truncating='post', 
        value=padding_token
    )
    
    # Ensure the sequences remain in 1D after padding
    data['sequence'] = [seq.tolist() for seq in padded_sequences]  # Convert each row to a list
    
    return data, scaler


def split_data(data):
    # Convert MIC and sequence_length to numpy array of type float32
    X = data[['MIC', 'sequence_length']].astype(np.float32).to_numpy()  # Input features (MIC and sequence_length)

    # Convert sequences to numpy array for y
    # Pad sequences to ensure they all have the same length
    y = np.array(data['sequence'].tolist())  # Target labels (encoded / padded sequences)

    # Split data into 70-15-15 split for train, val, test
    random_state = random.randint(0, 4294967295)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, train_size=(TRAIN_SPLIT / 100), random_state=random_state)
    
    # Ensure the total split is 100% by adjusting the validation and test split ratio
    val_test_split = VALID_SPLIT / (VALID_SPLIT + TEST_SPLIT)  # Calculate the ratio for validation vs test set
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, train_size=val_test_split, random_state=random_state)

    # Format the inputs correctly as required by the model
    train_inputs = [X_train[:, 0], X_train[:, 1]]  # MIC, sequence_length
    val_inputs   = [X_val  [:, 0], X_val  [:, 1]]
    test_inputs  = [X_test [:, 0], X_test [:, 1]]
    
    return train_inputs, val_inputs, test_inputs, y_train, y_val, y_test



def plot(history):
    plt.plot(history.history['loss'],     'r.-', label='Training loss')
    plt.plot(history.history['val_loss'], 'r',   label='Validation loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()
    
    plt.plot(history.history['accuracy'],     'g.-', label='Training accuracy')
    plt.plot(history.history['val_accuracy'], 'g',   label='Validation accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()


def plot_confusion_matrix(model, test_inputs, y_test):

    # Predict on the test set
    predictions = model.predict(test_inputs)
    
    # Convert predictions to class indices
    test_pred_labels = np.argmax(predictions, axis=-1)

    # Flatten the true and predicted labels for token-level comparison
    test_true_labels = y_test.ravel()
    test_pred_labels = test_pred_labels.ravel()

    # Ensure the label space includes all tokens (e.g., amino acids + padding)
    amino_acids = list(amino_acid_dict.keys())

    # Generate the confusion matrix
    cm = confusion_matrix(test_true_labels, test_pred_labels, labels=np.arange(len(amino_acids)))

    # Normalize the confusion matrix by row (true class)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = cm_normalized * 1000
    cm_normalized = cm_normalized.astype(np.int64)
    cm_normalized = np.nan_to_num(cm_normalized)  #->  Replace NaNs with zeros for rows with no instances

    # Display the normalized confusion matrix
    cmp = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=amino_acids)
    fig, ax = plt.subplots(figsize=(16,16))
    cmp.plot(ax=ax, cmap=plt.cm.copper, xticks_rotation='vertical')
    plt.title("Confusion Matrix for Amino Acids (Out of 1000)")
    plt.show()


def generate_peptide(model, mic_value, sequence_length_value, temperature=1.0):
    # Prepare the input data
    mic_input = np.array([[mic_value]], dtype=np.float32)  # Shape: (1, 1)
    len_input = np.array([[sequence_length_value]], dtype=np.float32)  # Shape: (1, 1)

    # Generate sequence predictions
    predicted_output = model.predict([mic_input, len_input], verbose=0)  # Shape: (1, sequence_length, vocab_size)

    # Convert the predicted output into a sequence of amino acids
    reverse_amino_acid_dict = {v: k for k, v in amino_acid_dict.items()}  # Reverse mapping: index -> amino acid

    # Get the most probable amino acid for each position with temperature sampling
    generated_sequence = []
    for t in range(sequence_length_value):
        logits = predicted_output[0, t]
        logits = logits / temperature  #------------------------------------->  Scale the logits by temperature
        probabilities = tf.nn.softmax(logits).numpy()  #---------------------->  Apply softmax to get probabilities
        token_index = np.random.choice(len(probabilities), p=probabilities)  #--->  Sample from the distribution
        
        if token_index == 2 and len(generated_sequence) >= 1:  #---->  If sequence ends, break loop
            break
        elif token_index > 2:  #------------------------------>  If amino acid, append to peptide sequence
            amino_acid = reverse_amino_acid_dict[token_index]  #--->  Convert token index to amino acid
            generated_sequence.append(amino_acid)  #------------->  Append amino acid to peptide sequence
    
    while len(generated_sequence) < sequence_length_value:  #-->  Pad sequence with '_' up to sequence_length
        generated_sequence.append('_')

    return ''.join(generated_sequence)


# Decode function for real sequences
def decode_sequence(encoded_sequence):
    reverse_amino_acid_dict = {v: k for k, v in amino_acid_dict.items()}
    return ''.join([reverse_amino_acid_dict[int(idx)] for idx in encoded_sequence if int(idx) > 2])