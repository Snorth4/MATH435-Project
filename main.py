import matplotlib.pyplot as plt
import tensorflow as tf, numpy as np
import keras, models, util

"""
TODO:

REMEMBER TO REMOVE:
util: ...
models: ...
main: ...
"""

def generate_peptide_sequence_with_temperature(model, mic_value, sequence_length_value, temperature=1.0):
    """
    Generates an anti-MRSA peptide sequence using the trained model with temperature sampling.
    """
    # Prepare the input data
    mic_input = np.array([[mic_value]], dtype=np.float32)  # Shape: (1, 1)
    len_input = np.array([[sequence_length_value]], dtype=np.float32)  # Shape: (1, 1)

    # Generate sequence predictions
    predicted_output = model.predict([mic_input, len_input], verbose=0)  # Shape: (1, sequence_length, vocab_size)

    # Convert the predicted output into a sequence of amino acids
    amino_acid_dict = util.amino_acid_dict  # Integer to amino acid mapping
    reverse_amino_acid_dict = {v: k for k, v in amino_acid_dict.items()}  # Reverse mapping: index -> amino acid

    # Get the most probable amino acid for each position with temperature sampling
    generated_sequence = []
    for t in range(sequence_length_value):
        logits = predicted_output[0, t]
        logits = logits / temperature  # Scale the logits by temperature
        probabilities = tf.nn.softmax(logits).numpy()  # Apply softmax to get probabilities
        token_index = np.random.choice(len(probabilities), p=probabilities)  # Sample from the distribution
        
        # Avoid adding the start and end tokens (0 and 2)
        if token_index == 2 and len(generated_sequence) >= 1:
            break  # Only stop if we have generated at least one amino acid
        elif token_index > 2:
            amino_acid = reverse_amino_acid_dict[token_index]  # Convert token index to amino acid
            generated_sequence.append(amino_acid)
    
    # Ensure the sequence has the requested length (padding if necessary)
    while len(generated_sequence) < sequence_length_value:
        generated_sequence.append('_')  # Append a placeholder (e.g., '_') for missing values

    return ''.join(generated_sequence)


def main():
    """This is the main for peptide research."""
    
    # Retrieve data; pre-processing
    data = util.load_data()  #------------------------------------------->  Load data
    data, scaler = util.normalize_data(data)  #------------------------>  Normalize data
    train_inputs, val_inputs, test_inputs, y_train, y_val, y_test = util.split_data(data)  #-->  Split up the data
    
    # Initialize the model
    model = models.autoencoder1(learning_rate = 0.001)
    early_stopping = keras.callbacks.EarlyStopping(
        monitor = 'val_loss',
        patience = 50,  #---------->  Number of epochs that make no improvements before stopping the training early
        restore_best_weights = True
    )

    # Train the model
    history = model.fit(
        train_inputs,  #------>  Training inputs - MIC, sequence length
        y_train,  #---------->  Training labels - sequence of amino acids
        batch_size = 16,  #--->  Size of the subset of data that is computed at once in a given epoch; small batch_sizes generalize better, larger batch_sizes are quicker
        epochs = 1000,  #-------------------->  Number of training loops
        callbacks = [early_stopping],  #-------->  Function to end training after overfitting is detected
        validation_data=(val_inputs, y_val),  #--->  Data that the model uses to determine loss and accuracy
        shuffle = True  #-------------------------->  Shuffle training data between each epoch
    )

    # Evaluate the model
    test_loss = model.evaluate(test_inputs, y_test)
    print(f'Test Loss: {test_loss}')

    # Plot the training and validation loss
    plt.plot(history.history['loss'],     'r.-', label='Training loss')
    plt.plot(history.history['val_loss'], 'r',   label='Validation loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()

    # Plot the training and validation accuracy
    plt.plot(history.history['accuracy'],     'g.-', label='Training accuracy')
    plt.plot(history.history['val_accuracy'], 'g',   label='Validation accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()

    MIC = [0.05, 12, 0.8, 4, 16, 19, 32, 32, 64, 2.5]
    sequence_length = [6, 51, 32, 12, 13, 8, 25, 27, 26, 9]
    peptide = []

    # Produce peptides with anti-MRSA properties
    for i in range(10):
        peptide.append(generate_peptide_sequence_with_temperature(model, MIC[i], sequence_length[i]))

    for i in range(10):
        print(f"MIC: {MIC[i]}, len: {sequence_length[i]}, sequence: {peptide[i]}")

    mic_input = np.array([[0.05]], dtype=np.float32)
    len_input = np.array([[6]], dtype=np.float32)

    predicted_probs = model.predict([mic_input, len_input])
    print(predicted_probs[0])  # Probability distribution for each token


if __name__ == '__main__':
    main()