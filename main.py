import tensorflow as tf, numpy as np, keras, models, util, random
from util import amino_acid_dict, generate_peptide, decode_sequence
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt


def main():
    # Retrieve, pre-process, & split the data
    data = util.load_data()
    normalized_data, scaler = util.preprocess_data(data)
    train_inputs, val_inputs, test_inputs, y_train, y_val, y_test = util.split_data(normalized_data)

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.001,
        decay_steps=10000,
        decay_rate=0.96
    )

    # Initialize the model
    model = models.autoencoder(learning_rate = 0.00066)
    # model = models.transformer_autoencoder(learning_rate = lr_schedule)

    early_stopping = keras.callbacks.EarlyStopping(
        monitor = 'val_loss',  #-->  Determines what early stops the training
        patience = 1000,  #------->  Number of epochs that make no improvements before stopping the training early
        restore_best_weights = True
    )

    # Train the model
    history = model.fit(
        train_inputs,  #------>  Training inputs - MIC, sequence length
        y_train,  #---------->  Training labels - tokenized sequence of amino acids
        batch_size = 32,  #--->  Size of the subset of data that is computed at once in a given epoch; small batch_sizes generalize better, larger batch_sizes run faster
        epochs = 100000,  #-------------------->  Number of training loops
        callbacks = [early_stopping],  #-------->  Function to end training after overfitting is detected
        validation_data=(val_inputs, y_val),  #--->  Data that the model uses to determine loss and accuracy; integral for early stopping and the patience hyperparameter
        shuffle = True  #-------------------------->  Shuffle training data between each epoch
    )

    # Evaluate the model
    test_loss = model.evaluate(test_inputs, y_test)
    print(f'Test Loss: {test_loss}')

    util.plot(history)
    
    util.plot_confusion_matrix(model, test_inputs, y_test)

    random_samples = data.sample(20)

    peptides = []
    for _, row in random_samples.iterrows():
        mic_value = float(row['MIC'])
        encoded_sequence = row['sequence']
        real_sequence = decode_sequence(encoded_sequence)
        sequence_length = len(real_sequence)

        # Generate predicted sequence
        predicted_sequence = generate_peptide(
            model, 
            mic_value = mic_value, 
            sequence_length_value = sequence_length,
            temperature = 1.0
        )

        peptides.append({
            'MIC': mic_value,
            'Real Sequence': real_sequence,
            'Predicted Sequence': predicted_sequence
        })

    # Output real and predicted sequences
    for peptide in peptides:
        print(f"    Normalized MIC: {peptide['MIC']}")
        print(f"     Real Sequence: {peptide['Real Sequence']}")
        print(f"Predicted Sequence: {peptide['Predicted Sequence']}\n")


if __name__ == '__main__':
    main()