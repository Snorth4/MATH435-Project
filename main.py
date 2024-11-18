import matplotlib.pyplot as plt
import tensorflow as tf
import keras, models, util

def main():
    """This is the main for peptide research."""
    
    # Load and preprocess data
    data = util.load_data()
    data = util.map_seq_to_int(data)

    # Split data into train, validation, and test sets
    X_train, X_val, X_test, y_train, y_val, y_test = util.split_data(data)
    util.print_data(X_train, X_val, X_test, y_train, y_val, y_test)

    # Separate MIC and sequence_length
    MIC_train, length_train = X_train[:, 0], X_train[:, 1]
    MIC_val,   length_val   = X_val  [:, 0], X_val  [:, 1]
    MIC_test,  length_test  = X_test [:, 0], X_test [:, 1]

    # Generate decoder inputs
    decoder_train = util.create_decoder_input(y_train)
    decoder_val   = util.create_decoder_input(y_val)
    decoder_test  = util.create_decoder_input(y_test)

    # Combine inputs
    train_inputs = [MIC_train, length_train, decoder_train]             # Train the encoder on the MIC and sequence length
    val_inputs   = [MIC_val,   length_val,   decoder_val]               # Train the decoder on the MIC and sequence length
    test_inputs  = [MIC_test,  length_test,  decoder_test]              # Grade the decoder on the sequence itself

    # Initialize the model
    model = models.autoencoder1(learning_rate = 0.002)

    # Early stopping callback
    early_stopping = keras.callbacks.EarlyStopping(
        monitor = 'val_loss',       # Metric to monitor (validation loss)
        patience = 50,              # Number of epochs with no improvement before stopping early
        restore_best_weights = True # Restore the best weights once training stops
    )

    # Train the model with early stopping
    history = model.fit(
        train_inputs,       # Training inputs
        y_train,            # Target labels
        batch_size = 32,
        epochs = 1000,
        callbacks = [early_stopping],
        validation_data=(val_inputs, y_val),
        shuffle = True
    )

    # Evaluate the model
    test_loss = model.evaluate(test_inputs, y_test)
    print(f'Test Loss: {test_loss}')

    # Plot the training and validation loss
    plt.plot(history.history['loss'],     'r.-', label='Training loss')
    plt.plot(history.history['val_loss'], 'r', label='Validation loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()