import tensorflow as tf
from keras import layers, optimizers, Model

def autoencoder1(learning_rate = 0.001):
    """A transformer-based autoencoder model for peptide sequence prediction with dropout regularization."""

    vocab_size = 23    # Number of unique amino acids + special tokens
    embedding_dim = 64
    dropout_rate = 0.5 # Dropout rate

    # Define inputs
    mic_input = layers.Input(shape=(1,), name='MIC')
    len_input = layers.Input(shape=(1,), name='sequence_length')
    decoder_input = layers.Input(shape=(None,), dtype=tf.int32, name='decoder_input')

    # Process MIC and sequence length
    mic_dense = layers.Dense(64, activation='relu')(mic_input)
    mic_dense = layers.Dropout(dropout_rate)(mic_dense)  # Apply dropout
    len_dense = layers.Dense(64, activation='relu')(len_input)
    len_dense = layers.Dropout(dropout_rate)(len_dense)  # Apply dropout

    # Combine MIC and length inputs
    encoder_input = layers.Concatenate()([mic_dense, len_dense])
    encoder_input = layers.Dense(128, activation='relu')(encoder_input)
    encoder_input = layers.Dropout(dropout_rate)(encoder_input)  # Apply dropout

    # Transformer Encoder
    encoder_output = layers.MultiHeadAttention(num_heads=2, key_dim=64)(
        encoder_input[:, None, :], encoder_input[:, None, :]
    )
    encoder_output = layers.GlobalAveragePooling1D()(encoder_output)

    # Apply dropout after Transformer Encoder
    encoder_output = layers.Dropout(dropout_rate)(encoder_output)

    # Decoder
    decoder_embedding = layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim)(decoder_input)
    decoder_lstm = layers.LSTM(256, return_sequences=True)(decoder_embedding)
    decoder_lstm = layers.Dropout(dropout_rate)(decoder_lstm)  # Apply dropout
    decoder_output = layers.Dense(vocab_size, activation='softmax')(decoder_lstm)

    optimizer = optimizers.Adam(learning_rate)

    # Model definition
    model = Model(inputs=[mic_input, len_input, decoder_input], outputs=decoder_output)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model