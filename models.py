import tensorflow as tf, numpy as np
from tensorflow.keras import Model, Sequential, optimizers
from tensorflow.keras.layers import Input, Dense, Dropout, Concatenate, MultiHeadAttention, GlobalAveragePooling1D, Embedding, LSTM, RepeatVector
from tensorflow.keras.optimizers import Adam, Nadam

def autoencoder1(learning_rate = 0.001):
    """A transformer-based autoencoder model for peptide sequence prediction with dropout regularization."""

    num_symbols = 23  #------------->  Number of unique amino acids + special tokens
    embedding_dim = 23  #------------>  Dimensions for embedding space
    dropout_rate = 0.5  #------------->  Dropout rate
    max_sequence_length = 171 + 2  #--->  Length of the largest sequence in the dataset plus the start and end symbols

    # Define inputs
    mic_input = Input(shape=(1,), name='MIC')  #-------------->  MIC input
    len_input = Input(shape=(1,), name='sequence_length')  #-->  Sequence length input

    # Process MIC and sequence length
    mic_dense = Dense(64, activation='relu')(mic_input)  #------>  This layer represents the MIC input          ##  
    mic_dense = Dropout(dropout_rate)(mic_dense)  #------------->  Regularize the input                         ##  These two layers represent the model's inputs
                                                                                                                ##  They are parallel to one another
    len_dense = Dense(64, activation='relu')(len_input)  #-------->  This layer represents the sequence length  ##  
    len_dense = Dropout(dropout_rate)(len_dense)  #--------------->  Regularize the input                       ##  

    # Combine MIC and length inputs
    encoder_input = Concatenate()([mic_dense, len_dense])  #--------->  Combine the input layers
    encoder_input = Dense(128, activation='relu')(encoder_input)  #-->  Interconnect the inputs
    encoder_input = Dense(64, activation='relu')(encoder_input)  #--->  Narrow the layer
    encoder_input = Dropout(dropout_rate)(encoder_input)  #---------->  Regularize the input

    # Transformer Encoder
    encoder_output = MultiHeadAttention(num_heads=2, key_dim=64)(encoder_input[:, None, :], encoder_input[:, None, :])  #---------->  Make the encoder
    encoder_output = GlobalAveragePooling1D()(encoder_output)  #-------->  Further narrow the neurons
    encoder_output = Dropout(dropout_rate)(encoder_output)  #------------>  Regularize the encoder

    # Decoder - Auto-regressively predict sequences
    decoder_input  = RepeatVector(max_sequence_length)(encoder_output)  #---->  Make the decoder
    decoder_lstm   = LSTM(256, return_sequences=True)(decoder_input)  #------->  Make the sequence
    decoder_lstm   = Dropout(dropout_rate)(decoder_lstm)  #-------------------->  Regularize the output
    decoder_output = Dense(num_symbols, activation='softmax')(decoder_lstm)  #-->  Final output layer

    # Compile model
    model = Model(inputs=[mic_input, len_input], outputs=decoder_output)
    model.compile(optimizer=Adam(learning_rate), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model