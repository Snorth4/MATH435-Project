import tensorflow as tf, numpy as np, util
from tensorflow.keras import Model, Sequential, optimizers
from tensorflow.keras.layers import Input, Dense, Dropout, Concatenate, MultiHeadAttention, GlobalAveragePooling1D, Embedding, LSTM, RepeatVector, Flatten
from tensorflow.keras.optimizers import Adam, Adagrad, RMSprop, SGD
from util import amino_acid_dict, MAX_PEPTIDE_LENGTH, MAX_SEQUENCE_LENGTH

                              # Helper table for our custom loss function
                              # Pad   Strt  End   AAAA  CCCC  DDDD  EEEE  FFFF  GGGG  HHHH  IIII  KKKK  LLLL  MMMM  NNNN  PPPP  QQQQ  RRRR  SSSS  TTTT  VVVV  WWWW  YYYY
aa_similarity_matrix = [[1.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00], # Pad
                        [0.00, 1.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00], # Strt
                        [0.00, 0.00, 1.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00], # End
                        [0.00, 0.00, 0.00, 1.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00], # AAAA
                        [0.00, 0.00, 0.00, 0.00, 1.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00], # CCCC
                        [0.00, 0.00, 0.00, 0.00, 0.00, 1.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00], # DDDD
                        [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00], # EEEE
                        [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00], # FFFF
                        [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00], # GGGG
                        [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00], # HHHH
                        [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00], # IIII
                        [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00], # KKKK
                        [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00], # LLLL
                        [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00], # MMMM
                        [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00], # NNNN
                        [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00], # PPPP
                        [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00], # QQQQ
                        [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00, 0.00, 0.00, 0.00, 0.00, 0.00], # RRRR
                        [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00, 0.00, 0.00, 0.00, 0.00], # SSSS
                        [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00, 0.00, 0.00, 0.00], # TTTT
                        [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00, 0.00, 0.00], # VVVV
                        [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00, 0.00], # WWWW
                        [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00]] # YYYY
aa_similarity_matrix = tf.convert_to_tensor(aa_similarity_matrix, dtype=tf.float32)


def sequence_mask(sequence):  #-------->  Create a mask for the padding, start, & end tokens
    mask_start = tf.not_equal(sequence, 1)
    mask_end   = tf.not_equal(sequence, 2)
    mask_pad   = tf.not_equal(sequence, 0)
    final_mask = tf.cast((mask_start & mask_end & mask_pad), dtype=tf.float32)

    return final_mask


def custom_loss_aa(y_true, y_pred):

    y_true_mask = sequence_mask(y_true)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)
    loss = loss * y_true_mask
    
    # Compute the mean loss, averaging over the non-padding tokens
    return tf.reduce_sum(loss) / tf.reduce_sum(y_true_mask)


def custom_loss_aa_len(y_true, y_pred):

    y_true_mask = sequence_mask(y_true)
    y_pred_indices = tf.argmax(y_pred, axis=-1)
    y_pred_mask = sequence_mask(y_pred_indices)

    true_sequence_lengths = tf.reduce_sum(y_true_mask, axis=-1)
    pred_sequence_lengths = tf.reduce_sum(y_pred_mask, axis=-1)

    length_differences = tf.abs(true_sequence_lengths - pred_sequence_lengths) / tf.cast(MAX_PEPTIDE_LENGTH, tf.float32)

    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)
    loss = loss * y_true_mask
    avg_loss = tf.reduce_sum(loss) / tf.reduce_sum(y_true_mask)

    total_loss = avg_loss + tf.reduce_mean(length_differences) * 1.0

    return total_loss


def custom_loss_aa_len_with_similarity(y_true, y_pred):
    # Create masks for sequences
    y_true_mask = sequence_mask(y_true)
    y_pred_indices = tf.argmax(y_pred, axis=-1)  # Predicted sequence indices
    y_pred_mask = sequence_mask(y_pred_indices)

    true_sequence_lengths = tf.reduce_sum(y_true_mask, axis=-1)
    pred_sequence_lengths = tf.reduce_sum(y_pred_mask, axis=-1)
    length_differences = tf.abs(true_sequence_lengths - pred_sequence_lengths) / tf.cast(MAX_PEPTIDE_LENGTH, tf.float32)

    # Convert true and predicted sequences to indices (assumed to be integer indices)
    y_true_indices = tf.cast(y_true, tf.int32)  # True sequence indices
    y_pred_indices = tf.cast(y_pred_indices, tf.int32)  # Predicted sequence indices
    
    # Gather similarity values for true amino acids (shape: [batch_size, seq_len])
    similarity_true = tf.gather(aa_similarity_matrix, y_true_indices, axis=-1, batch_dims=1)  # Shape: [batch_size, seq_len, num_symbols]
    
    # Gather predicted amino acid similarity values (shape: [batch_size, seq_len])
    similarity_pred = tf.gather(similarity_true, y_pred_indices, axis=-1, batch_dims=1)  # Shape: [batch_size, seq_len]

    # Compute the loss (negative log-likelihood)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)
    
    # Ensure shapes are compatible for element-wise multiplication
    loss = loss * (1.0 - similarity_pred)  # This adjusts the loss based on the similarity

    # Apply the mask to ignore padding tokens in the loss calculation
    loss = loss * y_true_mask

    # Average loss across non-padded positions
    avg_loss = tf.reduce_sum(loss) / tf.reduce_sum(y_true_mask)

    # Total loss includes both cross-entropy loss and length difference penalty
    total_loss = avg_loss + tf.reduce_mean(length_differences) * 1.0  # You can adjust the weight of length difference

    return total_loss


def autoencoder(learning_rate = 0.001):
    num_symbols = 23  #------------->  Number of unique amino acids + special tokens
    embedding_dim = 23  #------------>  Dimensions for embedding space
    dropout_rate = 0.2  #------------->  Dropout rate

    # MIC
    mic_input = Input(shape=(1,), name='MIC')  #-------------->  MIC input
    mic_dense = Dense(64, activation='relu')(mic_input)  #------>  This layer represents the MIC input
    mic_dense = Dropout(dropout_rate)(mic_dense)  #------------->  Regularize the input

    # Sequence length
    len_input = Input(shape=(1,), name='sequence_length')  #-->  Sequence length input
    len_dense = Dense(64, activation='relu')(len_input)  #-------->  This layer represents the sequence length
    len_dense = Dropout(dropout_rate)(len_dense)  #--------------->  Regularize the input

    # Combine MIC and length inputs
    encoder_input = Concatenate()([mic_dense, len_dense])  #--------->  Combine the input layers
    encoder_input = Dense(128, activation='relu')(encoder_input)  #-->  Interconnect the inputs
    encoder_input = Dropout(dropout_rate)(encoder_input)  #---------->  Regularize the input
    encoder_input = Dense(32, activation='relu')(encoder_input)  #-->  Interconnect the inputs
    encoder_input = Dropout(dropout_rate)(encoder_input)  #---------->  Regularize the input

    # Transformer Encoder
    encoder_output = MultiHeadAttention(num_heads=2, key_dim=32)(encoder_input[:, None, :], encoder_input[:, None, :])
    encoder_output = GlobalAveragePooling1D()(encoder_output)  #-------->  Further narrow the neurons
    encoder_output = Dropout(dropout_rate)(encoder_output)  #------------>  Regularize the encoder

    # Decoder - Auto-regressively predict sequences
    decoder_input  = RepeatVector(MAX_SEQUENCE_LENGTH)(encoder_output)  #---->  Make the decoder
    decoder_lstm   = LSTM(128, return_sequences=True)(decoder_input)  #------->  Make the sequence
    decoder_lstm   = Dropout(dropout_rate)(decoder_lstm)  #-------------------->  Regularize the output
    decoder_output = Dense(num_symbols, activation='softmax')(decoder_lstm)  #-->  Final output layer

    # Define model
    model = Model(inputs=[mic_input, len_input], outputs=decoder_output)

    optimizer = Adam    (learning_rate)
    # optimizer = Adagrad (learning_rate, weight_decay=0.00001)
    # optimizer = RMSprop (learning_rate)
    # optimizer = SGD     (learning_rate, momentum=0.0)

    # loss = 'sparse_categorical_crossentropy'
    # loss = custom_loss_aa
    loss = custom_loss_aa_len
    # loss = custom_loss_aa_len_with_similarity

    # Compile model
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    return model

def transformer_autoencoder(learning_rate=0.001):
    dropout_rate = 0.5
    input_dim = 64
    num_heads = 8
    ff_dim = 256
    embedding_dim = 32
    num_symbols = 23

    # MIC
    mic_input = Input(shape=(1,), name="MIC")
    len_dense = Dense(input_dim, activation='relu')(mic_input)  #------>  This layer represents the MIC input
    len_dense = Dropout(dropout_rate)(len_dense)  #------------->  Regularize the input

    # Sequence length
    len_input = Input(shape=(1,), name="sequence_length")
    len_dense = Dense(input_dim, activation='relu')(len_input)  #-------->  This layer represents the sequence length
    len_dense = Dropout(dropout_rate)(len_dense)  #--------------->  Regularize the input

    # Combine MIC and length inputs
    combined_features = Concatenate()([len_dense, len_dense])
    combined_features = Dense(input_dim*2, activation="relu")(combined_features)
    combined_features = Dropout(dropout_rate)(combined_features)

    # Transformer Encoder Block
    encoder_input = tf.expand_dims(combined_features, axis=1)  #-->  Expand dims to simulate sequence input
    attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=embedding_dim)(encoder_input, encoder_input)
    attention_output = Dropout(dropout_rate)(attention_output)

    # Feedforward layer after attention
    ff_output = Dense(ff_dim, activation="relu")(attention_output)
    ff_output = Dropout(dropout_rate)(ff_output)
    ff_output = Dense(embedding_dim, activation="relu")(ff_output)
    ff_output = Dropout(dropout_rate)(ff_output)
    encoder_output = Dropout(dropout_rate)(ff_output)

    # Flatten the encoder output
    encoder_representation = Flatten()(encoder_output)

    # Decoder for sequence generation
    decoder_input = RepeatVector(MAX_SEQUENCE_LENGTH)(encoder_representation)
    decoder_lstm  = LSTM(input_dim*4, return_sequences=True)(decoder_input)
    decoder_lstm   = Dropout(dropout_rate)(decoder_lstm)
    decoder_output = Dense(num_symbols, activation="softmax")(decoder_lstm)

    # Define model
    model = Model(inputs=[mic_input, len_input], outputs=decoder_output)

    optimizer = Adam    (learning_rate)
    # optimizer = Adagrad (learning_rate, weight_decay=0.00001)
    # optimizer = RMSprop (learning_rate)
    # optimizer = SGD     (learning_rate, momentum=0.0)

    # loss = 'sparse_categorical_crossentropy'
    loss = custom_loss_aa
    # loss = custom_loss_aa_len
    # loss = custom_loss_aa_len_with_similarity

    # Compile model
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    return model