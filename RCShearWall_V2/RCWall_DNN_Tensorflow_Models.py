from keras.metrics import MeanAbsoluteError, MeanSquaredError, RootMeanSquaredError
import keras.callbacks
import matplotlib.pyplot as plt
from keras.layers import Input, RepeatVector, concatenate, Bidirectional, LSTM, Dense, Dropout, Flatten, LayerNormalization, Add, MultiHeadAttention, Conv1D
from keras.models import Model
import tensorflow.keras.layers as layers
from keras.optimizers import Adam, SGD
from keras import backend as K
from keras.regularizers import l2
import tensorflow as tf
from RCWall_Data_Processing import *

# Allocate space for Bidirectional(LSTM)
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

# Activate the GPU
tf.config.list_physical_devices(device_type=None)
device = tf.config.list_physical_devices('GPU')
print("Num GPUs:", len(device))


# Define R2 metric
def r_square(y_true, y_pred):
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - SS_res / (SS_tot + K.epsilon())


class TimeSeriesTransformer(Model):
    def __init__(self, parameters_features, displacement_features, sequence_length, d_model=200):
        """
        Initialize the TimeSeriesTransformer as a Keras Model.

        Args:
            parameters_features (int): Number of input parameter features
            displacement_features (int): Number of displacement features
            sequence_length (int): Length of the input time series sequence
            d_model (int, optional): Dimensionality of the model. Defaults to 200.
        """
        super(TimeSeriesTransformer, self).__init__()
        self.parameters_features = parameters_features
        self.displacement_features = displacement_features
        self.sequence_length = sequence_length
        self.d_model = d_model

        # Parameter Processing Branch
        self.param_encoder = tf.keras.Sequential([
            Dense(d_model // 2),
            LayerNormalization(),
            tf.keras.layers.Activation('gelu'),
            Dropout(0.1)
        ])

        # Time Series Processing Branch
        self.series_encoder = tf.keras.Sequential([
            Dense(d_model // 2),
            LayerNormalization(),
            tf.keras.layers.Activation('gelu'),
            Dropout(0.1)
        ])

        # Positional Encoding
        self.positional_encoding = self._create_positional_encoding(d_model, sequence_length)

        # Processing Blocks
        self.processing_blocks = [
            self._create_processing_block(d_model) for _ in range(3)
        ]

        # Output Generation
        self.output_layer = tf.keras.Sequential([
            Dense(d_model * 2, activation='gelu'),
            Dropout(0.1),
            Dense(d_model * 4, activation='gelu'),
            Dropout(0.1),
            Dense(d_model * 2, activation='gelu'),
            Dropout(0.1),
            Dense(displacement_features)
        ])

        # Temporal Smoothing
        self.temporal_smoother = Conv1D(
            filters=displacement_features,
            kernel_size=5,
            padding='same',
            groups=displacement_features
        )

    def _create_positional_encoding(self, d_model, max_len):
        """Create positional encoding matrix."""
        position = np.arange(max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        pe = np.zeros((max_len, d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        return tf.constant(pe, dtype=tf.float32)

    def _create_processing_block(self, d_model):
        """Create a processing block with multi-head attention and LSTM."""
        return {
            'attention': MultiHeadAttention(num_heads=4, key_dim=d_model // 4),
            'norm1': LayerNormalization(),
            'lstm1': LSTM(d_model, return_sequences=True),
            'norm2': LayerNormalization(),
            'lstm2': LSTM(d_model, return_sequences=True),
            'norm3': LayerNormalization()
        }

    def call(self, inputs, training=False):
        """
        Forward pass of the model.

        Args:
            inputs (tuple): Tuple of (parameters, time_series)
            training (bool): Whether the model is in training mode

        Returns:
            tf.Tensor: Transformed time series
        """
        parameters, time_series = inputs

        # Expand parameters to match sequence length
        params_expanded = tf.repeat(
            tf.expand_dims(parameters, 1),
            repeats=self.sequence_length,
            axis=1
        )
        params_encoded = self.param_encoder(params_expanded, training=training)

        # Process time series
        series_encoded = self.series_encoder(
            tf.expand_dims(time_series, axis=-1),
            training=training
        )

        # Combine features
        combined = tf.concat([params_encoded, series_encoded], axis=-1)

        # Add positional encoding
        combined += self.positional_encoding[:combined.shape[1], :combined.shape[-1]]

        # Process through main blocks
        x = combined
        for block in self.processing_blocks:
            # Multi-head attention
            attn_output = block['attention'](x, x)
            x = block['norm1'](x + attn_output)

            # First LSTM
            lstm_output = block['lstm1'](x)
            x = block['norm2'](x + lstm_output)

            # Second LSTM
            lstm_output = block['lstm2'](x)
            x = block['norm3'](x + lstm_output)

        # Generate output sequence
        output = self.output_layer(x, training=training)

        # Apply temporal smoothing
        smoothed_output = tf.transpose(
            self.temporal_smoother(tf.transpose(output, perm=[0, 2, 1])),
            perm=[0, 2, 1]
        )

        return tf.squeeze(smoothed_output, axis=-1)


class TransformerWithPositionalEncoding:
    def __init__(self, PARAMETERS_FEATURES, DISPLACEMENT_FEATURES, sequence_length, d_model, num_heads, num_encoder_layers, num_decoder_layers, dff, dropout_rate=0.1):
        self.PARAMETERS_FEATURES = PARAMETERS_FEATURES
        self.DISPLACEMENT_FEATURES = DISPLACEMENT_FEATURES
        self.sequence_length = sequence_length
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dff = dff
        self.dropout_rate = dropout_rate
        self.model = self.build_model()

    # Positional encoding function
    def get_positional_encoding(self, sequence_length, embedding_dim):
        positions = np.arange(sequence_length)[:, np.newaxis]
        dimensions = np.arange(embedding_dim)[np.newaxis, :]
        angle_rates = 1 / np.power(10000, (2 * (dimensions // 2)) / np.float32(embedding_dim))
        angle_rads = positions * angle_rates

        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

        pos_encoding = angle_rads[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)

    def build_model(self):
        parameters_input = Input(shape=(self.PARAMETERS_FEATURES,), name='parameters_input')
        displacement_input = Input(shape=(self.sequence_length, self.DISPLACEMENT_FEATURES), name='displacement_input')

        # Positional encoding
        pos_encoding = self.get_positional_encoding(self.sequence_length, self.d_model)

        # Add positional encoding to the input
        displacement_input_with_pos = displacement_input + pos_encoding

        # Encoder
        encoder_input = Dense(self.d_model)(displacement_input_with_pos)
        for _ in range(self.num_encoder_layers):
            encoder_input = self.encoder_layer(encoder_input)

        # Decoder
        distributed_parameters = Dense(self.d_model)(tf.expand_dims(parameters_input, axis=1))
        distributed_parameters = tf.tile(distributed_parameters, [1, self.sequence_length, 1])
        decoder_input = Dense(self.d_model)(distributed_parameters)
        for _ in range(self.num_decoder_layers):
            decoder_input = self.decoder_layer(decoder_input, encoder_input)

        # Final Dense layers
        dense1 = Dense(200, activation='tanh')(decoder_input)
        dropout1 = Dropout(self.dropout_rate)(dense1)
        dense2 = Dense(100, activation='tanh')(dropout1)
        dropout2 = Dropout(self.dropout_rate)(dense2)
        output_shear = Dense(1, activation='linear', name='output_shear')(dropout2)

        model = Model(inputs=[parameters_input, displacement_input], outputs=output_shear)
        return model

    def encoder_layer(self, x):
        attn_output = MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.d_model // self.num_heads
        )(query=x, value=x, key=x)
        attn_output = Dropout(self.dropout_rate)(attn_output)
        out1 = LayerNormalization(epsilon=1e-6)(Add()([x, attn_output]))

        ffn_output = self.ffn(out1)
        ffn_output = Dropout(self.dropout_rate)(ffn_output)
        out2 = LayerNormalization(epsilon=1e-6)(Add()([out1, ffn_output]))

        return out2

    def decoder_layer(self, x, enc_output):
        attn1 = MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.d_model // self.num_heads
        )(query=x, value=x, key=x)
        attn1 = Dropout(self.dropout_rate)(attn1)
        out1 = LayerNormalization(epsilon=1e-6)(Add()([x, attn1]))

        attn2 = MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.d_model // self.num_heads
        )(query=out1, value=enc_output, key=enc_output)
        attn2 = Dropout(self.dropout_rate)(attn2)
        out2 = LayerNormalization(epsilon=1e-6)(Add()([out1, attn2]))

        ffn_output = self.ffn(out2)
        ffn_output = Dropout(self.dropout_rate)(ffn_output)
        out3 = LayerNormalization(epsilon=1e-6)(Add()([out2, ffn_output]))

        return out3

    def ffn(self, x):
        x = Dense(self.dff, activation='relu')(x)
        return Dense(self.d_model)(x)


class Bi_LSTM:
    def __init__(self, PARAMETERS_FEATURES, DISPLACEMENT_FEATURES, sequence_length):
        self.PARAMETERS_FEATURES = PARAMETERS_FEATURES
        self.DISPLACEMENT_FEATURES = DISPLACEMENT_FEATURES
        self.sequence_length = sequence_length
        self.model = self.build_model()

    def build_model(self):
        parameters_input = Input(shape=(self.PARAMETERS_FEATURES,), name='parameters_input')

        displacement_input = Input(shape=(self.sequence_length, self.DISPLACEMENT_FEATURES), name='displacement_input')

        distributed_parameters = RepeatVector(self.sequence_length)(parameters_input)

        concatenated_tensor = concatenate([displacement_input, distributed_parameters], axis=-1)
        print("concatenated_tensor = ", concatenated_tensor.shape)

        # Bidirectional LSTM layer with return_sequences=True
        lstm1 = Bidirectional(LSTM(200, return_sequences=True, activation='tanh', stateful=False)(concatenated_tensor))

        # Bidirectional LSTM layer with return_sequences=True
        lstm2 = Bidirectional(LSTM(200, return_sequences=True, activation='tanh', stateful=False)(lstm1))

        # Dense layer with 200 units
        dense1 = Dense(200, activation='tanh')(lstm2)
        dropout1 = Dropout(0.2)(dense1)

        # Dense layer with 100 units
        dense2 = Dense(100, activation='tanh')(dropout1)
        dropout2 = Dropout(0.2)(dense2)

        # ---------------------- Output layer --------------------------------------------
        output_shear = Flatten()(Dense(1, activation='linear', name='output_shear')(dropout2))

        # ---------------------- Build the model ------------------------------------------
        model = Model(inputs=[parameters_input, displacement_input], outputs=output_shear)

        return model


class LSTM_AE:
    def __init__(self, PARAMETERS_FEATURES, DISPLACEMENT_FEATURES, sequence_length):
        self.PARAMETERS_FEATURES = PARAMETERS_FEATURES
        self.DISPLACEMENT_FEATURES = DISPLACEMENT_FEATURES
        self.sequence_length = sequence_length
        self.model = self.build_model()

    def build_model(self):
        parameters_input = Input(shape=(self.PARAMETERS_FEATURES,), name='parameters_input')

        displacement_input = Input(shape=(self.sequence_length, self.DISPLACEMENT_FEATURES), name='displacement_input')

        distributed_parameters = RepeatVector(self.sequence_length)(parameters_input)

        concatenated_tensor = concatenate([displacement_input, distributed_parameters], axis=-1)
        print("concatenated_tensor = ", concatenated_tensor.shape)

        # LSTM layer for encoding
        lstm_encoder = LSTM(200, return_sequences=True, activation='tanh')(concatenated_tensor)
        encoded_sequence = LSTM(50, return_sequences=True, activation='tanh')(lstm_encoder)

        # LSTM layer for decoding
        lstm_decoder = LSTM(200, return_sequences=True, activation='tanh')(encoded_sequence)
        decoded_sequence = LSTM(self.DISPLACEMENT_FEATURES, return_sequences=True, activation='tanh')(lstm_decoder)

        # Dense layer with 200 units
        dense1 = Dense(200, activation='tanh')(decoded_sequence)
        dropout1 = Dropout(0.2)(dense1)

        # Dense layer with 100 units
        dense2 = Dense(100, activation='tanh')(dropout1)
        dropout2 = Dropout(0.2)(dense2)

        # ---------------------- Output layer --------------------------------------------
        output_shear = Flatten()(Dense(1, activation='linear', name='output_shear')(dropout2))

        # ---------------------- Build the model ------------------------------------------
        model = Model(inputs=[parameters_input, displacement_input], outputs=output_shear)

        return model


# Define hyperparameters
DATA_SIZE = 30000
SEQUENCE_LENGTH = 500
DISPLACEMENT_FEATURES = 1
PARAMETERS_FEATURES = 17
ANALYSIS = 'CYCLIC'
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 100
PATIENCE = 15
TEST_SIZE = 0.2
VAL_SIZE = 0.2
D_MODEL = 128
NUM_HEADS = 8
NUM_ENCODER_LAYERS = 2
NUM_DECODER_LAYERS = 2
DFF = 256
DROPOUT_RATE = 0.1

# Define hyperparameters
DATA_FOLDER = "RCWall_Data/Run_Full/FullData"
DATA_SIZE = 200000
SEQUENCE_LENGTH = 500
DISPLACEMENT_FEATURES = 1
PARAMETERS_FEATURES = 17
TEST_SIZE = 0.10
VAL_SIZE = 0.15
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
EPOCHS = 20
PATIENCE = 5

# Load and preprocess data
(InParams, InDisp, OutShear), (param_scaler, disp_scaler, shear_scaler) = load_data(DATA_SIZE,
                                                                                    SEQUENCE_LENGTH,
                                                                                    PARAMETERS_FEATURES,
                                                                                    DATA_FOLDER,
                                                                                    True,
                                                                                    True)

# ---------------------- Split Data -------------------------------
# Split data into training, validation, and testing sets (X: Inputs & Y: Outputs)
X_param_train, X_param_test, X_disp_train, X_disp_test, Y_shear_train, Y_shear_test = train_test_split(
    InParams, InDisp, OutShear, test_size=TEST_SIZE, random_state=42)

# ---------------------- Build the model ------------------------------------------
# model = LSTM_AE(PARAMETERS_FEATURES, DISPLACEMENT_FEATURES, SEQUENCE_LENGTH).model
# model = Bi_LSTM(PARAMETERS_FEATURES, DISPLACEMENT_FEATURES, SEQUENCE_LENGTH).model
# model = TransformerWithPositionalEncoding(
#     PARAMETERS_FEATURES, DISPLACEMENT_FEATURES, SEQUENCE_LENGTH,
#     D_MODEL, NUM_HEADS, NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, DFF, DROPOUT_RATE).model

model = TimeSeriesTransformer(PARAMETERS_FEATURES,
                              DISPLACEMENT_FEATURES,
                              SEQUENCE_LENGTH)

# --------------------- Compile the model -----------------------------------------
# Define Adam and SGD optimizers
adam_optimizer = Adam(LEARNING_RATE)
sgd_optimizer = SGD(LEARNING_RATE, momentum=0.9)
model.compile(optimizer=adam_optimizer, loss='mse', metrics=[RootMeanSquaredError()])  # metrics=[MeanAbsoluteError(), MeanSquaredError(), RootMeanSquaredError(), r_square])

# ---------------------- Print Model summary ---------------------------------------------
model.summary()

# ---------------------- Define the checkpoint callback ----------------------------
early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss",  # Loss to monitor for stopping
    patience=PATIENCE,  # stop training after 10 non-improved training
    mode="auto",
    baseline=None,
    restore_best_weights=True,
    verbose=2)

# ---------------------- Train the model ---------------------------------------------
history = model.fit(
    [X_param_train, X_disp_train],  # Input layer (GMA + STRUCTURAL PARAMETERS)
    [Y_shear_train],  # Output layer (SHEAR)
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=VAL_SIZE,
    callbacks=[early_stopping])  # checkpoint_callback or early_stopping

# ---------------------- Save the model ---------------------------------------------
# model.save("DNN_Models/DNN_LSTM-AE(CYCLIC)test")  # Save the model after training
# model.save("DNN_Models/DNN_LSTM-AE(PUSHOVER)")  # Save the model after training

# ---------------------- Plot Accuracy and Loss ----------------------------------------
# Find the epoch at which the best performance occurred
best_epoch = np.argmin(history.history['val_loss']) + 1  # +1 because epochs are 1-indexed

# Plot the training and validation loss
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label="Training Loss")
plt.plot(history.history['val_loss'], label="Validation Loss")
plt.scatter(best_epoch - 1, history.history['val_loss'][best_epoch - 1], color='red')  # -1 because Python is 0-indexed
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training and Validation Loss Over Epochs")
plt.legend()
plt.show()

# Plot the training and validation loss
# plt.figure(figsize=(10, 6))
# plt.plot(history.history['r_square'], label="Training Accuracy")
# plt.plot(history.history['val_r_square'], label="Validation Accuracy")
# plt.scatter(best_epoch - 1, history.history['val_r_square'][best_epoch - 1], color='red')  # -1 because Python is 0-indexed
# plt.xlabel("Epochs")
# plt.ylabel("Accuracy R2")
# plt.title("Training and Validation Accuracy Over Epochs")
# plt.legend()
# plt.show()

# ---------------------- Model testing ---------------------------------------------------
loss = model.evaluate([X_param_test, X_disp_test], [Y_shear_test])
print("Test loss:", loss)

# ---------------------- Plotting the results ---------------------------------------------
test_index = 3

new_input_parameters = X_param_test[0:test_index]  # Select corresponding influencing parameters
new_input_displacement = X_disp_test[0:test_index]  # Select a single example
real_shear = Y_shear_test[0:test_index]
# real_pushover = Y_pushover_test[0:test_index]

# Predict displacement for the new data
# predicted_shear = model.predict([new_input_parameters, new_input_displacement])
predicted_shear = model.predict([new_input_parameters, new_input_displacement])

# Plot the predicted displacement
plt.figure(figsize=(10, 6))
for i in range(test_index):
    plt.plot(predicted_shear[i], label=f'Predicted Shear - {i + 1}')
    plt.plot(real_shear[i], label=f'Real Shear - {i + 1}')
    plt.xlabel('Time Step', {'fontname': 'Cambria', 'fontstyle': 'italic', 'size': 14})
    plt.ylabel('Shear Load', {'fontname': 'Cambria', 'fontstyle': 'italic', 'size': 14})
    plt.title('Predicted Shear Time Series', {'fontname': 'Cambria', 'fontstyle': 'normal', 'size': 16})
    plt.yticks(fontname='Cambria', fontsize=14)
    plt.xticks(fontname='Cambria', fontsize=14)
    plt.tight_layout()
    plt.legend()
    plt.grid()
    plt.show()

# Plot the predicted displacement
# plt.figure(figsize=(10, 6))
for i in range(test_index):
    plt.plot(new_input_displacement[i], predicted_shear[i], label=f'Predicted Loop - {i + 1}')
    plt.plot(new_input_displacement[i], real_shear[i], label=f'Real Loop - {i + 1}')
    plt.xlabel('Displacement', fontdict={'fontname': 'Cambria', 'fontstyle': 'italic', 'size': 14})
    plt.ylabel('Shear Load', fontdict={'fontname': 'Cambria', 'fontstyle': 'italic', 'size': 14})
    plt.title('Predicted Hysteresis', fontdict={'fontname': 'Cambria', 'fontstyle': 'normal', 'size': 16})
    plt.yticks(fontname='Cambria', fontsize=14)
    plt.xticks(fontname='Cambria', fontsize=14)
    plt.tight_layout()
    plt.legend()
    plt.grid()
    plt.show()

new_input_parameters = denormalize(new_input_parameters, param_scaler, sequence=False)
new_input_displacement = denormalize(new_input_displacement, disp_scaler, sequence=True)
real_shear = denormalize(real_shear, shear_scaler, sequence=True)

predicted_shear = denormalize(predicted_shear, shear_scaler, sequence=True)

# Plot the predicted displacement
plt.figure(figsize=(10, 6))
for i in range(test_index):
    plt.plot(predicted_shear[i], label=f'Predicted Shear load - {i + 1}')
    plt.plot(real_shear[i], label=f'Real Shear load - {i + 1}')
    plt.xlabel('Time Step', {'fontname': 'Cambria', 'fontstyle': 'italic', 'size': 14})
    plt.ylabel('Shear Load', {'fontname': 'Cambria', 'fontstyle': 'italic', 'size': 14})
    plt.title('Predicted Shear Time Series', {'fontname': 'Cambria', 'fontstyle': 'normal', 'size': 16})
    plt.yticks(fontname='Cambria', fontsize=14)
    plt.xticks(fontname='Cambria', fontsize=14)
    plt.tight_layout()
    plt.legend()
    plt.grid()
    plt.show()

# Plot the predicted displacement
# plt.figure(figsize=(10, 6))
for i in range(test_index):
    plt.plot(new_input_displacement[i], predicted_shear[i], label=f'Predicted Loop - {i + 1}')
    plt.plot(new_input_displacement[i], real_shear[i], label=f'Real Loop - {i + 1}')
    plt.xlabel('Displacement', fontdict={'fontname': 'Cambria', 'fontstyle': 'italic', 'size': 14})
    plt.ylabel('Shear Load', fontdict={'fontname': 'Cambria', 'fontstyle': 'italic', 'size': 14})
    plt.title('Predicted Hysteresis', fontdict={'fontname': 'Cambria', 'fontstyle': 'normal', 'size': 16})
    plt.yticks(fontname='Cambria', fontsize=14)
    plt.xticks(fontname='Cambria', fontsize=14)
    plt.tight_layout()
    plt.legend()
    plt.grid()
    plt.show()
