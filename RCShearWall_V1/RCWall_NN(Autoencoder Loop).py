import pandas as pd
from keras.models import Model
from keras.optimizers import Adam, SGD, RMSprop
from keras.layers import LSTM, Dense, Input, Concatenate, Reshape, concatenate, Flatten, Bidirectional, Conv1D, GlobalMaxPooling1D, Softmax, Dropout, Activation, CuDNNLSTM, MultiHeadAttention, MaxPooling1D, LayerNormalization, Add, TimeDistributed, RepeatVector, Lambda, Attention, Multiply
import keras.callbacks
from keras import backend as K
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from RCWall_DataProcessing import *


# Define R2 metric
def r_square(y_true, y_pred):
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - SS_res / (SS_tot + K.epsilon())


# Define the number of sample to be used
batch_size = 1500  # 3404
num_features = 1  # Number of columns in InputDisplacement curve (Just One Displacement Column with fixed Dt)
sequence_length = 300
parameters_length = 10
num_features_input_displacement = 1
num_features_input_parameters = 10

returned_data, returned_scaler = read_data(batch_size, sequence_length, normalize_data=True, save_normalized_data=False, smoothed_data=False)
InParams, InDisp, OutCycShear = returned_data
param_scaler, disp_scaler, cyc_shear_scaler = returned_scaler

# ---------------------- Split Data -------------------------------
# Split data into training, validation, and testing sets (X: Inputs & Y: Outputs)
X_param_train, X_param_test, X_disp_train, X_disp_test, Y_shear_train, Y_shear_test = train_test_split(
    InParams, InDisp, OutCycShear, test_size=0.15, random_state=42)

# ---------------------- NN Model Building -------------------------
# Build the neural network model using functional API
parameters_input = Input(shape=(num_features_input_parameters,), name='parameters_input')
displacement_input = Input(shape=(None, num_features_input_displacement), name='displacement_input')

distributed_parameters = RepeatVector(sequence_length)(parameters_input)

concatenated_tensor = concatenate([displacement_input, distributed_parameters], axis=-1)
print('concatenated_tensor', concatenated_tensor.shape)

# LSTM layer for encoding
lstm_encoder = LSTM(200, return_sequences=True, activation='tanh')(concatenated_tensor)
encoded_sequence = LSTM(50, return_sequences=True, activation='tanh')(lstm_encoder)
print('encoded_sequence', encoded_sequence.shape)

# LSTM layer for decoding
lstm_decoder = LSTM(200, return_sequences=True, activation='tanh')(encoded_sequence)
decoded_sequence = LSTM(num_features_input_displacement, return_sequences=True, activation='tanh')(lstm_decoder)
print('decoded_sequence', decoded_sequence.shape)

# Dense layer with 100 units
dense1 = Dense(200)(decoded_sequence)

# Dense layer with 100 units
dense2 = Dense(100)(dense1)

# ---------------------- Output layer --------------------------------------------
output_shear = Flatten()(Dense(1, name='output_shear')(dense2))
# output_pushover = Flatten()(Dense(1, name='output_pushover')(dense3))
print('output_shear ', output_shear.shape)
# print('output_pushover ', output_pushover.shape)

# ---------------------- Define the checkpoint callback ----------------------------
early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss",  # Loss to monitor for stopping
    patience=20,  # stop training after 10 non-improved training
    mode="auto",
    baseline=None,
    restore_best_weights=True,
    verbose=2)

# Initialize an empty DataFrame to store metrics
all_metrics_df = pd.DataFrame(columns=['Data', 'Epoch', 'Optimizer', 'Learning Rate', 'Batch Size', 'Training Loss', 'Validation Loss', 'Test Loss', 'Training R2', 'Validation R2', 'Test R2'])

# Loop over hyperparameters
for lr in [0.00001, 0.0001, 0.001, 0.01]:
    for optimizer in [Adam, SGD, RMSprop]:
        for batch_size in [8, 16, 32, 64]:
            # Build the model
            model = Model(inputs=[parameters_input, displacement_input], outputs=output_shear)
            model.compile(optimizer=optimizer(learning_rate=lr), loss='mse', metrics=['mse', r_square])

            # Train the model
            history = model.fit(
                [X_param_train, X_disp_train],
                [Y_shear_train],
                epochs=250,  # Reduce the epochs for quick testing
                batch_size=batch_size,
                validation_split=0.15,
                callbacks=[early_stopping]
            )

            # Find the epoch at which the best performance occurred
            best_epoch = np.argmin(history.history['val_loss']) + 1

            loss = model.evaluate([X_param_test, X_disp_test], [Y_shear_test])
            print("Test loss:", loss)

            # Create a DataFrame with metrics
            metrics_df = pd.DataFrame({
                'Data': 'Best 04',
                'Epoch': [best_epoch],
                'Optimizer': [optimizer],
                'Learning Rate': [lr],
                'Batch Size': [batch_size],
                'Training Loss': [history.history['loss'][best_epoch - 1]],
                'Validation Loss': [history.history['val_loss'][best_epoch - 1]],
                'Test Loss': [loss[0]],  # Assuming loss[0] is the test loss
                'Training R2': [history.history['r_square'][best_epoch - 1]],
                'Validation R2': [history.history['val_r_square'][best_epoch - 1]],
                'Test R2': [loss[2]],  # Assuming loss[0] is the test loss
            })

            # Append metrics to the overall DataFrame
            all_metrics_df = pd.concat([all_metrics_df, metrics_df], ignore_index=True)

# Save all metrics to a CSV file
all_metrics_df.to_csv('AE_metrics(04-2).csv', index=False)
