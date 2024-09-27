from keras.models import Model
from keras.optimizers import Adam, SGD
from keras import backend as K
from keras.layers import LSTM, Dense, Input, Concatenate, Reshape, concatenate, Flatten, Bidirectional, Conv1D, GlobalMaxPooling1D, Softmax, Dropout, Activation, CuDNNLSTM, MultiHeadAttention, MaxPooling1D, LayerNormalization, Add, TimeDistributed, RepeatVector, Lambda, Attention, Multiply
import keras.callbacks
from keras_nlp.layers import TransformerEncoder, TransformerDecoder
# from tensorflow_transformers import TransformerEncoder
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from RCWall_DataProcessing import *

# Define R2 metric
def r_square(y_true, y_pred):
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - SS_res / (SS_tot + K.epsilon())


# Define the number of sample to be used
batch_size = 30000  # 3404
num_features = 1  # Number of columns in InputDisplacement curve (Just One Displacement Column with fixed Dt)
sequence_length = 500
parameters_length = 10
num_features_input_displacement = 1
num_features_input_parameters = 10

returned_data, returned_scaler = load_data(batch_size, sequence_length, normalize_data=True, save_normalized_data=False, smoothed_data=False)
InParams, InDisp, OutCycShear = returned_data
param_scaler, disp_scaler, cyc_shear_scaler = returned_scaler

# ---------------------- Split Data -------------------------------
# Split data into training, validation, and testing sets (X: Inputs & Y: Outputs)
X_param_train, X_param_test, X_disp_train, X_disp_test, Y_shear_train, Y_shear_test = train_test_split(
    InParams, InDisp, OutCycShear, test_size=0.15, random_state=42)

# ---------------------- Improved NN Model Building -------------------------
# Build the neural network model using functional API
parameters_input = Input(shape=(num_features_input_parameters,), name='parameters_input')
displacement_input = Input(shape=(None, num_features_input_displacement), name='displacement_input')

distributed_parameters = RepeatVector(sequence_length)(parameters_input)

concatenated_tensor = concatenate([displacement_input, distributed_parameters], axis=-1)
print('concatenated_tensor', concatenated_tensor.shape)

# Transformer encoder layer
transformer_encoder = TransformerEncoder(intermediate_dim=512, num_heads=4)(concatenated_tensor)
encoded_sequence = transformer_encoder(concatenated_tensor, training=True, mask=None)
print('encoded_sequence', encoded_sequence.shape)

# Transformer decoder layer
transformer_decoder = TransformerDecoder(intermediate_dim=512, num_heads=4)(encoded_sequence)
decoded_sequence = transformer_decoder(encoded_sequence, training=True, look_ahead_mask=None, padding_mask=None)
print('decoded_sequence', decoded_sequence.shape)

# Dense layer with 100 units and dropout
dense1 = Dense(100)(decoded_sequence)
dropout1 = Dropout(0.2)(dense1)

# Dense layer with 100 units and dropout
dense2 = Dense(100)(dropout1)
dropout2 = Dropout(0.2)(dense2)

# ---------------------- Output layer --------------------------------------------
output_shear = Flatten()(Dense(1, name='output_shear')(dropout2))

# ---------------------- Build the model ------------------------------------------
# Build the model
model = Model(inputs=[parameters_input, displacement_input], outputs=output_shear)

# ---------------------- Compile the model -----------------------------------------
# Define Adam and SGD optimizers
adam_optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=adam_optimizer,
              loss='mse',
              metrics=['mse']) #, 'accuracy'

# ---------------------- Print Model summary ---------------------------------------------
model.summary()

# ---------------------- Define the checkpoint callback ----------------------------
early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss",  # Loss to monitor for stopping
    patience=50,  # stop training after 10 non-improved training
    mode="auto",
    baseline=None,
    restore_best_weights=True,
    verbose=2)

# ---------------------- Train the model ---------------------------------------------
history = model.fit(
    [X_param_train, X_disp_train],  # Input layer (GMA + STRUCTURAL PARAMETERS)
    [Y_shear_train],  # Output layer (SHEAR)
    epochs=600,
    batch_size=32,
    validation_split=0.15,
    callbacks=[early_stopping]  # checkpoint_callback or early_stopping
)
# ---------------------- Save the model ---------------------------------------------
# model.save("NN_functionalAPI_DynamicSystem")  # Save the model after training

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

# plt.figure(figsize=(10, 6))
# plt.plot(history.history['shear_output_accuracy'], label="Training Accuracy")
# plt.plot(history.history['val_shear_output_accuracy'], label="Validation Accuracy")
# plt.scatter(best_epoch - 1, history.history['val_shear_output_accuracy'][best_epoch - 1], color='red')  # -1 because Python is 0-indexed
# plt.xlabel("Epochs")
# plt.ylabel("Accuracy")
# plt.title("Training and Validation Accuracy Over Epochs")
# plt.legend()
# plt.show()
# # Plot the training and validation accuracy
# plt.plot(history.history['mse'], label="Training MSE")
# plt.plot(history.history['val_mse'], label="Validation MSE")
# plt.xlabel("Epochs")
# plt.ylabel("Mean Squared Error (MSE)")
# plt.title("Training and Validation MSE Over Epochs")
# plt.legend()
# plt.show()

# Evaluate the model
loss = model.evaluate([X_param_test, X_disp_test], [Y_shear_test])
print("Test loss:", loss)


# ---------------------- Plotting the results ---------------------------------------------
test_index = 3

new_input_parameters = X_param_test[0:test_index]  # Select corresponding influencing parameters
new_input_displacement = X_disp_test[0:test_index]  # Select a single example
real_shear = Y_shear_test[0:test_index]
real_displacement = Y_disp_test[0:test_index]  # Select a single example

# Predict displacement for the new data
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
plt.figure(figsize=(10, 6))
for i in range(test_index):
    plt.plot(new_input_displacement[i], label=f'Input Displacement - {i + 1}')
    plt.plot(real_displacement[i], label=f'Real Displacement - {i + 1}')
    plt.xlabel('Time Step', {'fontname': 'Cambria', 'fontstyle': 'italic', 'size': 14})
    plt.ylabel('Displacement', {'fontname': 'Cambria', 'fontstyle': 'italic', 'size': 14})
    plt.title('Predicted Displacement Time Series', {'fontname': 'Cambria', 'fontstyle': 'normal', 'size': 16})
    plt.yticks(fontname='Cambria', fontsize=14)
    plt.xticks(fontname='Cambria', fontsize=14)
    plt.tight_layout()
    plt.legend()
    plt.grid()
    plt.show()

# Plot the predicted displacement
# plt.figure(figsize=(10, 6))
for i in range(test_index):
    plt.plot(real_displacement[i], predicted_shear[i], label=f'Predicted Loop - {i + 1}')
    # plt.plot(new_input_displacement[i, :-1], predicted_shear[i, 1:], label=f'Input displacement - {i + 1}')
    plt.plot(real_displacement[i], real_shear[i], label=f'True Loop - {i + 1}')
    # plt.plot(new_input_displacement[i, :-1], real_shear[i, 1:], label=f'Input displacement - {i + 1}')
    plt.xlabel('Displacement', fontdict={'fontname': 'Cambria', 'fontstyle': 'italic', 'size': 14})
    plt.ylabel('Shear Load', fontdict={'fontname': 'Cambria', 'fontstyle': 'italic', 'size': 14})
    plt.title('Predicted Hysteresis', fontdict={'fontname': 'Cambria', 'fontstyle': 'normal', 'size': 16})
    plt.yticks(fontname='Cambria', fontsize=14)
    plt.xticks(fontname='Cambria', fontsize=14)
    plt.tight_layout()
    plt.legend()
    plt.grid()
    plt.show()



# Inverse transform the normalized data before plotting
param_scaler, disp_scaler, cyc_shear_scaler, cyc_disp_scaler, push_shear_scaler, push_disp_scaler = returned_scaler

new_input_parameters = param_scaler.inverse_transform(new_input_parameters)
new_input_displacement = restoration(new_input_displacement, InDisp_min, InDisp_max)
real_shear = restoration(real_shear, OutCycShear_min, OutCycShear_max)
real_displacement = restoration(real_displacement, OutCycDisp_min, OutCycDisp_max)
predicted_shear = restoration(predicted_shear, OutCycShear_min, OutCycShear_max)
# predicted_displacement = cyc_disp_scaler.inverse_transform(predicted_displacement)

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
plt.figure(figsize=(10, 6))
for i in range(test_index):
    plt.plot(new_input_displacement[i], label=f'Input Displacement - {i + 1}')
    plt.plot(real_displacement[i], label=f'Real Displacement - {i + 1}')
    plt.xlabel('Time Step', {'fontname': 'Cambria', 'fontstyle': 'italic', 'size': 14})
    plt.ylabel('Displacement', {'fontname': 'Cambria', 'fontstyle': 'italic', 'size': 14})
    plt.title('Predicted Displacement Time Series', {'fontname': 'Cambria', 'fontstyle': 'normal', 'size': 16})
    plt.yticks(fontname='Cambria', fontsize=14)
    plt.xticks(fontname='Cambria', fontsize=14)
    plt.tight_layout()
    plt.legend()
    plt.grid()
    plt.show()

# Plot the predicted displacement
# plt.figure(figsize=(10, 6))
for i in range(test_index):
    # plt.plot(new_input_displacement[i], predicted_shear[i], label=f'Predicted Loop - {i + 1}')
    plt.plot(new_input_displacement[i, :-1], predicted_shear[i, 1:], label=f'Input displacement - {i + 1}')
    plt.plot(new_input_displacement[i], real_shear[i], label=f'True Loop - {i + 1}')
    # plt.plot(new_input_displacement[i, :-1], real_shear[i, 1:], label=f'Input displacement - {i + 1}')
    plt.xlabel('Displacement', fontdict={'fontname': 'Cambria', 'fontstyle': 'italic', 'size': 14})
    plt.ylabel('Shear Load', fontdict={'fontname': 'Cambria', 'fontstyle': 'italic', 'size': 14})
    plt.title('Predicted Hysteresis', fontdict={'fontname': 'Cambria', 'fontstyle': 'normal', 'size': 16})
    plt.yticks(fontname='Cambria', fontsize=14)
    plt.xticks(fontname='Cambria', fontsize=14)
    plt.tight_layout()
    plt.legend()
    plt.grid()
    plt.show()