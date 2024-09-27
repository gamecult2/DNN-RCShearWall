from keras.models import Model
from keras.optimizers import Adam, SGD
from keras.layers import LSTM, Dense, Input, Concatenate, Reshape, concatenate, Flatten, Bidirectional, Conv1D, GlobalMaxPooling1D, Softmax, Dropout, Activation, CuDNNLSTM, MultiHeadAttention, MaxPooling1D, LayerNormalization, Add, TimeDistributed, RepeatVector, Lambda, Attention, Multiply
import keras.callbacks
import matplotlib.pyplot as plt
from keras.saving.save import load_model
from sklearn.model_selection import train_test_split
from RCWall_DataProcessing import *

# Define the number of sample to be used
batch_size = 3000  # 3404
num_features = 1  # Number of columns in InputDisplacement curve (Just One Displacement Column with fixed Dt)
sequence_length = 500
parameters_length = 10
num_features_input_displacement = 1
num_features_input_parameters = 10

returned_data, returned_scaler = read_data(batch_size, sequence_length, normalize_data=True, save_normalized_data=False, smoothed_data=False)
InParams, InDisp, OutCycShear, OutPushShear = returned_data
param_scaler, disp_scaler, cyc_shear_scaler, cyc_pushover_scaler = returned_scaler

# ---------------------- Split Data -------------------------------
# Split data into training, validation, and testing sets (X: Inputs & Y: Outputs)
X_param_train, X_param_test, X_disp_train, X_disp_test, Y_shear_train, Y_shear_test, Y_pushover_train, Y_pushover_test = train_test_split(
    InParams, InDisp, OutCycShear, OutPushShear, test_size=0.15, random_state=42)

# ---------------------- NN Model Building -------------------------
# Load the individual models
model_cnn = load_model("NN_CNN")
model_lstm = load_model("NN_DeepLSTM")
model_autoencoder = load_model("NN_Autoencoder")

# Build the neural network model using functional API
parameters_input = Input(shape=(num_features_input_parameters,), name='parameters_input')
displacement_input = Input(shape=(None, num_features_input_displacement), name='displacement_input')

# Get the output of each model
output_cnn = model_cnn([parameters_input, displacement_input])
output_lstm = model_lstm([parameters_input, displacement_input])
output_autoencoder = model_autoencoder([parameters_input, displacement_input])
print('output_cnn', output_cnn.shape)
print('output_lstm', output_lstm.shape)
print('output_autoencoder', output_autoencoder.shape)

# Concatenate the outputs
merged = concatenate([output_cnn, output_lstm, output_autoencoder])
print('merged', merged.shape)
# ---------------------- Output layer --------------------------------------------
output_shear = Flatten()(Dense(1, name='output_shear')(merged))
print('output_shear', output_shear.shape)

# ---------------------- Build the model ------------------------------------------
# Build the model
model = Model(inputs=[parameters_input, displacement_input], outputs=output_shear)

# ---------------------- Compile the model -----------------------------------------
# Define Adam and SGD optimizers
adam_optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=adam_optimizer,
              loss='mse',
              metrics=['mse'])

# ---------------------- Print Model summary ---------------------------------------------
model.summary()

# ---------------------- Define the checkpoint callback ----------------------------
early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss",  # Loss to monitor for stopping
    patience=100,  # stop training after 10 non-improved training
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
# model.save("NN_DeepLSTM03")  # Save the model after training

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

# ---------------------- Model testing ---------------------------------------------------
# Evaluate the model
loss = model.evaluate([X_param_test, X_disp_test], [Y_shear_test])
print("Test loss:", loss)

# ---------------------- Plotting the results ---------------------------------------------
test_index = 3

new_input_parameters = X_param_test[0:test_index]  # Select corresponding influencing parameters
new_input_displacement = X_disp_test[0:test_index]  # Select a single example
real_shear = Y_shear_test[0:test_index]

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
real_shear = denormalize(real_shear, cyc_shear_scaler, sequence=True)
predicted_shear = denormalize(predicted_shear, cyc_shear_scaler, sequence=True)

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
