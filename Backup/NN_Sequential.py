"""
Neural Network Model with Sequential layer
"""
import numpy as np
from tensorflow import keras
from keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, GRU, LSTM, Dense, RepeatVector, TimeDistributed
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt


# Generate synthetic data ------TODO Change with real data
np.random.seed(42)
num_samples = 1000                                                    # Number of samples to feed the Neural Network
num_features = 1                                                      # Number of columns in GM curve (Just acceleration)
gm_sequence_length = 100                                              # Maximum length of GM curve
st_sequence_length = 10                                               # The number of structural parameters
sequence_length = st_sequence_length + gm_sequence_length             # The length of the sequence

# Inputs data
# structural_data = np.tile(np.arange(1, st_sequence_length + 1).reshape(1, st_sequence_length, num_features), (num_samples, 1, 1))
structural_data = np.random.uniform(0, 1, (num_samples, st_sequence_length, num_features)).astype(np.float32).round(5)
ground_motion_data = np.random.uniform(0, 1, (num_samples, gm_sequence_length, num_features)).astype(np.float32).round(5)
# Combine structural_data and ground_motion_data (Generate an array of 110 length)
combined_data = np.concatenate((ground_motion_data, structural_data), axis=1).astype(np.float32).round(5)
# Output data
displacement_data = ground_motion_data * 2

# Data preprocessing and sequencing
input_sequences = []
output_sequences = []

for i in range(num_samples):
    input_sequences.append(combined_data[i])
    output_sequences.append(displacement_data[i])

X = np.array(input_sequences)
Y = np.array(output_sequences)

# Splitting into train, validation, and test sets
split_ratio = [0.6, 0.2, 0.2]
split_index_1 = int(split_ratio[0] * len(X))
split_index_2 = int((split_ratio[0] + split_ratio[1]) * len(X))
X_train, X_val, X_test = X[:split_index_1], X[split_index_1:split_index_2], X[split_index_2:]
Y_train, Y_val, Y_test = Y[:split_index_1], Y[split_index_1:split_index_2], Y[split_index_2:]

# target_shape = (Y_train.shape[0], sequence_length, num_features)
# padding_size = target_shape[1] - Y_train.shape[1]
# padding = np.zeros((Y_train.shape[0], padding_size, 1))    # Create an array of zeros to pad Y_train
# Y_train = np.concatenate((Y_train, padding), axis=1)       # Reshape and add the padding
#
# target_shape = (Y_val.shape[0], sequence_length, num_features)
# padding_size = target_shape[1] - Y_val.shape[1]
# padding = np.zeros((Y_val.shape[0], padding_size, 1))    # Create an array of zeros to pad Y_train
# Y_val = np.concatenate((Y_val, padding), axis=1)         # Reshape and add the padding
#
# target_shape = (Y_test.shape[0], sequence_length, num_features)
# padding_size = target_shape[1] - Y_test.shape[1]
# padding = np.zeros((Y_test.shape[0], padding_size, 1))    # Create an array of zeros to pad Y_train
# Y_test = np.concatenate((Y_test, padding), axis=1)        # Reshape and add the padding

# Define and compile the model
model = keras.Sequential([
    # SimpleRNN(32, input_shape=(gm_sequence_length + st_sequence_length, num_features), use_bias=False, activation='relu', return_sequences=True),
    # Encoder
    # LSTM(64, input_shape=(sequence_length, num_features), activation='relu', return_sequences=True),  #
    # Repeat the output of the encoder for each time step in the output sequence
    # RepeatVector(gm_sequence_length),
    # Decoder
    # LSTM(units=64, return_sequences=True),
    # LSTM(64, activation='relu', return_sequences=True),
    # TimeDistributed layer to apply the same dense layer to each time step
    # TimeDistributed(Dense(units=num_features))
    # Dense(100)# units=1, activation='relu'
    # Dense(units=gm_sequence_length, activation='linear')  # Output shape matches gm_sequence_length
    LSTM(64, activation='relu', input_shape=(sequence_length, num_features), return_sequences=True),
    LSTM(32, activation='relu', input_shape=(sequence_length, num_features)),
    Dense(gm_sequence_length)  # Output sequence length is set to 50
])
model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()

# Define the checkpoint callback
checkpoint_callback = ModelCheckpoint("best_model.h5", save_best_only=True, verbose=1)

# Train the model
epochs = 100
batch_size = 32

history = model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=epochs, batch_size=batch_size, callbacks=[checkpoint_callback])
# model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=epochs, batch_size=batch_size)

# Plot the training and validation loss
plt.plot(history.history['loss'], label="Train Loss")
plt.plot(history.history['val_loss'], label="Val Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training and Validation Loss Over Epochs")
plt.legend()
plt.show()

# Evaluate the model
mse = model.evaluate(X_test, Y_test)
print("Test Mean Squared Error:", mse)

# Make predictions (Generate Random Data to predict the equivalent displacement)
# new_acceleration_sequence = np.random.uniform(0, 1, (sequence_length, num_features))
new_acceleration_sequence = X_test[10]

# Predict the displacement sequence
predicted_displacement_sequence = model.predict(np.array([new_acceleration_sequence]))

print("New Acceleration Sequence:")
print(new_acceleration_sequence)
print("Predicted Displacement Sequence:")
print(predicted_displacement_sequence)

# # Plot the predicted displacement data.
# plt.plot(new_acceleration_sequence, label='True displacement')
# plt.plot(predicted_displacement_sequence[0, :], label='Predicted displacement')
# plt.legend()
# plt.show()

# Plot the predicted displacement data.
plt.plot(new_acceleration_sequence*2, label='True displacement')
plt.plot(predicted_displacement_sequence[0, :], label='Predicted displacement')
plt.legend()
plt.show()
plt.show()