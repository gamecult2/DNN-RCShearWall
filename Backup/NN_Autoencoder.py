"""
Neural Network Model with Encoder-Decoder layer
"""

import numpy as np
from tensorflow import keras
from keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, RepeatVector, TimeDistributed
import matplotlib.pyplot as plt
# import visualkeras
# from ann_visualizer.visualize import ann_viz

# Generate synthetic data ------TODO Change with real data
np.random.seed(0)
num_samples = 1000              # Number of samples to feed the Neural Network
num_features = 1               # Number of columns in GM curve (Just acceleration)
gm_sequence_length = 100        # Maximum length of GM curve
st_sequence_length = 10         # The number of structural parameters

# Inputs data
structural_data = np.random.uniform(0, 1, (num_samples, st_sequence_length, num_features))
ground_motion_data = np.random.uniform(0, 1, (num_samples, gm_sequence_length, num_features))
# Combine structural_data and ground_motion_data (Generate an array of 110 length)
combined_data = np.concatenate((structural_data, ground_motion_data), axis=1)

# Plot the predicted displacement data.
#
# plt.plot(ground_motion_data[0, :, 0], 100, label='Predicted displacement')
# plt.legend()
# plt.show()

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

# target_shape = ((Y_train.shape[0], gm_sequence_length + st_sequence_length, num_features))
# padding_size = target_shape[1] - Y_train.shape[1]
# padding = np.zeros((Y_train.shape[0], padding_size, 1))    # Create an array of zeros to pad Y_train
# Y_train = np.concatenate((Y_train, padding), axis=1)  # Reshape and add the padding

# Define and compile the model (the encoder-decoder architecture)
model = keras.Sequential([
    # Encoder
    LSTM(128, activation='relu', input_shape=(gm_sequence_length + st_sequence_length, num_features), ),  #return_sequences=True
    # Repeat the output of the encoder for each time step in the output sequence
    RepeatVector(gm_sequence_length),
    # Decoder
    LSTM(units=128, return_sequences=True),
    # LSTM(64, activation='relu', return_sequences=True),
    # TimeDistributed layer to apply the same dense layer to each time step
    TimeDistributed(Dense(units=num_features))
    # Dense(units=gm_sequence_length, activation='linear')  # Output shape matches gm_sequence_length
])

model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()
# Train the model
epochs = 100
batch_size = 32
model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=epochs, batch_size=batch_size)

# Evaluate the model
mse = model.evaluate(X_test, Y_test)
print("Test Mean Squared Error:", mse)

# Make predictions (Generate Random Data to predict the equivalent displacement)
# new_acceleration_sequence = np.random.uniform(0, 1, (gm_sequence_length+st_sequence_length, num_features))
new_acceleration_sequence = X_test[0]

# Predict the displacement sequence
predicted_displacement_sequence = model.predict(np.array([new_acceleration_sequence]))

# Reshape the output tensor
# predicted_displacement_sequence = predicted_displacement_sequence.reshape(
#     (predicted_displacement_sequence.shape[0], predicted_displacement_sequence.shape[1], 1)
# )

print("New Acceleration Sequence:")
print(new_acceleration_sequence)
print("Predicted Displacement Sequence:")
print(predicted_displacement_sequence)

# Plot the predicted displacement data.
plt.plot(Y_test[0], label='True displacement')
plt.plot(predicted_displacement_sequence[0, :, 0], label='Predicted displacement')
plt.legend()
plt.show()
# gamma = list(zip(new_acceleration_sequence, predicted_displacement_sequence))
# print(gamma)