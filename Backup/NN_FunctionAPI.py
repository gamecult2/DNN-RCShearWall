"""
Neural Network Model with Function API
Status : Working Need More Checking
"""
import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import LSTM, Dense, Input, Concatenate, Reshape, concatenate, Flatten
import keras.callbacks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


tf.config.list_physical_devices(device_type=None)
physical_devices = tf.config.list_physical_devices('GPU')
print("Num GPUs:", len(physical_devices))

# Generate random data for demonstration
num_samples = 1000  # Number of samples to feed the Neural Network
num_features = 2  # Number of columns in GM curve (Just acceleration)
gma_sequence_length = 300  # Maximum points in the GM curve and displacement
st_sequence_length = 5  # Number of Structural parameters

np.random.seed(0)

# Generate random acceleration time series
gma_data = np.random.uniform(-3, 2, size=(num_samples, gma_sequence_length, num_features)).astype(np.float32).round(5)

# Generate random influencing parameter data
influencing_parameters = np.random.uniform(0, 2, size=(num_samples, st_sequence_length)).astype(np.float32).round(5)

# Generate random displacement data (target)
# displacement_data = np.random.uniform(0, 1, size=(num_samples, gma_sequence_length))
displacement_data = gma_data[:, :, 0] * 2

# Split data into training, validation, and testing sets
X_acceleration_train, X_acceleration_test, X_influencing_train, X_influencing_test, y_train, y_test = train_test_split(
    gma_data, influencing_parameters, displacement_data, test_size=0.2, random_state=42
)

# Build the neural network model using functional API
# Layer 1
acceleration_input = Input(shape=(gma_sequence_length, num_features), name='acceleration_input')
lstm_layer = LSTM(gma_sequence_length, return_sequences=True)(acceleration_input)  # LSTM layer for acceleration
flat1 = Flatten()(lstm_layer)

# Layer 2
parameters_input = Input(shape=(st_sequence_length,), name='parameters_input')
dense_layer = Dense(gma_sequence_length)(parameters_input)  # Dense layer for influencing parameters
flat2 = Flatten()(dense_layer)

# Merge the 2 inputs layer with concatenate LSTM and Dense layers
merged = concatenate([flat1, flat2])

# Output layer for displacement
displacement_output = Dense(gma_sequence_length)(merged)

model = Model(inputs=[acceleration_input, parameters_input], outputs=displacement_output)

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])  # , metrics=['mean_absolute_error']
model.summary()

# Define the checkpoint callback
early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss",  # Loss to monitor for stopping
    patience=5,  # stop training after 5 non-improved training
    verbose=2
)

# Train the model
history = model.fit(
    [X_acceleration_train, X_influencing_train],  # Input layer (GMA + STRUCTURAL PARAMETERS)
    y_train,  # Output layer (DISPLACEMENT)
    epochs=30,
    batch_size=32,
    validation_split=0.1,
    callbacks=[early_stopping]  # checkpoint_callback or early_stopping
)
model.save("NN_functionalAPI_DynamicSystem")  # Save the model after training

# Plot the training and validation loss
plt.plot(history.history['loss'], label="Train Loss")
plt.plot(history.history['val_loss'], label="Val Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training and Validation Loss Over Epochs")
plt.legend()
plt.show()

# Evaluate the model
loss = model.evaluate([X_acceleration_test, X_influencing_test], y_test)
print("Test loss:", loss)

test_index = 10
new_acceleration_sequence = X_acceleration_test[test_index:test_index + 1]  # Select a single example
new_influencing_parameters = X_influencing_test[test_index:test_index + 1]  # Select corresponding influencing parameters
real_displacement_sequence = y_test[test_index:test_index + 1]

# Predict displacement for the new data
predicted_displacement = model.predict([new_acceleration_sequence, new_influencing_parameters])

# Plot the predicted displacement
plt.figure(figsize=(10, 6))
plt.plot(predicted_displacement[0], label='Predicted Displacement')
plt.plot(real_displacement_sequence[0], label='True displacement')
plt.xlabel('Time Step', {'fontname': 'Cambria', 'fontstyle': 'italic', 'size': 14})
plt.ylabel('Displacement', {'fontname': 'Cambria', 'fontstyle': 'italic', 'size': 14})
plt.title('Predicted Displacement Time Series', {'fontname': 'Cambria', 'fontstyle': 'normal', 'size': 16})
plt.yticks(fontname='Cambria', fontsize=14)
plt.xticks(fontname='Cambria', fontsize=14)
plt.legend()
plt.grid()
plt.show()

new_acceleration_sequence2 = np.random.uniform(-1, 1, size=(1, gma_sequence_length, num_features))
new_influencing_parameters2 = np.random.uniform(-0, 4, size=(1, st_sequence_length))
real_displacement_sequence2 = new_acceleration_sequence2[:, :, 0] * 2

# Predict displacement for the new data
predicted_displacement2 = model.predict([new_acceleration_sequence2, new_influencing_parameters2])

# Plot the predicted displacement
plt.figure(figsize=(10, 6))
plt.plot(predicted_displacement2[0], label='Predicted Displacement')
plt.plot(real_displacement_sequence2[0], label='True displacement')
plt.xlabel('Time Step', {'fontstyle': 'italic', 'size': 14})
plt.ylabel('Displacement', {'fontstyle': 'italic', 'size': 14})
plt.title('Predicted Displacement Time Series', {'fontstyle': 'normal', 'size': 16})
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
plt.legend()
plt.grid()
plt.show()