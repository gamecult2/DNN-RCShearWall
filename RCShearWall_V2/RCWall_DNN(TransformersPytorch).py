import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from RCWall_DataProcessing import *

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Print device information
print(f"Using device: {device}")


# Define R2 metric
def r_square(y_true, y_pred):
    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
    return 1 - ss_res / (ss_tot + 1e-6)


# Positional Encoding for Transformer
# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model, dropout=0.1, max_len=5000):
#         super(PositionalEncoding, self).__init__()
#         self.dropout = nn.Dropout(p=dropout)
#
#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         pe = pe.unsqueeze(0).transpose(0, 1)
#         self.register_buffer('pe', pe)
#
#     def forward(self, x):
#         x = x + self.pe[:x.size(0), :]
#         return self.dropout(x)


# Model definition using Transformer without embedding
class TransformerModel(nn.Module):
    def __init__(self, input_dim=1, d_model=64, nhead=4, num_layers=2, dropout=0.2):
        super(TransformerModel, self).__init__()

        self.encoder = nn.Linear(input_dim, d_model)
        # self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.decoder = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.encoder(x)
        # x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = self.decoder(x[:, -1, :])  # Ensure x is sliced appropriately
        return x


# Define the number of samples to be used
batch_size = 100
num_features = 1
sequence_length = 499
parameters_length = 10
num_features_input_displacement = 1
num_features_input_parameters = 10
embedding_dim = 64
pushover = False

returned_data, returned_scaler = read_data(batch_size, sequence_length, normalize_data=True, save_normalized_data=False, pushover=pushover)
InParams, InDisp, OutShear = returned_data
param_scaler, disp_scaler, shear_scaler = returned_scaler

# Split data into training, validation, and testing sets (X: Inputs & Y: Outputs)
X_param_train, X_param_test, X_disp_train, X_disp_test, Y_shear_train, Y_shear_test = train_test_split(
    InParams, InDisp, OutShear, test_size=0.20, random_state=42)

# Convert to PyTorch tensors
X_param_train, X_param_test = torch.tensor(X_param_train, dtype=torch.float32), torch.tensor(X_param_test, dtype=torch.float32)
X_disp_train, X_disp_test = torch.tensor(X_disp_train, dtype=torch.float32), torch.tensor(X_disp_test, dtype=torch.float32)
Y_shear_train, Y_shear_test = torch.tensor(Y_shear_train, dtype=torch.float32), torch.tensor(Y_shear_test, dtype=torch.float32)

# Create PyTorch Datasets and DataLoaders
train_dataset = TensorDataset(X_param_train, X_disp_train, Y_shear_train)
test_dataset = TensorDataset(X_param_test, X_disp_test, Y_shear_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# Define the model
class FullModel(nn.Module):
    def __init__(self, sequence_length, d_model, nhead, num_layers):
        super(FullModel, self).__init__()
        # self.positional_encoding = PositionalEncoding(d_model, 0.1, sequence_length)
        self.transformer = TransformerModel(d_model=d_model, nhead=nhead, num_layers=num_layers, dropout=0.1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(sequence_length * d_model, 100)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(100, 100)
        self.output_layer = nn.Linear(100, 1)

    def forward(self, parameters, displacement):
        parameters = parameters.unsqueeze(1).repeat(1, sequence_length, 1)
        displacement = displacement.unsqueeze(-1)
        x = torch.cat((displacement, parameters), dim=-1)
        print('5- Dis+Param Shape = ', x.shape)
        #  = self.positional_encoding(x)
        x = self.transformer(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        output = self.output_layer(x)
        return output


# Initialize the model, loss function, and optimizer
model = FullModel(sequence_length, d_model=embedding_dim, nhead=8, num_layers=8)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 1
patience = 10
best_val_loss = float('inf')
patience_counter = 0

train_losses, val_losses = [], []

for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    for parameters, displacement, shear in train_loader:
        optimizer.zero_grad()
        output = model(parameters, displacement)
        loss = criterion(output, shear)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    val_loss = 0.0
    model.eval()
    with torch.no_grad():
        for parameters, displacement, shear in test_loader:
            output = model(parameters, displacement)
            loss = criterion(output, shear)
            val_loss += loss.item()

    train_loss /= len(train_loader)
    val_loss /= len(test_loader)
    train_losses.append(train_loss)
    val_losses.append(val_loss)

    print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), "best_model.pth")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping")
            break

# Load the best model
model.load_state_dict(torch.load("best_model.pth"))

# Plot the training and validation loss
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label="Training Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training and Validation Loss Over Epochs")
plt.legend()
plt.show()

# Model testing
model.eval()
test_loss = 0.0
with torch.no_grad():
    for parameters, displacement, shear in test_loader:
        output = model(parameters, displacement)
        loss = criterion(output, shear)
        test_loss += loss.item()

test_loss /= len(test_loader)
print("Test loss:", test_loss)

# Plotting the results
test_index = 3
new_input_parameters = X_param_test[:test_index]
new_input_displacement = X_disp_test[:test_index]
real_shear = Y_shear_test[:test_index]

# Predict displacement for the new data
model.eval()
with torch.no_grad():
    predicted_shear = model(new_input_parameters, new_input_displacement).cpu().numpy()

# Plot the predicted displacement
plt.figure(figsize=(10, 6))
for i in range(test_index):
    plt.plot(predicted_shear[i], label=f'Predicted Shear - {i + 1}')
    plt.plot(real_shear[i].cpu().numpy(), label=f'Real Shear - {i + 1}')
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
for i in range(test_index):
    plt.plot(new_input_displacement[i].cpu().numpy(), predicted_shear[i], label=f'Predicted Loop - {i + 1}')
    plt.plot(new_input_displacement[i].cpu().numpy(), real_shear[i].cpu().numpy(), label=f'Real Loop - {i + 1}')
    plt.xlabel('Displacement', fontdict={'fontname': 'Cambria', 'fontstyle': 'italic', 'size': 14})
    plt.ylabel('Shear Load', fontdict={'fontname': 'Cambria', 'fontstyle': 'italic', 'size': 14})
    plt.title('Predicted Shear vs Displacement', {'fontname': 'Cambria', 'fontstyle': 'normal', 'size': 16})
    plt.yticks(fontname='Cambria', fontsize=14)
    plt.xticks(fontname='Cambria', fontsize=14)
    plt.tight_layout()
    plt.legend()
    plt.grid()
    plt.show()
