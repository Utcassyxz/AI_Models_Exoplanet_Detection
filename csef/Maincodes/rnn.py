import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np

# For train/test split, scaling, and evaluation metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

############################################
# 1. Define the RNN (LSTM) model
############################################
class RNNModel(nn.Module):
    def __init__(self, input_length, input_size=1, hidden_size=64, num_layers=1):
        """
        Args:
            input_length (int): Number of time steps (i.e. sequence length).
            input_size (int): Number of features per time step (default=1).
            hidden_size (int): Number of features in the hidden state.
            num_layers (int): Number of recurrent layers.
        """
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layer; batch_first=True means input shape: (batch, seq_len, input_size)
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, 
                            num_layers=num_layers, batch_first=True)
        
        # A fully connected layer that maps the hidden state at the last time step to a single output
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        # Initialize hidden state and cell state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        
        # Forward propagate LSTM
        out, (hn, cn) = self.lstm(x, (h0, c0))
        
        # out has shape: (batch_size, seq_len, hidden_size)
        # Use the output at the last time step for classification
        last_output = out[:, -1, :]
        x = self.fc(last_output)
        x = self.sigmoid(x)
        return x

############################################
# 2. Train/Evaluate function with model naming functionality
############################################
def train_and_evaluate_rnn(csv_file_path, epochs=50, batch_size=32, lr=1e-3, model_name="RNN_Model"):
    """
    Reads CSV, splits into train/test sets, scales data, trains the RNN model, 
    evaluates on the test set using accuracy, precision, recall, and F1 score,
    and saves the model with a user-specified name.
    """
    # 1. Load CSV data
    df = pd.read_csv(csv_file_path)
    
    # Check if "label" column exists; if not, assume the last column is the label
    if "label" in df.columns:
        X = df.drop(columns=["label"]).values
        y = df["label"].values
    else:
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
    
    print(f"Data shape: {X.shape}, Labels shape: {y.shape}")
    print("Unique labels:", np.unique(y))
    
    # 2. Quick check of label distribution
    label_counts = np.bincount(y.astype(int))
    for label_val, count in enumerate(label_counts):
        print(f"Label {label_val} count: {count}")
    
    # 3. Train/test split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
    
    # 4. Scale the data (fit on train, apply on test)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)
    
    # 5. Convert to tensors
    # For RNNs with batch_first=True, input shape should be: (batch_size, seq_len, input_size)
    # Here, we treat each feature as a time step with a single feature (input_size=1).
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).unsqueeze(-1)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).unsqueeze(-1)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)
    
    # Create DataLoader for training
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # 6. Initialize model, loss function, and optimizer
    input_length = X_train.shape[1]  # Number of time steps
    model = RNNModel(input_length=input_length, input_size=1, hidden_size=64, num_layers=1)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # 7. Training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        avg_loss = running_loss / len(train_loader)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}")
    
    # 8. Evaluation on the test set
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_tensor).squeeze()
        test_preds   = (test_outputs > 0.5).int()
        
        # Convert to NumPy for metrics calculation
        y_true = y_test_tensor.squeeze().int().numpy()
        y_pred = test_preds.numpy()
        
        acc       = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall    = recall_score(y_true, y_pred, zero_division=0)
        f1        = f1_score(y_true, y_pred, zero_division=0)
    
    print("\n=== Final Evaluation on Test Set ===")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    
    # 9. Save the trained model with the user-specified name
    save_path = f"{model_name}.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Model saved as '{save_path}'")

############################################
# 3. Main entry
############################################
if __name__ == "__main__":
    # Change this to your CSV file path
    csv_file_path = "synthetic_exoplanet_dataset_1000.csv"
    
    # Allow the user to input a custom model name; default to 'RNN_Model' if left blank.
    user_model_name = input("Enter the name for the AI model (default 'RNN_Model'): ").strip()
    if not user_model_name:
        user_model_name = "RNN_Model"
    
    train_and_evaluate_rnn(
        csv_file_path=csv_file_path,
        epochs=50,
        batch_size=32,
        lr=1e-3,
        model_name=user_model_name
    )
