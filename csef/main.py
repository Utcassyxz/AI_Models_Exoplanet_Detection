import os
import time
import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import animation
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, precision_recall_curve, roc_curve
from astropy.timeseries import BoxLeastSquares
from sklearn.ensemble import RandomForestClassifier
import joblib
import torch
import torch.nn as nn
import torch.optim as optim

# -------------------- Helper Functions --------------------
def prompt_custom_code():
    """
    Prompt the user to enter a custom project code in fixed format.
    For example: PROJECT_001.
    The code must be alphanumeric (plus underscores).
    """
    while True:
        code = input("Please enter your custom project code (e.g., PROJECT_001): ").strip()
        if re.fullmatch(r"[A-Za-z0-9_]+", code):
            return code
        else:
            print("Invalid format. Please use only letters, digits and underscores.")

def get_filename(base_name, custom_code, ext):
    """
    Construct a filename using the base name, custom code and extension.
    """
    return f"{base_name}_{custom_code}.{ext}"

def interpret_score(score):
    """Convert a numeric score into a qualitative description."""
    if score >= 0.90:
        return "Excellent"
    elif score >= 0.80:
        return "Very Good"
    elif score >= 0.70:
        return "Good"
    elif score >= 0.60:
        return "Fair"
    else:
        return "Poor"

def interpret_time(time_val, best_time):
    """Compare training time relative to the best (fastest) model."""
    ratio = time_val / best_time if best_time > 0 else 1
    if ratio <= 1.2:
        return "Fast"
    elif ratio <= 1.5:
        return "Moderate"
    else:
        return "Slow"

# -------------------- Environment Check & Directories --------------------
if "CONDA_DEFAULT_ENV" in os.environ:
    print("Anaconda environment detected:", os.environ["CONDA_DEFAULT_ENV"])
else:
    print("It is recommended to run this code in an Anaconda environment for optimal performance.")

os.makedirs("SavedModels", exist_ok=True)
os.makedirs("ModelVisualization", exist_ok=True)
os.makedirs("DataAnalysis", exist_ok=True)

np.random.seed(42)
torch.manual_seed(42)

# -------------------- Global Parameters & User Input --------------------
MODEL_CHOICE = input("Select model (ALL, CNN-COMPARE, CNN-A, CNN-B, CNN-C, RNN, RF, BLS): ").strip().upper()
if MODEL_CHOICE not in ["ALL", "CNN-COMPARE", "CNN-A", "CNN-B", "CNN-C", "RNN", "RF", "BLS"]:
    print("Invalid choice. Defaulting to ALL.")
    MODEL_CHOICE = "ALL"

try:
    DATA_SIZE = int(input("Select dataset size (1000, 10000, or 50000): ").strip())
    if DATA_SIZE not in [1000, 10000, 50000]:
        print("Invalid choice. Defaulting to 1000 samples.")
        DATA_SIZE = 1000
except Exception:
    print("Invalid input. Using default size 1000 samples.")
    DATA_SIZE = 1000

SEQ_LENGTH = 100  # length of each light curve

custom_code = prompt_custom_code()

# Construct output file names using the custom code
LABEL_DIST_FILENAME       = get_filename("label_distribution", custom_code, "png")
LOSS_CURVES_FILENAME      = get_filename("loss_curves", custom_code, "png")
ROC_CURVE_FILENAME        = get_filename("roc_curve", custom_code, "png")
PR_CURVE_FILENAME         = get_filename("pr_curve", custom_code, "png")
CONFUSION_MATRIX_FILENAME = get_filename("confusion_matrices", custom_code, "png")
CNN_COMPARE_BAR_FILENAME  = get_filename("cnn_comparison_bar", custom_code, "png")
CNN_COMPARE_ANIM_FILENAME = get_filename("cnn_comparison_animation", custom_code, "mp4")
OVERALL_COMP_BAR_FILENAME = get_filename("overall_model_comparison", custom_code, "png")
OVERALL_COMP_ANIM_FILENAME= get_filename("overall_model_comparison_animation", custom_code, "mp4")
METRICS_SUMMARY_FILENAME  = get_filename("metrics_summary", custom_code, "csv")

# -------------------- Data Generation --------------------
def generate_data(n_samples, seq_length=100, noise_level=0.002):
    """
    Generate synthetic light curve data.
    Positive samples (label 1) include a transit signal.
    Negative samples (label 0) contain only noise.
    """
    X, y = [], []
    for _ in range(n_samples):
        if np.random.rand() > 0.5:
            flux = np.ones(seq_length) + np.random.normal(0, noise_level, seq_length)
            y.append(0)
        else:
            radius_ratio = np.random.uniform(0.01, 0.1)
            depth = radius_ratio ** 2
            period = np.random.uniform(0.5, 10.0)
            duration = np.random.randint(1, 21)
            flux = np.ones(seq_length)
            t0 = 0.5
            transit_times = [t0]
            k = 1
            while t0 + k * period <= 1:
                transit_times.append(t0 + k * period)
                k += 1
            k = 1
            while t0 - k * period >= 0:
                transit_times.append(t0 - k * period)
                k += 1
            for t in transit_times:
                center_idx = int(round(t * (seq_length - 1)))
                start_idx = max(0, center_idx - duration // 2)
                end_idx = min(seq_length, start_idx + duration)
                flux[start_idx:end_idx] -= depth
            flux += np.random.normal(0, noise_level, seq_length)
            y.append(1)
        X.append(flux)
    return np.array(X), np.array(y)

print(f"\nGenerating {DATA_SIZE} synthetic light curve samples...")
data_save_path = os.path.join("DataAnalysis", f"generated_data_{custom_code}.npz")
if os.path.exists(data_save_path):
    use_saved = input(f"Saved data found at {data_save_path}. Use this data for training? (Y/N): ").strip().upper()
    if use_saved == 'Y':
        data = np.load(data_save_path)
        X, y = data['X'], data['y']
    else:
        X, y = generate_data(DATA_SIZE, seq_length=SEQ_LENGTH)
        np.savez(data_save_path, X=X, y=y)
else:
    X, y = generate_data(DATA_SIZE, seq_length=SEQ_LENGTH)
    np.savez(data_save_path, X=X, y=y)
print(f"Data generated: {len(X)} samples.")
print(f"Positive sample proportion: {100 * np.mean(y):.2f}%")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# -------------------- Data Analysis Visualization --------------------
plt.figure(figsize=(5, 4))
unique_vals, counts = np.unique(y, return_counts=True)
bars = plt.bar(unique_vals, counts, tick_label=["No Transit", "Transit"])
plt.xlabel("Class")
plt.ylabel("Count")
plt.title("Label Distribution")
plt.grid(axis="y", linestyle="--", alpha=0.7)
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 2, f'{int(height)}', ha='center', va='bottom', fontsize=9)
plt.figtext(0.5, 0.01, "This plot shows the number of samples in each class.", wrap=True, horizontalalignment='center', fontsize=9)
plt.tight_layout()
label_path = os.path.join("DataAnalysis", LABEL_DIST_FILENAME)
plt.savefig(label_path, bbox_inches="tight")
plt.close()
print(f"Label distribution plot saved to {label_path}")

# -------------------- Prepare Data for NN Models --------------------
X_train_nn = X_train.reshape(-1, SEQ_LENGTH, 1)
X_test_nn  = X_test.reshape(-1, SEQ_LENGTH, 1)
X_train_tensor = torch.tensor(X_train_nn, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_train_tensor_train, X_train_tensor_val, y_train_tensor_train, y_train_tensor_val = train_test_split(
    X_train_tensor, y_train_tensor, test_size=0.2, random_state=42
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------- CNN Model Descriptions --------------------
print("\n=== CNN Model Descriptions ===")
print("1. Basic CNN: 2 convolutional layers. Suitable for simple, short sequence tasks. Fast and lightweight.")
print("2. Deep CNN: 3 convolutional layers with batch normalization and dropout. Suitable for complex, multi-scale feature extraction (requires more resources).")
print("3. Lightweight CNN: Uses fewer filters, ideal for resource-constrained or real-time applications (may trade off some accuracy).")

# -------------------- Model Definitions --------------------
class CNN_A(nn.Module):
    def __init__(self, seq_length=100):
        super(CNN_A, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3)
        self.pool = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3)
        L_out = ((seq_length - 2) // 2 - 2) // 2  
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * L_out, 64)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

class CNN_B(nn.Module):
    def __init__(self):
        super(CNN_B, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.25),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.25),
            nn.AdaptiveAvgPool1d(1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class CNN_C(nn.Module):
    def __init__(self):
        super(CNN_C, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=3, padding=1),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(8, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.AdaptiveAvgPool1d(1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class RNNModel(nn.Module):
    def __init__(self):
        super(RNNModel, self).__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=64, batch_first=True)
        self.fc = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        output, (h_n, c_n) = self.lstm(x)
        x = h_n[-1]
        x = self.fc(x)
        x = self.sigmoid(x)
        return x

# -------------------- Training Function --------------------
def train_model(model, train_data, train_labels, val_data, val_labels, epochs, batch_size, device):
    model.to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters())
    dataset = torch.utils.data.TensorDataset(train_data, train_labels)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    history = {'loss': [], 'val_loss': []}
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs).view(-1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(dataloader.dataset)
        history['loss'].append(epoch_loss)
        model.eval()
        with torch.no_grad():
            val_outputs = model(val_data.to(device)).view(-1)
            val_loss = criterion(val_outputs, val_labels.to(device)).item()
        history['val_loss'].append(val_loss)
        print(f"Epoch {epoch+1}/10 - Loss: {epoch_loss:.4f} - Val Loss: {val_loss:.4f}")
    return model, history

def build_rf():
    return RandomForestClassifier(n_estimators=100, random_state=42)

# -------------------- Main Script --------------------
def main():
    if MODEL_CHOICE == "ALL":
        models_to_train = {
            "Basic CNN": CNN_A(seq_length=SEQ_LENGTH),
            "Deep CNN": CNN_B(),
            "Lightweight CNN": CNN_C(),
            "RNN": RNNModel(),
            "RF": "RF",
            "BLS": "BLS"
        }
    elif MODEL_CHOICE == "CNN-COMPARE" or MODEL_CHOICE == "ALL CNN":
        models_to_train = {
            "Basic CNN": CNN_A(seq_length=SEQ_LENGTH),
            "Deep CNN": CNN_B(),
            "Lightweight CNN": CNN_C()
        }
    elif MODEL_CHOICE in ["CNN-A", "CNN-B", "CNN-C"]:
        mapping = {"CNN-A": "Basic CNN", "CNN-B": "Deep CNN", "CNN-C": "Lightweight CNN"}
        key = mapping[MODEL_CHOICE]
        if MODEL_CHOICE == "CNN-A":
            models_to_train = {key: CNN_A(seq_length=SEQ_LENGTH)}
        elif MODEL_CHOICE == "CNN-B":
            models_to_train = {key: CNN_B()}
        else:
            models_to_train = {key: CNN_C()}
    elif MODEL_CHOICE in ["RNN", "RF", "BLS"]:
        models_to_train = {MODEL_CHOICE: RNNModel() if MODEL_CHOICE == "RNN" else MODEL_CHOICE}
    else:
        models_to_train = {"Basic CNN": CNN_A(seq_length=SEQ_LENGTH)}
    
    model_metrics = {}

    for model_name, model_obj in models_to_train.items():
        print(f"\n====== Training {model_name} ======")
        if model_obj in ["RF", "BLS"]:
            if model_obj == "RF":
                model_rf = build_rf()
                start = time.time()
                model_rf.fit(X_train, y_train)
                t_elapsed = time.time() - start
                y_pred_rf = model_rf.predict(X_test)
                y_pred_prob_rf = model_rf.predict_proba(X_test)[:, 1]
                f1 = f1_score(y_test, y_pred_rf)
                roc = roc_auc_score(y_test, y_pred_prob_rf)
                model_metrics[model_name] = {"F1": f1, "ROC": roc, "Time": t_elapsed}
                joblib.dump(model_rf, os.path.join("SavedModels", f"rf_model_{custom_code}.joblib"))
                print(f"{model_name}: F1 = {f1:.4f} ({interpret_score(f1)}), ROC-AUC = {roc:.4f} ({interpret_score(roc)}), Time = {t_elapsed:.2f}s")
            else:
                start = time.time()
                def run_bls_on_sample(flux, durations=np.linspace(0.01, 0.2, 5)):
                    t = np.linspace(0, 1, len(flux))
                    bls = BoxLeastSquares(t, flux)
                    result = bls.autopower(durations, minimum_period=0.5, maximum_period=10.0, frequency_factor=0.1)
                    return np.max(result.power)
                bls_scores = [run_bls_on_sample(flux) for flux in X_test]
                t_elapsed = time.time() - start
                bls_threshold = np.median(bls_scores)
                y_pred_bls = np.array([1 if score >= bls_threshold else 0 for score in bls_scores])
                f1 = f1_score(y_test, y_pred_bls)
                try:
                    roc = roc_auc_score(y_test, bls_scores)
                except ValueError:
                    roc = 0.0
                model_metrics[model_name] = {"F1": f1, "ROC": roc, "Time": t_elapsed}
                print(f"{model_name}: F1 = {f1:.4f} ({interpret_score(f1)}), ROC-AUC = {roc:.4f} ({interpret_score(roc)}), Time = {t_elapsed:.2f}s")
        else:
            start = time.time()
            trained_model, history = train_model(model_obj, X_train_tensor_train, y_train_tensor_train,
                                                 X_train_tensor_val, y_train_tensor_val,
                                                 epochs=10, batch_size=32, device=device)
            t_elapsed = time.time() - start
            torch.save(trained_model.state_dict(), os.path.join("SavedModels", f"{model_name.replace(' ', '_')}_{custom_code}.pth"))
            trained_model.eval()
            X_test_tensor = torch.tensor(X_test_nn, dtype=torch.float32)
            with torch.no_grad():
                outputs = trained_model(X_test_tensor.to(device)).view(-1)
            y_pred_prob = outputs.cpu().numpy()
            y_pred_label = (y_pred_prob >= 0.5).astype(int)
            f1 = f1_score(y_test, y_pred_label)
            roc = roc_auc_score(y_test, y_pred_prob)
            model_metrics[model_name] = {"F1": f1, "ROC": roc, "Time": t_elapsed}
            plt.figure(figsize=(5,4))
            plt.plot(history['loss'], marker='o', label='Train Loss')
            plt.plot(history['val_loss'], marker='s', label='Val Loss')
            plt.title(f"{model_name} Loss Curves")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.grid(linestyle="--", alpha=0.7)
            plt.figtext(0.5, 0.01, "Loss curves show model convergence during training.", wrap=True, horizontalalignment='center', fontsize=9)
            loss_filename = get_filename(f"loss_curves_{model_name.replace(' ', '_')}", custom_code, "png")
            plt.tight_layout()
            plt.savefig(os.path.join("ModelVisualization", loss_filename), bbox_inches="tight")
            plt.close()
            print(f"{model_name}: F1 = {f1:.4f} ({interpret_score(f1)}), ROC-AUC = {roc:.4f} ({interpret_score(roc)}), Time = {t_elapsed:.2f}s")
    
    if len(model_metrics) >= 2:
        overall_models = list(model_metrics.keys())
        overall_f1 = [model_metrics[m]["F1"] for m in overall_models]
        overall_roc = [model_metrics[m]["ROC"] for m in overall_models]
        overall_time = [model_metrics[m]["Time"] for m in overall_models]
        best_time = min(overall_time)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        bars = axes[0].bar(overall_models, overall_f1, color='skyblue')
        axes[0].set_title("Overall F1 Score")
        axes[0].set_ylabel("F1 Score")
        axes[0].set_ylim(0, 1)
        axes[0].grid(axis="y", linestyle="--", alpha=0.7)
        for bar, score in zip(bars, overall_f1):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, interpret_score(score),
                         ha='center', va='bottom', fontsize=9)
        
        bars = axes[1].bar(overall_models, overall_roc, color='lightgreen')
        axes[1].set_title("Overall ROC-AUC")
        axes[1].set_ylabel("ROC-AUC")
        axes[1].set_ylim(0, 1)
        axes[1].grid(axis="y", linestyle="--", alpha=0.7)
        for bar, score in zip(bars, overall_roc):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, interpret_score(score),
                         ha='center', va='bottom', fontsize=9)
        
        bars = axes[2].bar(overall_models, overall_time, color='salmon')
        axes[2].set_title("Overall Training/Processing Time (s)")
        axes[2].set_ylabel("Time (s)")
        axes[2].set_ylim(0, max(overall_time)*1.2)
        axes[2].grid(axis="y", linestyle="--", alpha=0.7)
        for bar, t in zip(bars, overall_time):
            axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, interpret_time(t, best_time),
                         ha='center', va='bottom', fontsize=9)
        plt.figtext(0.5, -0.02, "Overall Comparison: Left: F1 Score, Middle: ROC-AUC, Right: Training/Processing Time.\nHigher F1/ROC-AUC is better; lower time is preferable.", wrap=True, horizontalalignment='center', fontsize=9)
        plt.tight_layout()
        overall_bar_path = os.path.join("ModelVisualization", OVERALL_COMP_BAR_FILENAME)
        plt.savefig(overall_bar_path, bbox_inches="tight")
        plt.close()
        print(f"Overall comparison bar chart saved to {overall_bar_path}")
        
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(overall_models, overall_f1, color='mediumpurple')
        ax.set_ylim(0, 1.2)
        ax.set_title("Overall Model Comparison - F1 Score")
        ax.set_ylabel("Score")
        def animate(frame):
            metric = frame % 3
            if metric == 0:
                values = overall_f1
                ax.set_title("Overall Model Comparison - F1 Score")
                ax.set_ylabel("F1 Score")
                ax.set_ylim(0, 1.2)
            elif metric == 1:
                values = overall_roc
                ax.set_title("Overall Model Comparison - ROC-AUC")
                ax.set_ylabel("ROC-AUC")
                ax.set_ylim(0, 1.2)
            elif metric == 2:
                values = [t / best_time for t in overall_time]
                ax.set_title("Overall Model Comparison - Normalized Time")
                ax.set_ylabel("Normalized Time")
                ax.set_ylim(0, 1.2)
            for bar, val in zip(bars, values):
                bar.set_height(val)
            return bars
        anim = animation.FuncAnimation(fig, animate, frames=30, interval=1000, blit=False, repeat=True)
        overall_anim_path = os.path.join("ModelVisualization", OVERALL_COMP_ANIM_FILENAME)
        try:
            anim.save(overall_anim_path, writer="ffmpeg")
            print(f"Animated overall comparison saved to {overall_anim_path}")
        except Exception as e:
            overall_anim_path = os.path.join("ModelVisualization", "overall_model_comparison_animation.gif")
            anim.save(overall_anim_path, writer="pillow")
            print(f"Animated overall comparison saved to {overall_anim_path}")
        plt.close()
    
    cnn_variants = [m for m in model_metrics if m in ["Basic CNN", "Deep CNN", "Lightweight CNN"]]
    if len(cnn_variants) >= 2:
        cnn_f1 = [model_metrics[m]["F1"] for m in cnn_variants]
        cnn_roc = [model_metrics[m]["ROC"] for m in cnn_variants]
        cnn_time = [model_metrics[m]["Time"] for m in cnn_variants]
        best_cnn_time = min(cnn_time)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        bars = axes[0].bar(cnn_variants, cnn_f1, color='skyblue')
        axes[0].set_title("CNN Variants F1 Score")
        axes[0].set_ylabel("F1 Score")
        axes[0].set_ylim(0, 1)
        axes[0].grid(axis="y", linestyle="--", alpha=0.7)
        for bar, score in zip(bars, cnn_f1):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, interpret_score(score),
                         ha='center', va='bottom', fontsize=9)
        
        bars = axes[1].bar(cnn_variants, cnn_roc, color='lightgreen')
        axes[1].set_title("CNN Variants ROC-AUC")
        axes[1].set_ylabel("ROC-AUC")
        axes[1].set_ylim(0, 1)
        axes[1].grid(axis="y", linestyle="--", alpha=0.7)
        for bar, score in zip(bars, cnn_roc):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, interpret_score(score),
                         ha='center', va='bottom', fontsize=9)
        
        bars = axes[2].bar(cnn_variants, cnn_time, color='salmon')
        axes[2].set_title("CNN Variants Training Time (s)")
        axes[2].set_ylabel("Time (s)")
        axes[2].set_ylim(0, max(cnn_time)*1.2)
        axes[2].grid(axis="y", linestyle="--", alpha=0.7)
        for bar, t in zip(bars, cnn_time):
            axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, interpret_time(t, best_cnn_time),
                         ha='center', va='bottom', fontsize=9)
        plt.figtext(0.5, -0.02, "CNN Variants Comparison: Basic CNN is fast & lightweight; Deep CNN extracts complex features (but is slower); Lightweight CNN is efficient for constrained scenarios.", wrap=True, horizontalalignment='center', fontsize=9)
        plt.tight_layout()
        cnn_internal_path = os.path.join("ModelVisualization", get_filename("cnn_variants_comparison", custom_code, "png"))
        plt.savefig(cnn_internal_path, bbox_inches="tight")
        plt.close()
        print(f"CNN variants comparison chart saved to {cnn_internal_path}")
    
    summary_df = pd.DataFrame({
        "Model": list(model_metrics.keys()),
        "F1 Score": [f"{model_metrics[m]['F1']:.4f} ({interpret_score(model_metrics[m]['F1'])})" for m in model_metrics],
        "ROC-AUC": [f"{model_metrics[m]['ROC']:.4f} ({interpret_score(model_metrics[m]['ROC'])})" for m in model_metrics],
        "Time (s)": [f"{model_metrics[m]['Time']:.2f} ({interpret_time(model_metrics[m]['Time'], min([model_metrics[x]['Time'] for x in model_metrics]))})" for m in model_metrics]
    })
    summary_path = os.path.join("DataAnalysis", METRICS_SUMMARY_FILENAME)
    summary_df.to_csv(summary_path, index=False)
    print(f"\nMetrics summary saved to {summary_path}")
    
    print("\nAll models have been trained and all visualizations have been saved.")
    print("Please check the following folders for results:")
    print("  SavedModels          - Trained model files")
    print("  ModelVisualization   - Plots and figures with detailed explanations")
    print("  DataAnalysis         - Data analysis outputs (e.g., label distribution, metrics summary)")
    print("\nThank you for using this advanced platform. Enjoy your comprehensive visualization and analysis!")

if __name__ == "__main__":
    main()
