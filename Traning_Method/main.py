import os
import time
import math
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

# Create output directories if they don't exist
os.makedirs("models", exist_ok=True)
os.makedirs("figures", exist_ok=True)

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# =================== Data Generation Function ===================
def generate_data(n_samples, seq_length=100, noise_level=0.002):
    """
    Generate synthetic light curve data.
    - Positive samples (label 1): include a transit signal.
    - Negative samples (label 0): contain only noise.
    Each light curve is a sequence of length 'seq_length'.
    """
    X = []
    y = []
    for _ in range(n_samples):
        if np.random.rand() > 0.5:
            # Negative sample: baseline of ones with Gaussian noise
            flux = np.ones(seq_length)
            flux += np.random.normal(0, noise_level, seq_length)
            y.append(0)
        else:
            # Positive sample: includes a transit signal
            radius_ratio = np.random.uniform(0.01, 0.1)   # planet/star radius ratio
            depth = radius_ratio ** 2                    # transit depth approximation
            period = np.random.uniform(0.5, 10.0)         # orbital period in days
            duration = np.random.randint(1, 21)           # transit duration (number of points)
            flux = np.ones(seq_length)
            t0 = 0.5  # transit center time
            time_arr = np.linspace(0, 1, seq_length)
            transit_times = [t0]
            # Add future transit times within [0,1]
            k = 1
            while t0 + k * period <= 1:
                transit_times.append(t0 + k * period)
                k += 1
            # Add past transit times within [0,1]
            k = 1
            while t0 - k * period >= 0:
                transit_times.append(t0 - k * period)
                k += 1
            # For each transit, subtract depth over a duration window
            for t in transit_times:
                center_idx = int(round(t * (seq_length - 1)))
                start_idx = max(0, center_idx - duration // 2)
                end_idx = min(seq_length, start_idx + duration)
                flux[start_idx:end_idx] -= depth
            flux += np.random.normal(0, noise_level, seq_length)
            y.append(1)
        X.append(flux)
    return np.array(X), np.array(y)

# =================== PyTorch Model Definitions ===================

class CNNModel(nn.Module):
    def __init__(self, seq_length=100):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3)
        self.pool = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3)
        # 计算经过两次卷积和池化后的序列长度（假设无填充）
        # 第一次卷积: (100 - 3 + 1)=98, 池化后: 98//2=49;
        # 第二次卷积: (49 - 3 + 1)=47, 池化后: 47//2=23
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 23, 64)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # x: (batch, seq_length, 1) -> 转换为 (batch, 1, seq_length)
        x = x.transpose(1, 2)
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

class RNNModel(nn.Module):
    def __init__(self):
        super(RNNModel, self).__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=64, batch_first=True)
        self.fc = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # x: (batch, seq_length, 1)
        output, (h_n, c_n) = self.lstm(x)
        # 使用最后一层隐藏状态
        x = h_n[-1]
        x = self.fc(x)
        x = self.sigmoid(x)
        return x

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
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            outputs = outputs.view(-1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(dataloader.dataset)
        history['loss'].append(epoch_loss)
        # 验证集计算
        model.eval()
        with torch.no_grad():
            val_inputs = val_data.to(device)
            val_labels = val_labels.to(device)
            val_outputs = model(val_inputs)
            val_outputs = val_outputs.view(-1)
            val_loss = criterion(val_outputs, val_labels).item()
        history['val_loss'].append(val_loss)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f} - Val Loss: {val_loss:.4f}")
    return model, history

def build_rf():
    """
    Initialize and return a RandomForestClassifier.
    """
    return RandomForestClassifier(n_estimators=100, random_state=42)

# =================== Main Script ===================
if __name__ == "__main__":
    # ---- User Input ----
    model_choice = input("Select model (CNN, RNN, RF, BLS or 'all'): ").strip().upper()
    if model_choice not in ["CNN", "RNN", "RF", "BLS", "ALL"] or model_choice == "":
        print("Invalid choice. Defaulting to 'ALL'.")
        model_choice = "ALL"

    try:
        data_size = int(input("Select dataset size (1000, 10000, or 50000): ").strip())
        if data_size not in [1000, 10000, 50000]:
            print("Invalid choice. Defaulting to 1000 samples.")
            data_size = 1000
    except Exception:
        print("Invalid input. Using default size 1000 samples.")
        data_size = 1000

    # ---- Data Generation ----
    seq_length = 100  # length of each light curve
    print(f"\nGenerating {data_size} synthetic light curve samples...")
    X, y = generate_data(data_size, seq_length=seq_length)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    print(f"Data generated. Training samples: {len(X_train)}, Testing samples: {len(X_test)}")
    print(f"Positive class proportion in training set: {100 * np.mean(y_train):.2f}%")
    # Reshape for neural network models (CNN/RNN require 3D input)
    X_train_nn = X_train.reshape(-1, seq_length, 1)
    X_test_nn  = X_test.reshape(-1, seq_length, 1)

    # 将训练数据转换为 torch tensor，并划分训练/验证集（80/20）
    X_train_tensor = torch.tensor(X_train_nn, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_train_tensor_train, X_train_tensor_val, y_train_tensor_train, y_train_tensor_val = train_test_split(
        X_train_tensor, y_train_tensor, test_size=0.2, random_state=42)

    # ---- Determine Models to Run ----
    run_cnn = (model_choice == "ALL" or model_choice == "CNN")
    run_rnn = (model_choice == "ALL" or model_choice == "RNN")
    run_rf  = (model_choice == "ALL" or model_choice == "RF")
    run_bls = (model_choice == "ALL" or model_choice == "BLS")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Model Training ----
    if run_cnn:
        print("\nTraining CNN model with PyTorch...")
        model_cnn = CNNModel(seq_length=seq_length)
        start_time = time.time()
        model_cnn, history_cnn = train_model(model_cnn,
                                             X_train_tensor_train, y_train_tensor_train,
                                             X_train_tensor_val, y_train_tensor_val,
                                             epochs=10, batch_size=32, device=device)
        cnn_time = time.time() - start_time
        print(f"CNN training completed in {cnn_time:.2f} seconds.")
        torch.save(model_cnn.state_dict(), os.path.join("models", "cnn_model.pth"))

    if run_rnn:
        print("\nTraining RNN model with PyTorch...")
        model_rnn = RNNModel()
        start_time = time.time()
        model_rnn, history_rnn = train_model(model_rnn,
                                             X_train_tensor_train, y_train_tensor_train,
                                             X_train_tensor_val, y_train_tensor_val,
                                             epochs=10, batch_size=32, device=device)
        rnn_time = time.time() - start_time
        print(f"RNN training completed in {rnn_time:.2f} seconds.")
        torch.save(model_rnn.state_dict(), os.path.join("models", "rnn_model.pth"))

    if run_rf:
        print("\nTraining Random Forest model...")
        model_rf = build_rf()
        start_time = time.time()
        model_rf.fit(X_train, y_train)
        rf_time = time.time() - start_time
        print(f"Random Forest training completed in {rf_time:.2f} seconds.")
        joblib.dump(model_rf, os.path.join("models", "rf_model.joblib"))

    if run_bls:
        print("\nRunning BLS algorithm on test data... (This may take a while)")
        def run_bls_on_sample(flux, durations=np.linspace(0.01, 0.2, 5)):
            t = np.linspace(0, 1, len(flux))
            bls = BoxLeastSquares(t, flux)
            result = bls.autopower(durations, minimum_period=0.5, maximum_period=10.0, frequency_factor=0.1)
            return np.max(result.power)
        start_time = time.time()
        bls_scores = [run_bls_on_sample(flux) for flux in X_test]
        bls_time = time.time() - start_time
        # Use median of BLS scores as threshold
        bls_threshold = np.median(bls_scores)
        y_pred_bls = np.array([1 if score >= bls_threshold else 0 for score in bls_scores])
        print(f"BLS processing completed in {bls_time:.2f} seconds.")

    # ---- Model Evaluation ----
    # 转换测试数据为 tensor
    X_test_tensor = torch.tensor(X_test_nn, dtype=torch.float32)

    if run_cnn:
        model_cnn.eval()
        with torch.no_grad():
            outputs = model_cnn(X_test_tensor.to(device)).view(-1)
            y_pred_prob_cnn = outputs.cpu().numpy()
        y_pred_label_cnn = (y_pred_prob_cnn >= 0.5).astype(int)
        f1_cnn = f1_score(y_test, y_pred_label_cnn)
        auc_cnn = roc_auc_score(y_test, y_pred_prob_cnn)
        cm_cnn = confusion_matrix(y_test, y_pred_label_cnn)
    if run_rnn:
        model_rnn.eval()
        with torch.no_grad():
            outputs = model_rnn(X_test_tensor.to(device)).view(-1)
            y_pred_prob_rnn = outputs.cpu().numpy()
        y_pred_label_rnn = (y_pred_prob_rnn >= 0.5).astype(int)
        f1_rnn = f1_score(y_test, y_pred_label_rnn)
        auc_rnn = roc_auc_score(y_test, y_pred_prob_rnn)
        cm_rnn = confusion_matrix(y_test, y_pred_label_rnn)
    if run_rf:
        y_pred_label_rf = model_rf.predict(X_test)
        y_pred_prob_rf = model_rf.predict_proba(X_test)[:, 1]
        f1_rf = f1_score(y_test, y_pred_label_rf)
        auc_rf = roc_auc_score(y_test, y_pred_prob_rf)
        cm_rf = confusion_matrix(y_test, y_pred_label_rf)
    if run_bls:
        f1_bls = f1_score(y_test, y_pred_bls)
        try:
            auc_bls = roc_auc_score(y_test, bls_scores)
        except ValueError:
            auc_bls = 0.0
        cm_bls = confusion_matrix(y_test, y_pred_bls)

    print("\n========== Evaluation Metrics ==========")
    if run_cnn:
        print(f"CNN:   F1 = {f1_cnn:.4f}, ROC-AUC = {auc_cnn:.4f}")
    if run_rnn:
        print(f"RNN:   F1 = {f1_rnn:.4f}, ROC-AUC = {auc_rnn:.4f}")
    if run_rf:
        print(f"RF:    F1 = {f1_rf:.4f}, ROC-AUC = {auc_rf:.4f}")
    if run_bls:
        print(f"BLS:   F1 = {f1_bls:.4f}, ROC-AUC = {auc_bls:.4f}")
    times = []
    if run_cnn: times.append(f"CNN = {cnn_time:.2f}")
    if run_rnn: times.append(f"RNN = {rnn_time:.2f}")
    if run_rf:  times.append(f"RF = {rf_time:.2f}")
    if run_bls: times.append(f"BLS = {bls_time:.2f}")
    print("\nTraining Times (seconds): " + ", ".join(times))

    # ---- Visualization ----
    # Plot 1: Loss curves (for CNN and/or RNN)
    if run_cnn or run_rnn:
        if run_cnn and run_rnn:
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            axes[0].plot(history_cnn['loss'], label='Train Loss')
            axes[0].plot(history_cnn.get('val_loss', []), label='Val Loss')
            axes[0].set_title("CNN Training Loss")
            axes[0].set_xlabel("Epoch")
            axes[0].set_ylabel("Loss")
            axes[0].legend()
            axes[1].plot(history_rnn['loss'], label='Train Loss')
            axes[1].plot(history_rnn.get('val_loss', []), label='Val Loss')
            axes[1].set_title("RNN Training Loss")
            axes[1].set_xlabel("Epoch")
            axes[1].set_ylabel("Loss")
            axes[1].legend()
            plt.tight_layout()
            plt.savefig(os.path.join("figures", "loss_curves.png"))
            plt.close()
        elif run_cnn:
            plt.figure(figsize=(5, 4))
            plt.plot(history_cnn['loss'], label='Train Loss')
            if 'val_loss' in history_cnn:
                plt.plot(history_cnn['val_loss'], label='Val Loss')
            plt.title("CNN Training Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join("figures", "loss_curves.png"))
            plt.close()
        elif run_rnn:
            plt.figure(figsize=(5, 4))
            plt.plot(history_rnn['loss'], label='Train Loss')
            if 'val_loss' in history_rnn:
                plt.plot(history_rnn['val_loss'], label='Val Loss')
            plt.title("RNN Training Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join("figures", "loss_curves.png"))
            plt.close()

    # Plot 2: ROC curves
    plt.figure(figsize=(6, 4))
    if run_cnn:
        fpr, tpr, _ = roc_curve(y_test, y_pred_prob_cnn)
        plt.plot(fpr, tpr, label=f"CNN (AUC={auc_cnn:.2f})")
    if run_rnn:
        fpr, tpr, _ = roc_curve(y_test, y_pred_prob_rnn)
        plt.plot(fpr, tpr, label=f"RNN (AUC={auc_rnn:.2f})")
    if run_rf:
        fpr, tpr, _ = roc_curve(y_test, y_pred_prob_rf)
        plt.plot(fpr, tpr, label=f"RF (AUC={auc_rf:.2f})")
    if run_bls:
        fpr, tpr, _ = roc_curve(y_test, bls_scores)
        plt.plot(fpr, tpr, label=f"BLS (AUC={auc_bls:.2f})")
    plt.plot([0, 1], [0, 1], 'k--', label="Chance")
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join("figures", "roc_curve.png"))
    plt.close()

    # Plot 3: Precision-Recall curves
    plt.figure(figsize=(6, 4))
    if run_cnn:
        prec, rec, _ = precision_recall_curve(y_test, y_pred_prob_cnn)
        plt.plot(rec, prec, label="CNN")
    if run_rnn:
        prec, rec, _ = precision_recall_curve(y_test, y_pred_prob_rnn)
        plt.plot(rec, prec, label="RNN")
    if run_rf:
        prec, rec, _ = precision_recall_curve(y_test, y_pred_prob_rf)
        plt.plot(rec, prec, label="RF")
    if run_bls:
        prec, rec, _ = precision_recall_curve(y_test, bls_scores)
        plt.plot(rec, prec, label="BLS")
    plt.axhline(y=np.mean(y_test), color='k', linestyle='--', label=f"Baseline ({np.mean(y_test):.2f})")
    plt.title("Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(os.path.join("figures", "pr_curve.png"))
    plt.close()

    # Plot 4: Confusion matrix heatmaps
    matrices = []
    titles = []
    if run_cnn:
        matrices.append(cm_cnn); titles.append("CNN Confusion Matrix")
    if run_rnn:
        matrices.append(cm_rnn); titles.append("RNN Confusion Matrix")
    if run_rf:
        matrices.append(cm_rf); titles.append("RF Confusion Matrix")
    if run_bls:
        matrices.append(cm_bls); titles.append("BLS Confusion Matrix")
    if len(matrices) == 1:
        plt.figure(figsize=(4, 3))
        sns.heatmap(matrices[0], annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=[0, 1], yticklabels=[0, 1])
        plt.title(titles[0])
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.tight_layout()
        plt.savefig(os.path.join("figures", "confusion_matrices.png"))
        plt.close()
    elif len(matrices) > 1:
        cols = len(matrices)
        fig, axes = plt.subplots(1, cols, figsize=(4 * cols, 3))
        if cols == 1:
            axes = [axes]
        for ax, mat, ttl in zip(axes, matrices, titles):
            sns.heatmap(mat, annot=True, fmt="d", cmap="Blues", cbar=False,
                        xticklabels=[0, 1], yticklabels=[0, 1], ax=ax)
            ax.set_title(ttl)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
        plt.tight_layout()
        plt.savefig(os.path.join("figures", "confusion_matrices.png"))
        plt.close()

    # Plot 5: Comparison bar charts for F1, ROC-AUC, and Training Time
    models_list = []
    f1_list = []
    auc_list = []
    time_list = []
    if run_cnn:
        models_list.append("CNN"); f1_list.append(f1_cnn); auc_list.append(auc_cnn); time_list.append(cnn_time)
    if run_rnn:
        models_list.append("RNN"); f1_list.append(f1_rnn); auc_list.append(auc_rnn); time_list.append(rnn_time)
    if run_rf:
        models_list.append("RF"); f1_list.append(f1_rf); auc_list.append(auc_rf); time_list.append(rf_time)
    if run_bls:
        models_list.append("BLS"); f1_list.append(f1_bls); auc_list.append(auc_bls); time_list.append(bls_time)
    if len(models_list) >= 2:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].bar(models_list, f1_list, color='skyblue')
        axes[0].set_title("F1 Score")
        axes[0].set_ylim(0, 1)
        axes[1].bar(models_list, auc_list, color='lightgreen')
        axes[1].set_title("ROC-AUC")
        axes[1].set_ylim(0, 1)
        axes[2].bar(models_list, time_list, color='salmon')
        axes[2].set_title("Training Time (s)")
        axes[2].set_ylim(0, max(time_list) * 1.2)
        plt.tight_layout()
        plt.savefig(os.path.join("figures", "model_comparison.png"))
        plt.close()

    # Plot 6: Error sample plot (e.g., CNN correct but RF wrong)
    if run_cnn and run_rf:
        cnn_correct = (y_pred_label_cnn == y_test)
        rf_correct = (y_pred_label_rf == y_test)
        error_idx = None
        for i in range(len(y_test)):
            if y_test[i] == 1 and cnn_correct[i] and not rf_correct[i]:
                error_idx = i
                break
        if error_idx is None:
            for i in range(len(y_test)):
                if y_test[i] == 0 and cnn_correct[i] and not rf_correct[i]:
                    error_idx = i
                    break
        if error_idx is not None:
            plt.figure(figsize=(6, 4))
            plt.plot(X_test[error_idx], label="Light Curve")
            plt.title(f"Sample {error_idx}: True={y_test[error_idx]}, CNN={y_pred_label_cnn[error_idx]}, RF={y_pred_label_rf[error_idx]}")
            plt.xlabel("Time Step")
            plt.ylabel("Normalized Flux")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join("figures", "error_sample.png"))
            plt.close()
            print(f"Saved error sample plot for sample index {error_idx}.")
        else:
            print("No error sample found (CNN correct but RF wrong).")

    # ---- Save Metrics Summary ----
    summary_df = pd.DataFrame({
        "Model": models_list,
        "F1 Score": f1_list,
        "ROC-AUC": auc_list,
        "Training Time (s)": time_list
    })
    summary_df["F1 Score"] = summary_df["F1 Score"].round(4)
    summary_df["ROC-AUC"] = summary_df["ROC-AUC"].round(4)
    summary_df["Training Time (s)"] = summary_df["Training Time (s)"].round(2)
    summary_df.to_csv("metrics_summary.csv", index=False)
    print("\nMetrics summary saved to 'metrics_summary.csv'.")

    # ---- Animated Bar Chart ----
    if len(models_list) >= 2:
        fig, ax = plt.subplots(figsize=(6 if len(models_list) <= 3 else 7.5, 4))
        bars = ax.bar(models_list, f1_list, color='mediumpurple')
        ax.set_ylim(0, 1.2)
        ax.set_title("Model Comparison - F1 Score")
        ax.set_ylabel("Score")
        def animate(frame):
            metric = frame % 3
            if metric == 0:
                values = f1_list
                ax.set_title("Model Comparison - F1 Score")
                ax.set_ylabel("Score")
                ax.set_ylim(0, 1.2)
            elif metric == 1:
                values = auc_list
                ax.set_title("Model Comparison - ROC-AUC")
                ax.set_ylabel("Score")
                ax.set_ylim(0, 1.2)
            elif metric == 2:
                max_time_val = max(time_list) if time_list else 1
                values = [t / max_time_val for t in time_list]
                ax.set_title("Model Comparison - Training Time (Normalized)")
                ax.set_ylabel("Normalized Time")
                ax.set_ylim(0, 1.2)
            for bar, val in zip(bars, values):
                bar.set_height(val)
            return bars
        anim = animation.FuncAnimation(fig, animate, frames=30, interval=1000, blit=False, repeat=True)
        # 尝试使用 ffmpeg 保存，如果不可用则保存为 GIF
        anim_file = os.path.join("figures", "model_comparison_animation.mp4")
        try:
            anim.save(anim_file, writer="ffmpeg")
            print(f"Animated bar chart saved as '{anim_file}'.")
        except Exception as e:
            print("ffmpeg not available or error encountered:", e)
            anim_file = os.path.join("figures", "model_comparison_animation.gif")
            anim.save(anim_file, writer="pillow")
            print(f"Animated bar chart saved as '{anim_file}'.")
        plt.close()

    # ---- Final Message ----
    if model_choice == "ALL":
        print("\nAll models and figures have been saved successfully!")
    else:
        print(f"\n{model_choice} model and figures have been saved successfully!")
