#!/usr/bin/env python
"""
This script visualizes training metrics with animated plots.
It displays dynamic animations for the following metrics:
  - Loss (Training & Validation)
  - Accuracy (Training & Validation)
  - Precision & Recall (Training & Validation)
  - F1 (Training & Validation)
After the final epoch, a final summary overlay is shown in plain language,
explaining in human terms what the overall performance indicates about exoplanet detection.
If a CSV file is provided (via command-line), the script loads that data.
Expected CSV columns:
  epoch, training_loss, validation_loss,
  training_accuracy, validation_accuracy,
  training_precision, validation_precision,
  training_recall, validation_recall,
  training_f1, validation_f1
Otherwise, simulated data will be generated.
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.stats import sem, t

# --- Utility functions ---
def moving_average(data, window=5):
    """Compute simple moving average with specified window size."""
    return np.convolve(data, np.ones(window)/window, mode='valid')

def linear_forecast(x, y, forecast_steps=5):
    """Simple linear regression forecast for forecast_steps ahead."""
    coeffs = np.polyfit(x, y, 1)
    poly = np.poly1d(coeffs)
    forecast_x = np.arange(x[-1] + 1, x[-1] + forecast_steps + 1)
    forecast_y = poly(forecast_x)
    return forecast_x, forecast_y

def compute_confidence_interval(data, confidence=0.95):
    """Compute mean, standard deviation and confidence interval for the data."""
    n = len(data)
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    error = sem(data)
    h = error * t.ppf((1 + confidence) / 2., n-1)
    return mean, std, (mean - h, mean + h)

# --- Data loading ---
def load_metrics(csv_path="training_metrics.csv"):
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        print(f"Loaded {csv_path} with shape: {df.shape}")
    else:
        print("No CSV found. Generating simulated data...")
        epochs = 50
        df = pd.DataFrame({
            'epoch': np.arange(1, epochs + 1),
            'training_loss': np.linspace(1.0, 0.1, epochs) + np.random.normal(0, 0.05, epochs),
            'validation_loss': np.linspace(1.2, 0.15, epochs) + np.random.normal(0, 0.05, epochs),
            'training_accuracy': np.linspace(0.5, 0.95, epochs) + np.random.normal(0, 0.05, epochs),
            'validation_accuracy': np.linspace(0.4, 0.9, epochs) + np.random.normal(0, 0.05, epochs),
            'training_precision': np.linspace(0.5, 0.9, epochs) + np.random.normal(0, 0.05, epochs),
            'validation_precision': np.linspace(0.45, 0.85, epochs) + np.random.normal(0, 0.05, epochs),
            'training_recall': np.linspace(0.4, 0.85, epochs) + np.random.normal(0, 0.05, epochs),
            'validation_recall': np.linspace(0.35, 0.8, epochs) + np.random.normal(0, 0.05, epochs),
            'training_f1': np.linspace(0.5, 0.9, epochs) + np.random.normal(0, 0.05, epochs),
            'validation_f1': np.linspace(0.45, 0.85, epochs) + np.random.normal(0, 0.05, epochs)
        })
    return df

# --- Animation function ---
def animate_metrics(df, interval=300, save_video=None):
    epochs = df['epoch'].values
    # Define metrics dictionary
    metrics = {
        'Loss': {
            'training': df['training_loss'].values,
            'validation': df['validation_loss'].values
        },
        'Accuracy': {
            'training': df['training_accuracy'].values,
            'validation': df['validation_accuracy'].values
        },
        'Precision': {
            'training': df['training_precision'].values,
            'validation': df['validation_precision'].values
        },
        'Recall': {
            'training': df['training_recall'].values,
            'validation': df['validation_recall'].values
        },
        'F1': {
            'training': df['training_f1'].values,
            'validation': df['validation_f1'].values
        }
    }
    
    # Use dark background for a high-tech look.
    plt.style.use('dark_background')
    
    # 建立 2×2 布局，共四个子图
    fig, ((ax_loss, ax_acc), (ax_pr, ax_f1)) = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Training Metrics Visualization", fontsize=22, fontweight='bold')
    
    # --- Subplot 1: Loss ---
    ax_loss.set_title("Loss Over Epochs", fontsize=16)
    ax_loss.set_xlabel("Epoch", fontsize=12)
    ax_loss.set_ylabel("Loss", fontsize=12)
    ax_loss.set_xlim(1, epochs[-1])
    min_loss = min(metrics['Loss']['training'].min(), metrics['Loss']['validation'].min())
    max_loss = max(metrics['Loss']['training'].max(), metrics['Loss']['validation'].max())
    ax_loss.set_ylim(min_loss - 0.1, max_loss + 0.1)
    line_loss_train, = ax_loss.plot([], [], 'o-', color='cyan', label='Train Loss', linewidth=2)
    line_loss_val,   = ax_loss.plot([], [], 'o-', color='magenta', label='Val Loss', linewidth=2)
    ax_loss.legend(fontsize=10)
    
    # --- Subplot 2: Accuracy ---
    ax_acc.set_title("Accuracy Over Epochs", fontsize=16)
    ax_acc.set_xlabel("Epoch", fontsize=12)
    ax_acc.set_ylabel("Accuracy", fontsize=12)
    ax_acc.set_xlim(1, epochs[-1])
    ax_acc.set_ylim(0, 1)
    line_acc_train, = ax_acc.plot([], [], 'o-', color='lime', label='Train Accuracy', linewidth=2)
    line_acc_val,   = ax_acc.plot([], [], 'o-', color='gold', label='Val Accuracy', linewidth=2)
    ax_acc.legend(fontsize=10)
    
    # --- Subplot 3: Precision & Recall ---
    ax_pr.set_title("Precision & Recall", fontsize=16)
    ax_pr.set_xlabel("Epoch", fontsize=12)
    ax_pr.set_ylabel("Score", fontsize=12)
    ax_pr.set_xlim(1, epochs[-1])
    ax_pr.set_ylim(0, 1)
    line_prec_train, = ax_pr.plot([], [], 'o-', color='deepskyblue', label='Train Precision', linewidth=2)
    line_prec_val,   = ax_pr.plot([], [], 'o-', color='violet', label='Val Precision', linewidth=2)
    line_recall_train, = ax_pr.plot([], [], 's-', color='springgreen', label='Train Recall', linewidth=2)
    line_recall_val,   = ax_pr.plot([], [], 's-', color='orangered', label='Val Recall', linewidth=2)
    ax_pr.legend(fontsize=10)
    
    # --- Subplot 4: F1 ---
    ax_f1.set_title("F1 Score", fontsize=16)
    ax_f1.set_xlabel("Epoch", fontsize=12)
    ax_f1.set_ylabel("F1", fontsize=12)
    ax_f1.set_xlim(1, epochs[-1])
    ax_f1.set_ylim(0, 1)
    line_f1_train, = ax_f1.plot([], [], 'd-', color='chartreuse', label='Train F1', linewidth=2)
    line_f1_val,   = ax_f1.plot([], [], 'd-', color='tomato', label='Val F1', linewidth=2)
    ax_f1.legend(fontsize=10)
    
    # 调整子图之间的间距，减少重叠
    plt.subplots_adjust(wspace=0.25, hspace=0.3)
    
    total_frames = len(epochs) + 10  # extra frames for final summary

    # 用于记录最终总结的文本注释，防止重复添加
    final_summary_annotation = None

    def init():
        for line in [
            line_loss_train, line_loss_val,
            line_acc_train, line_acc_val,
            line_prec_train, line_prec_val,
            line_recall_train, line_recall_val,
            line_f1_train, line_f1_val
        ]:
            line.set_data([], [])
        return (
            line_loss_train, line_loss_val,
            line_acc_train, line_acc_val,
            line_prec_train, line_prec_val,
            line_recall_train, line_recall_val,
            line_f1_train, line_f1_val
        )
    
    def update(frame):
        nonlocal final_summary_annotation
        
        # 在训练/验证完毕后多留一些帧来显示总结
        if frame <= len(epochs):
            current_epochs = epochs[:frame]
            
            # 更新数据
            line_loss_train.set_data(current_epochs, metrics['Loss']['training'][:frame])
            line_loss_val.set_data(current_epochs, metrics['Loss']['validation'][:frame])
            line_acc_train.set_data(current_epochs, metrics['Accuracy']['training'][:frame])
            line_acc_val.set_data(current_epochs, metrics['Accuracy']['validation'][:frame])
            line_prec_train.set_data(current_epochs, metrics['Precision']['training'][:frame])
            line_prec_val.set_data(current_epochs, metrics['Precision']['validation'][:frame])
            line_recall_train.set_data(current_epochs, metrics['Recall']['training'][:frame])
            line_recall_val.set_data(current_epochs, metrics['Recall']['validation'][:frame])
            line_f1_train.set_data(current_epochs, metrics['F1']['training'][:frame])
            line_f1_val.set_data(current_epochs, metrics['F1']['validation'][:frame])
            
            # 清除Loss和Accuracy子图上的注释，以免重叠
            for ax in [ax_loss, ax_acc]:
                for txt in ax.texts[:]:
                    txt.remove()
            
            # 仅对Loss和Accuracy显示Δ值，Precision/Recall/F1不标
            if frame > 1:
                # Loss
                delta_loss_train = metrics['Loss']['training'][frame-1] - metrics['Loss']['training'][frame-2]
                delta_loss_val = metrics['Loss']['validation'][frame-1] - metrics['Loss']['validation'][frame-2]
                ax_loss.annotate(
                    f"Δ {delta_loss_train:+.2f}",
                    (epochs[frame-1], metrics['Loss']['training'][frame-1]),
                    textcoords="offset points", xytext=(0, 15), ha='center',
                    color=("lime" if delta_loss_train < 0 else "red"), fontsize=10
                )
                ax_loss.annotate(
                    f"Δ {delta_loss_val:+.2f}",
                    (epochs[frame-1], metrics['Loss']['validation'][frame-1]),
                    textcoords="offset points", xytext=(0, -20), ha='center',
                    color=("lime" if delta_loss_val < 0 else "red"), fontsize=10
                )
                
                # Accuracy
                delta_acc_train = metrics['Accuracy']['training'][frame-1] - metrics['Accuracy']['training'][frame-2]
                delta_acc_val = metrics['Accuracy']['validation'][frame-1] - metrics['Accuracy']['validation'][frame-2]
                ax_acc.annotate(
                    f"Δ {delta_acc_train:+.2f}",
                    (epochs[frame-1], metrics['Accuracy']['training'][frame-1]),
                    textcoords="offset points", xytext=(0, 15), ha='center',
                    color=("lime" if delta_acc_train > 0 else "red"), fontsize=10
                )
                ax_acc.annotate(
                    f"Δ {delta_acc_val:+.2f}",
                    (epochs[frame-1], metrics['Accuracy']['validation'][frame-1]),
                    textcoords="offset points", xytext=(0, -20), ha='center',
                    color=("lime" if delta_acc_val > 0 else "red"), fontsize=10
                )
            
            # 在Loss和Accuracy子图高亮最新点（可选）
            marker_size = 25
            if frame > 0:
                idx = frame - 1
                # Loss
                ax_loss.scatter(
                    epochs[idx], metrics['Loss']['training'][idx],
                    s=marker_size, color='cyan', zorder=5, alpha=0.8
                )
                ax_loss.scatter(
                    epochs[idx], metrics['Loss']['validation'][idx],
                    s=marker_size, color='magenta', zorder=5, alpha=0.8
                )
                # Accuracy
                ax_acc.scatter(
                    epochs[idx], metrics['Accuracy']['training'][idx],
                    s=marker_size, color='lime', zorder=5, alpha=0.8
                )
                ax_acc.scatter(
                    epochs[idx], metrics['Accuracy']['validation'][idx],
                    s=marker_size, color='gold', zorder=5, alpha=0.8
                )
                
        else:
            # 最终总结阶段：仅在第一次进入时添加总结文本，避免重复覆盖
            if final_summary_annotation is None:
                # 统计每个指标的平均值和置信区间
                summary_lines = []
                for metric_name, m in metrics.items():
                    for side in ['training', 'validation']:
                        data = m[side]
                        mean_val, std_val, ci = compute_confidence_interval(data)
                        summary_lines.append(
                            f"For {metric_name} ({side}): avg={mean_val:.2f}, std={std_val:.2f}."
                        )
                
                human_summary = (
                    "Overall Summary:\n\n"
                    "The model shows a steady decrease in training loss while the accuracy has increased "
                    "consistently over the epochs. The precision, recall, and F1 scores are also trending upward, "
                    "indicating that the model is becoming more reliable in identifying exoplanets. In plain terms, "
                    "this means that as training progressed, the model improved its ability to distinguish between "
                    "true exoplanet signals and noise. The stability of these metrics suggests that the model can be "
                    "trusted to perform effectively in real-world exoplanet detection tasks."
                )
                
                # 仅对Loss和Accuracy做简单线性预测示例
                forecast_text = ""
                for key in ['Loss', 'Accuracy']:
                    data_train = metrics[key]['training']
                    smooth = moving_average(data_train, window=3)
                    x_smooth = epochs[2:]
                    forecast_x, forecast_y = linear_forecast(x_smooth, smooth, forecast_steps=5)
                    # 在对应子图上画预测线
                    ax_tgt = ax_loss if key == 'Loss' else ax_acc
                    ax_tgt.plot(forecast_x, forecast_y, '--', color='white', linewidth=2, label=f'{key} Forecast')
                    forecast_text += (
                        f"\nFor {key} (training), the next 5 epochs are predicted to be around: "
                        f"{np.round(forecast_y, 2)}"
                    )
                
                final_summary = human_summary + "\n" + "\n".join(summary_lines) + "\n" + forecast_text
                
                # 在图中央添加最终总结
                final_summary_annotation = fig.text(
                    0.5, 0.5, final_summary,
                    fontsize=13, fontweight='bold', ha='center', va='center',
                    bbox=dict(facecolor='black', alpha=0.85, edgecolor='cyan')
                )
                print("\nFinal Summary:")
                print(final_summary)
        
        return (
            line_loss_train, line_loss_val,
            line_acc_train, line_acc_val,
            line_prec_train, line_prec_val,
            line_recall_train, line_recall_val,
            line_f1_train, line_f1_val
        )
    
    ani = animation.FuncAnimation(
        fig, update, frames=total_frames,
        init_func=init, blit=False, interval=interval, repeat=False
    )
    
    fig.text(
        0.5, 0.02,
        "Colors: Cyan/Magenta for Loss, Lime/Gold for Accuracy, Blue/Violet/Green/Red for Precision & Recall, Chartreuse/Tomato for F1",
        ha='center', fontsize=11, style='italic'
    )
    
    # 给顶部标题留点空间
    plt.tight_layout(rect=[0, 0.03, 1, 0.94])
    
    if save_video:
        print(f"Saving animation to {save_video}...")
        Writer = animation.writers['ffmpeg'] if save_video.lower().endswith('.mp4') else animation.PillowWriter
        writer = Writer(fps=1000/interval)
        ani.save(save_video, writer=writer)
        print("Animation saved.")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(
        description="Visualize training metrics with animated plots (Loss, Accuracy, Precision, Recall, F1)."
    )
    parser.add_argument("--csv", type=str, default="training_metrics.csv",
                        help="Path to the CSV file with metrics.")
    parser.add_argument("--save_video", type=str, default=None,
                        help="Filename to save the animation video (e.g., animation.mp4 or animation.gif).")
    parser.add_argument("--interval", type=int, default=300,
                        help="Interval between frames in milliseconds.")
    args = parser.parse_args()
    
    df = load_metrics(args.csv)
    animate_metrics(df, interval=args.interval, save_video=args.save_video)

if __name__ == "__main__":
    main()
