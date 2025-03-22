import matplotlib.pyplot as plt

# Dataset sizes (number of records)
dataset_sizes = [1000, 10000, 50000]

# CNN metrics (blue) — crafted for crossing lines
cnn_accuracy  = [85, 92, 96]
cnn_precision = [82, 93, 91]
cnn_recall    = [87, 89, 98]
cnn_f1        = [84, 92, 97]

# RNN metrics (green) — crafted for crossing lines
rnn_accuracy  = [77, 85, 90]
rnn_precision = [80, 82, 92]
rnn_recall    = [74, 88, 88]
rnn_f1        = [76, 89, 89]

# Random Forest metrics (red) — crafted for crossing lines
rf_accuracy   = [70, 79, 75]
rf_precision  = [72, 76, 89]
rf_recall     = [68, 82, 74]
rf_f1         = [69, 78, 76]

plt.figure(figsize=(12, 8))

# --- CNN Plots (Blue) ---
plt.plot(dataset_sizes, cnn_accuracy,  marker='o', color='blue',  linestyle='-',  label='CNN Accuracy')
plt.plot(dataset_sizes, cnn_precision, marker='s', color='blue',  linestyle='--', label='CNN Precision')
plt.plot(dataset_sizes, cnn_recall,    marker='^', color='blue',  linestyle='-.', label='CNN Recall')
plt.plot(dataset_sizes, cnn_f1,        marker='D', color='blue',  linestyle=':',  label='CNN F1 Score')

# --- RNN Plots (Green) ---
plt.plot(dataset_sizes, rnn_accuracy,  marker='o', color='green', linestyle='-',  label='RNN Accuracy')
plt.plot(dataset_sizes, rnn_precision, marker='s', color='green', linestyle='--', label='RNN Precision')
plt.plot(dataset_sizes, rnn_recall,    marker='^', color='green', linestyle='-.', label='RNN Recall')
plt.plot(dataset_sizes, rnn_f1,        marker='D', color='green', linestyle=':',  label='RNN F1 Score')

# --- Random Forest Plots (Red) ---
plt.plot(dataset_sizes, rf_accuracy,   marker='o', color='red',   linestyle='-',  label='RF Accuracy')
plt.plot(dataset_sizes, rf_precision,  marker='s', color='red',   linestyle='--', label='RF Precision')
plt.plot(dataset_sizes, rf_recall,     marker='^', color='red',   linestyle='-.', label='RF Recall')
plt.plot(dataset_sizes, rf_f1,         marker='D', color='red',   linestyle=':',  label='RF F1 Score')

# Annotate each data point with its value
def annotate_line(x, y, color):
    for xi, yi in zip(x, y):
        plt.annotate(f"{yi}%", (xi, yi), textcoords="offset points", xytext=(0,5), ha='center', color=color)

annotate_line(dataset_sizes, cnn_accuracy,  'blue')
annotate_line(dataset_sizes, cnn_precision, 'blue')
annotate_line(dataset_sizes, cnn_recall,    'blue')
annotate_line(dataset_sizes, cnn_f1,        'blue')

annotate_line(dataset_sizes, rnn_accuracy,  'green')
annotate_line(dataset_sizes, rnn_precision, 'green')
annotate_line(dataset_sizes, rnn_recall,    'green')
annotate_line(dataset_sizes, rnn_f1,        'green')

annotate_line(dataset_sizes, rf_accuracy,   'red')
annotate_line(dataset_sizes, rf_precision,  'red')
annotate_line(dataset_sizes, rf_recall,     'red')
annotate_line(dataset_sizes, rf_f1,         'red')

plt.title("Comprehensive Performance Metrics of Exoplanet Detection Models")
plt.xlabel("Dataset Size")
plt.ylabel("Percentage (%)")
plt.xscale('log')
plt.xticks(dataset_sizes, ["1,000", "10,000", "50,000"])
plt.legend(loc='upper left', fontsize='small', ncol=2)
plt.grid(True)
plt.tight_layout()
plt.show()
