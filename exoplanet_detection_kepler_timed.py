import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import lightkurve as lk
from sklearn.model_selection import train_test_split
import time

# 开始整体计时
start_time = time.time()

###########################################
# Step 1: 下载 Kepler 光变曲线数据
###########################################
print("Fetching Kepler light curve data...")
data_fetch_start = time.time()

# 搜索一个 Kepler 目标（例如 KIC 8462852，即著名的 Tabby's Star）
search_result = lk.search_lightcurve('KIC 8462852', mission='Kepler')

# 下载所有可用的光变曲线文件
lc_collection = search_result.download_all()

# 将多个光变曲线拼接成一个更长的曲线
lc = lc_collection.stitch()

data_fetch_end = time.time()
print(f"Data fetching completed in {data_fetch_end - data_fetch_start:.2f} seconds.")

###########################################
# Step 2: 数据预处理
###########################################
print("Preprocessing light curve data...")
preprocess_start = time.time()

# 对光变曲线归一化与去噪
normalized_lc = lc.normalize()
cleaned_lc = normalized_lc.remove_nans()

# 转换为 Pandas DataFrame
df = cleaned_lc.to_pandas()

# 提取时间和 flux（光度）数据
time_series = np.array(df.index)   # 时间值
flux_series = np.array(df.flux)      # 光度值

# 对 flux 做归一化处理（零均值、单位方差）
flux_series = (flux_series - np.mean(flux_series)) / np.std(flux_series)

# 检查数据点是否足够
if len(flux_series) < 100:
    raise ValueError("Not enough data points after cleaning. Try a different Kepler target.")

# 提示用户选择使用的数据点数（100, 10000, 或 50000）
valid_choices = ['100', '10000', '50000']
data_size_input = input("Enter the number of data points to use (choose 100, 10000, or 50000): ").strip()
if data_size_input not in valid_choices:
    print("Invalid input. Defaulting to 10000 data points.")
    data_size = 10000
else:
    data_size = int(data_size_input)

# 如果实际数据点不足，则使用全部数据
if data_size > len(flux_series):
    print(f"Requested {data_size} data points, but only {len(flux_series)} are available. Using all available data.")
    data_size = len(flux_series)

# 截取指定数量的数据
flux_series = flux_series[:data_size]

# 生成合成标签（模拟：1 表示有外行星凌日，0 表示无；此处随机注入 10% 的 1）
labels = np.zeros(len(flux_series))
indices_with_transit = np.random.choice(len(labels), size=len(labels) // 10, replace=False)
labels[indices_with_transit] = 1

# 调整数据形状
# 注意：这里每个样本为一个数据点（即单个 flux 值），故 shape 为 (数据点数, 1)
flux_series = flux_series.reshape(-1, 1)
labels = labels.reshape(-1)

preprocess_end = time.time()
print(f"Data preprocessing completed in {preprocess_end - preprocess_start:.2f} seconds.")
print(f"Final data shape: {flux_series.shape}, labels shape: {labels.shape}")

###########################################
# Step 3: 划分训练集和测试集
###########################################
X_train, X_test, y_train, y_test = train_test_split(flux_series, labels, test_size=0.2, random_state=42)

# 转换为 PyTorch 张量
# 注意：为了符合 1D CNN 输入格式，使用 unsqueeze(1) 将数据扩展为 (batch, channels, length)
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

print("Input shape for CNN:", X_train_tensor.shape)

###########################################
# Step 4: 定义 1D CNN 模型
###########################################
class ExoplanetDetector(nn.Module):
    def __init__(self, input_length):
        """
        input_length: 每个样本中的数据点数量（即卷积层最后 flatten 时的长度因子）
        """
        super(ExoplanetDetector, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        # 经过两层卷积后，特征图尺寸不变（因为用了 padding），因此 flatten 后的特征数量为 64 * input_length
        self.fc1 = nn.Linear(64 * input_length, 128)
        self.fc2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # x 的形状: (batch_size, 1, input_length)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x.squeeze()  # 确保输出形状正确

# 根据训练数据的每个样本长度确定模型输入尺寸
input_length = X_train_tensor.shape[2]
model = ExoplanetDetector(input_length)

###########################################
# Step 5: 定义损失函数和优化器
###########################################
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

###########################################
# Step 6: 训练模型（并记录损失以便后续可视化）
###########################################
num_epochs = 10
train_start = time.time()
print("Training the 1D CNN model...")

# 用于记录每个 epoch 的训练损失
training_losses = []

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)  # 前向传播
    loss = criterion(outputs, y_train_tensor)  # 计算损失
    loss.backward()  # 反向传播
    optimizer.step()  # 更新权重

    training_losses.append(loss.item())
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}')

train_end = time.time()
print(f"Training completed in {train_end - train_start:.2f} seconds.")

# 绘制训练损失曲线
plt.figure(figsize=(8, 5))
plt.plot(range(1, num_epochs + 1), training_losses, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.title('Training Loss Curve')
plt.grid(True)
plt.show()

###########################################
# Step 7: 保存训练好的模型
###########################################
# 提示用户输入模型名称（若留空则使用默认名称）
model_name = input("Enter new AI model name (default 'kepler_exoplanet_detector'): ").strip()
if not model_name:
    model_name = "kepler_exoplanet_detector"

save_path = f"{model_name}.pth"
torch.save(model.state_dict(), save_path)
print(f"Model training complete and saved as '{save_path}'.")

###########################################
# Step 8: 模型评估
###########################################
eval_start = time.time()
model.eval()
with torch.no_grad():
    predictions = model(X_test_tensor)
    predicted_labels = (predictions > 0.5).float()
    accuracy = (predicted_labels == y_test_tensor).sum().item() / len(y_test_tensor)
    print(f'Test Accuracy: {accuracy * 100:.2f}%')
eval_end = time.time()
print(f"Evaluation completed in {eval_end - eval_start:.2f} seconds.")

###########################################
# Step 9: 绘制测试样本光变曲线及其预测结果
###########################################
plt.figure(figsize=(12, 5))
plt.plot(X_test_tensor.squeeze().numpy(), label="Test Light Curve", alpha=0.7)
pred_label_str = 'Exoplanet' if predicted_labels[0] == 1 else 'No Exoplanet'
plt.title(f"Sample Test Light Curve - Predicted: {pred_label_str}")
plt.xlabel("Time (sample index)")
plt.ylabel("Normalized Flux")
plt.legend()
plt.show()

# 结束整体计时
end_time = time.time()
print(f"Total execution time: {end_time - start_time:.2f} seconds.")
