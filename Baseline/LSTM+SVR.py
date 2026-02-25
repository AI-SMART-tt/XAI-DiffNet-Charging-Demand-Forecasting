import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import time
import os
import matplotlib.pyplot as plt
import geopandas as gpd

# 可调参数集中定义
MODEL_NAME = "SVR"  # 模型名称，可选"LSTM"或"SVR"，此处使用LSTM
LEARNING_RATE = 0.001  # 学习率
BATCH_SIZE = 32  # 批量大小
EPOCHS = 200  # 训练轮数
HIDDEN_DIM = 64  # LSTM隐藏层维度
NUM_LAYERS = 2  # LSTM层数
SEQUENCE_LENGTH = 12  # 输入时间序列长度（过去5分钟间隔的12个点，即1小时）
FORECAST_STEPS = 3  # 预测未来步长（默认1，预测下一时间步）
TRAIN_SPLIT = 0.6  # 训练集比例
VAL_SPLIT = 0.2  # 验证集比例
TEST_SPLIT = 0.2  # 测试集比例
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 设备选择
EPSILON = 1  # MAPE分母的常数
OUTPUT_DIR = "output"  # 输出目录
SHP_PATH = "data_shap/上海市_with_ID_merged.shp"  # SHP文件路径
CMAP = 'BuGn'  # 地图颜色映射
ALPHA = 0.8  # 地图面要素透明度，范围 [0, 1]
LOSS_CURVE_COLORS = ['blue', 'orange']  # 损失曲线颜色：[训练曲线, 验证曲线]

# 创建输出目录
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# 设置Matplotlib使用英文字体
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.sans-serif'] = ['Arial']


# 数据加载与预处理
def load_and_preprocess_data():
    """加载并预处理数据"""
    town_ids = pd.read_csv("data_graph/graph2/town_ids.csv")
    town_list = town_ids['ID'].tolist()
    town_names = town_ids['乡'].tolist()
    charging_data = pd.read_csv("data0/charging_energy_all.csv")
    charging_data['time'] = pd.to_datetime(charging_data['time'])
    charging_data.set_index('time', inplace=True)
    charging_data.fillna(0, inplace=True)
    return charging_data, town_list, town_names


# 创建时间序列数据（支持多步预测）
def create_sequences(data, seq_length, forecast_steps):
    """创建时间序列数据，支持多步预测"""
    X, y = [], []
    for i in range(len(data) - seq_length - forecast_steps + 1):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length:i + seq_length + forecast_steps])
    return np.array(X), np.array(y)


# 定义LSTM模型（支持多步预测）
class LSTMModel(nn.Module):
    """LSTM模型定义，支持多步预测"""

    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, forecast_steps):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.forecast_steps = forecast_steps
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim * forecast_steps)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(DEVICE)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(DEVICE)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])  # 取最后一个时间步
        out = out.view(-1, self.forecast_steps, out.size(1) // self.forecast_steps)
        return out


# 计算评估指标
def calculate_metrics(y_true, y_pred):
    """计算MAE, RMSE, MAPE, SMAPE, WAPE, R2"""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + EPSILON))) * 100
    smape = np.mean(2.0 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + EPSILON)) * 100
    wape = np.sum(np.abs(y_true - y_pred)) / (np.sum(np.abs(y_true)) + EPSILON) * 100
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, mape, smape, wape, r2


# 可视化指标到SHP地图
def visualize_metrics_to_shp(per_town_metrics, town_list, output_dir):
    """将MAPE, SMAPE, WAPE, R2可视化为SHP地图"""
    gdf = gpd.read_file(SHP_PATH)
    print("SHP文件ID类型:", gdf['ID'].dtype)
    print("town_list前几个ID:", town_list[:5])

    gdf['ID'] = gdf['ID'].astype(str)
    town_list_str = [str(town_id) for town_id in town_list]

    gdf['MAPE'] = np.nan
    gdf['SMAPE'] = np.nan
    gdf['WAPE'] = np.nan
    gdf['R2'] = np.nan

    matched_ids = []
    unmatched_ids = []
    for town_id in town_list_str:
        if town_id in gdf['ID'].values:
            metrics = per_town_metrics.get(int(town_id) if town_id.isdigit() else town_id, {})
            gdf.loc[gdf['ID'] == town_id, 'MAPE'] = metrics.get('MAPE', np.nan)
            gdf.loc[gdf['ID'] == town_id, 'SMAPE'] = metrics.get('SMAPE', np.nan)
            gdf.loc[gdf['ID'] == town_id, 'WAPE'] = metrics.get('WAPE', np.nan)
            gdf.loc[gdf['ID'] == town_id, 'R2'] = metrics.get('R2', np.nan)
            matched_ids.append(town_id)
        else:
            unmatched_ids.append(town_id)

    print(f"匹配的ID数量: {len(matched_ids)}")
    print(f"未匹配的ID: {unmatched_ids}")
    print("MAPE列统计:", gdf['MAPE'].describe())
    print("SMAPE列统计:", gdf['SMAPE'].describe())
    print("WAPE列统计:", gdf['WAPE'].describe())
    print("R2列统计:", gdf['R2'].describe())

    gdf.to_file(os.path.join(output_dir, '上海市_with_metrics.shp'))

    metrics_to_plot = [
        ('MAPE', 'Mean Absolute Percentage Error (%)'),
        ('SMAPE', 'Symmetric Mean Absolute Percentage Error (%)'),
        ('WAPE', 'Weighted Absolute Percentage Error (%)'),
        ('R2', 'R-squared')
    ]

    for metric, title in metrics_to_plot:
        fig, ax = plt.subplots(figsize=(10, 10))
        gdf.plot(column=metric, ax=ax, legend=True, cmap=CMAP, alpha=ALPHA,
                 missing_kwds={'color': 'lightgrey', 'label': 'Missing Data'})
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('Longitude', fontsize=12)
        ax.set_ylabel('Latitude', fontsize=12)
        plt.savefig(os.path.join(output_dir, f'{metric.lower()}_map.png'))
        plt.close()


# 训练和评估模型
def train_and_evaluate():
    """训练模型并评估"""
    charging_data, town_list, town_names = load_and_preprocess_data()

    available_columns = charging_data.columns.tolist()
    town_list_str = [str(town_id) for town_id in town_list]
    matched_towns = [col for col in town_list_str if col in available_columns]
    if not matched_towns:
        raise ValueError("没有找到匹配的乡镇ID列，请检查 charging_energy_all.csv 的列名与 town_ids.csv 的ID是否一致")

    matched_indices = [town_list_str.index(col) for col in matched_towns]
    town_list = [town_list[i] for i in matched_indices]
    town_names = [town_names[i] for i in matched_indices]
    print(f"匹配的乡镇数量: {len(matched_towns)}")
    if len(matched_towns) < len(town_list_str):
        print(f"警告: 只有 {len(matched_towns)} 个乡镇ID在充电数据中找到，未匹配的乡镇ID将被忽略")

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(charging_data[matched_towns])

    total_size = len(scaled_data)
    train_size = int(total_size * TRAIN_SPLIT)
    val_size = int(total_size * VAL_SPLIT)
    train_data = scaled_data[:train_size]
    val_data = scaled_data[train_size:train_size + val_size]
    test_data = scaled_data[train_size + val_size:]

    X_train, y_train = create_sequences(train_data, SEQUENCE_LENGTH, FORECAST_STEPS)
    X_val, y_val = create_sequences(val_data, SEQUENCE_LENGTH, FORECAST_STEPS)
    X_test, y_test = create_sequences(test_data, SEQUENCE_LENGTH, FORECAST_STEPS)

    X_train = torch.FloatTensor(X_train).to(DEVICE)
    y_train = torch.FloatTensor(y_train).to(DEVICE)
    X_val = torch.FloatTensor(X_val).to(DEVICE)
    y_val = torch.FloatTensor(y_val).to(DEVICE)
    X_test = torch.FloatTensor(X_test).to(DEVICE)
    y_test = torch.FloatTensor(y_test).to(DEVICE)

    input_dim = len(matched_towns)
    output_dim = len(matched_towns)
    model = LSTMModel(input_dim, HIDDEN_DIM, NUM_LAYERS, output_dim, FORECAST_STEPS).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    start_time = time.time()
    train_losses = []
    val_losses = []
    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        output = model(X_train)
        train_loss = criterion(output, y_train)
        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_output = model(X_val)
            val_loss = criterion(val_output, y_val)

        train_losses.append(train_loss.item())
        val_losses.append(val_loss.item())

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{EPOCHS}, Train Loss: {train_loss.item():.4f}, Val Loss: {val_loss.item():.4f}")

    training_time = time.time() - start_time

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, EPOCHS + 1), train_losses, label='Training Loss', color=LOSS_CURVE_COLORS[0])
    plt.plot(range(1, EPOCHS + 1), val_losses, label='Validation Loss', color=LOSS_CURVE_COLORS[1])
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training and Validation Loss Curve')
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, 'loss_curve.png'))
    plt.close()

    model.eval()
    with torch.no_grad():
        y_pred = model(X_test).cpu().numpy()
        y_test_np = y_test.cpu().numpy()

    y_pred = scaler.inverse_transform(y_pred.reshape(-1, y_pred.shape[-1])).reshape(y_pred.shape)
    y_test_np = scaler.inverse_transform(y_test_np.reshape(-1, y_test_np.shape[-1])).reshape(y_test_np.shape)

    time_index = charging_data.index[train_size + val_size + SEQUENCE_LENGTH + FORECAST_STEPS - 1:]
    pred_columns = [f"pred_{town_id}_step{s + 1}" for town_id in town_list for s in range(FORECAST_STEPS)]
    true_columns = [f"true_{town_id}_step{s + 1}" for town_id in town_list for s in range(FORECAST_STEPS)]
    pred_df = pd.DataFrame(y_pred.reshape(y_pred.shape[0], -1), columns=pred_columns,
                           index=time_index[:y_pred.shape[0]])
    true_df = pd.DataFrame(y_test_np.reshape(y_test_np.shape[0], -1), columns=true_columns,
                           index=time_index[:y_test_np.shape[0]])
    result_df = pd.concat([true_df, pred_df], axis=1)
    result_df.to_csv(os.path.join(OUTPUT_DIR, 'predictions.csv'))

    # 计算整体指标（平均所有步长）
    overall_metrics = calculate_metrics(y_test_np.flatten(), y_pred.flatten())
    print("\nOverall Metrics (across all forecast steps):")
    print(f"MAE: {overall_metrics[0]:.4f}, RMSE: {overall_metrics[1]:.4f}, MAPE: {overall_metrics[2]:.4f}%, "
          f"SMAPE: {overall_metrics[3]:.4f}%, WAPE: {overall_metrics[4]:.4f}%, R2: {overall_metrics[5]:.4f}, "
          f"Time: {training_time:.2f}s")

    # 计算每个街道的指标（平均所有步长）
    per_town_metrics = {}
    for i, town_id in enumerate(town_list):
        town_name = town_names[i]
        town_y_test = y_test_np[:, :, i]  # 形状: (样本数, FORECAST_STEPS)
        town_y_pred = y_pred[:, :, i]
        metrics = calculate_metrics(town_y_test.flatten(), town_y_pred.flatten())
        per_town_metrics[town_id] = {
            '名称': town_name,
            'MAE': metrics[0],
            'RMSE': metrics[1],
            'MAPE': metrics[2],
            'SMAPE': metrics[3],
            'WAPE': metrics[4],
            'R2': metrics[5],
            'Time': training_time
        }
        print(f"\nTownship {town_name} (ID: {town_id}):")
        print(f"MAE: {metrics[0]:.4f}, RMSE: {metrics[1]:.4f}, MAPE: {metrics[2]:.4f}%, "
              f"SMAPE: {metrics[3]:.4f}%, WAPE: {metrics[4]:.4f}%, R2: {metrics[5]:.4f}, "
              f"Time: {training_time:.2f}s")

    metrics_df = pd.DataFrame.from_dict(per_town_metrics, orient='index')
    metrics_df.to_csv(os.path.join(OUTPUT_DIR, 'town_metrics.csv'))

    visualize_metrics_to_shp(per_town_metrics, town_list, OUTPUT_DIR)

    return overall_metrics, per_town_metrics


# 主函数
if __name__ == "__main__":
    overall_metrics, per_town_metrics = train_and_evaluate()