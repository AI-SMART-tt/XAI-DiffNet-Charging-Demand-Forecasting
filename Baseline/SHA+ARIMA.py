import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings
import time
import os
import matplotlib.pyplot as plt
import geopandas as gpd

warnings.filterwarnings('ignore')

# ============================================================
# 可调参数集中定义
# ============================================================
MODEL_NAME      = "SHA"        # 可选 "SHA" / "ARIMA" / "HoltWinters"
SEQUENCE_LENGTH = 12             # 仅用于与原代码对齐
FORECAST_STEPS  = 3              # 预测未来步长
TRAIN_SPLIT     = 0.6            # 训练集比例
VAL_SPLIT       = 0.2            # 验证集比例（SHA/ARIMA/HW不使用，仅保留分割结构）
TEST_SPLIT      = 0.2            # 测试集比例
EPSILON         = 1              # MAPE 分母平滑常数
OUTPUT_DIR      = "output"       # 输出目录
SHP_PATH        = "data_shap/上海市_with_ID_merged.shp"
CMAP            = 'BuGn'
ALPHA           = 0.8

# SHA 参数
SHA_PERIOD      = 144             # 季节周期（5分钟×288=1天）

# ARIMA 参数
ARIMA_ORDER     = (2, 1, 2)      # ARIMA(p,d,q)
ARIMA_ROLLING   = False          # False=固定拟合（快，无泄露）；True=滚动重拟合（慢）

# Holt-Winters 参数
HW_SEASONAL     = 'add'          # 季节性类型：'add'（加法）或 'mul'（乘法）
HW_TREND        = 'add'          # 趋势类型：'add' / 'mul' / None
HW_PERIOD       = 288            # 季节周期（5分钟×288=1天）
HW_ROLLING      = False          # False=固定拟合（快，无泄露）；True=滚动重拟合（慢）

# ============================================================
# 创建输出目录 & 绘图字体
# ============================================================
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

plt.rcParams['font.family']     = 'Arial'
plt.rcParams['font.sans-serif'] = ['Arial']


# ============================================================
# 数据加载与预处理
# ============================================================
def load_and_preprocess_data():
    """加载并预处理数据"""
    town_ids      = pd.read_csv("data_graph/graph2/town_ids.csv", encoding='gbk')
    town_list     = town_ids['ID'].tolist()
    town_names    = town_ids['乡'].tolist()
    charging_data = pd.read_csv("data0/charging_energy_all.csv", encoding='gbk')
    charging_data['time'] = pd.to_datetime(charging_data['time'])
    charging_data.set_index('time', inplace=True)
    charging_data.fillna(0, inplace=True)
    return charging_data, town_list, town_names


# ============================================================
# 构建真实值目标数组
# ============================================================
def create_test_targets(test_data, forecast_steps):
    """
    从测试集中提取真实值。
    返回 shape: (n_samples, forecast_steps, n_towns)
    """
    y = []
    for i in range(len(test_data) - forecast_steps + 1):
        y.append(test_data[i: i + forecast_steps])
    return np.array(y)


# ============================================================
# SHA 模型（无改动）
# ============================================================
def sha_predict(train_data, test_data, forecast_steps, period=SHA_PERIOD):
    """
    Seasonal Historical Average (SHA) 预测。
    对训练集中与目标时刻相同周期位置的所有历史值取均值。
    返回 shape: (n_samples, forecast_steps, n_towns)
    """
    train_len   = len(train_data)
    n_samples   = len(test_data) - forecast_steps + 1
    n_towns     = train_data.shape[1]
    predictions = np.zeros((n_samples, forecast_steps, n_towns))

    period_means = np.zeros((period, n_towns))
    for pos in range(period):
        indices = list(range(pos, train_len, period))
        if indices:
            period_means[pos] = np.mean(train_data[indices], axis=0)
        else:
            period_means[pos] = train_data[-1]

    for t in range(n_samples):
        global_start = train_len + t
        for s in range(forecast_steps):
            pos = (global_start + s) % period
            predictions[t, s, :] = period_means[pos]

    return predictions


# ============================================================
# ARIMA 模型（修复数据泄露）
# ============================================================
def arima_predict(train_data, test_data, forecast_steps,
                  order=ARIMA_ORDER, rolling=ARIMA_ROLLING):
    """
    Classical ARIMA 预测，逐乡镇独立建模。

    rolling=False（推荐）:
        仅用训练集拟合一次，对测试集每个起始位置做 forecast，
        预测过程中完全不引入测试集真实值，无数据泄露。

    rolling=True:
        每步用截止到当前的历史数据重新 fit，将已预测步的
        测试集真实值（t-1 步）纳入历史，但预测第 t 步时
        不使用第 t 步真实值，无数据泄露。

    返回 shape: (n_samples, forecast_steps, n_towns)
    """
    n_towns     = train_data.shape[1]
    n_samples   = len(test_data) - forecast_steps + 1
    predictions = np.zeros((n_samples, forecast_steps, n_towns))

    for i in range(n_towns):
        print(f"  [ARIMA] 正在拟合乡镇 {i + 1}/{n_towns} ...")
        series = train_data[:, i].tolist()

        if not rolling:
            # ----------------------------------------------------------------
            # 非滚动模式（修复版）：
            # 训练集拟合一次后，对每个测试起始位置 t，
            # 用训练集末尾状态向前 forecast (t + forecast_steps) 步，
            # 取后 forecast_steps 步作为第 t 个样本的预测值。
            # 全程不 append 任何测试集真实值，彻底消除数据泄露。
            # ----------------------------------------------------------------
            try:
                fitted_model = ARIMA(series, order=order).fit()
            except Exception as e:
                print(f"    警告: 乡镇 {i} 初次拟合失败 ({e})，使用训练集末值填充")
                predictions[:, :, i] = series[-1]
                continue

            # 一次性 forecast 到最远预测位置：t=n_samples-1 时需要
            # forecast (n_samples - 1 + forecast_steps) 步
            max_horizon = n_samples - 1 + forecast_steps
            try:
                all_forecasts = fitted_model.forecast(steps=max_horizon)
                # all_forecasts[t : t + forecast_steps] 对应第 t 个样本
                for t in range(n_samples):
                    predictions[t, :, i] = all_forecasts[t: t + forecast_steps]
            except Exception:
                predictions[:, :, i] = series[-1]

        else:
            # ----------------------------------------------------------------
            # 滚动模式（无泄露，原逻辑正确）：
            # 预测第 t 步时，history 只包含训练集 + 测试集前 t 个真实值，
            # 不包含第 t 步本身，无数据泄露。
            # ----------------------------------------------------------------
            history = series.copy()
            for t in range(n_samples):
                try:
                    fitted   = ARIMA(history, order=order).fit()
                    forecast = fitted.forecast(steps=forecast_steps)
                    predictions[t, :, i] = forecast
                except Exception:
                    predictions[t, :, i] = history[-1]
                # 预测完第 t 步后，才将第 t 步真实值加入历史
                if t < len(test_data):
                    history.append(test_data[t, i])

    return predictions


# ============================================================
# Holt-Winters 模型（修复数据泄露）
# ============================================================
def holtwinters_predict(train_data, test_data, forecast_steps,
                        trend=HW_TREND, seasonal=HW_SEASONAL,
                        period=HW_PERIOD, rolling=HW_ROLLING):
    """
    Holt-Winters 三次指数平滑预测，逐乡镇独立建模。

    rolling=False（推荐）:
        仅用训练集拟合一次，对所有测试起始位置统一做长程 forecast，
        完全不引入测试集真实值，无数据泄露。

    rolling=True:
        每步用截止到当前的历史数据重新 fit，预测第 t 步时
        不使用第 t 步真实值，无数据泄露。

    返回 shape: (n_samples, forecast_steps, n_towns)
    """
    n_towns     = train_data.shape[1]
    n_samples   = len(test_data) - forecast_steps + 1
    predictions = np.zeros((n_samples, forecast_steps, n_towns))

    def _safe_series(arr):
        if seasonal == 'mul':
            arr = np.where(arr <= 0, 1e-6, arr)
        return arr

    for i in range(n_towns):
        print(f"  [HoltWinters] 正在拟合乡镇 {i + 1}/{n_towns} ...")
        series = _safe_series(train_data[:, i])

        if not rolling:
            # ----------------------------------------------------------------
            # 非滚动模式（修复版）：
            # 训练集拟合一次，一次性 forecast 到最远位置，
            # 按滑动窗口切片取各样本预测值，不 append 任何测试真实值。
            # ----------------------------------------------------------------
            try:
                fitted_model = ExponentialSmoothing(
                    series,
                    trend=trend,
                    seasonal=seasonal,
                    seasonal_periods=period,
                    initialization_method='estimated'
                ).fit(optimized=True)
            except Exception as e:
                print(f"    警告: 乡镇 {i} 初次拟合失败 ({e})，使用训练集末值填充")
                predictions[:, :, i] = series[-1]
                continue

            max_horizon = n_samples - 1 + forecast_steps
            try:
                all_forecasts = fitted_model.forecast(max_horizon)
                for t in range(n_samples):
                    predictions[t, :, i] = all_forecasts[t: t + forecast_steps]
            except Exception:
                predictions[:, :, i] = series[-1]

        else:
            # ----------------------------------------------------------------
            # 滚动模式（无泄露，原逻辑正确）：
            # 预测完第 t 步后才将第 t 步真实值加入历史。
            # ----------------------------------------------------------------
            history = list(series)
            for t in range(n_samples):
                try:
                    hist_arr = _safe_series(np.array(history))
                    fitted   = ExponentialSmoothing(
                        hist_arr,
                        trend=trend,
                        seasonal=seasonal,
                        seasonal_periods=period,
                        initialization_method='estimated'
                    ).fit(optimized=True)
                    predictions[t, :, i] = fitted.forecast(forecast_steps)
                except Exception:
                    predictions[t, :, i] = history[-1]
                # 预测完第 t 步后，才将第 t 步真实值加入历史
                if t < len(test_data):
                    history.append(_safe_series(np.array([test_data[t, i]]))[0])

    return predictions


# ============================================================
# 评估指标（无改动）
# ============================================================
def calculate_metrics(y_true, y_pred):
    """计算 MAE, RMSE, MAPE, SMAPE, WAPE, R2"""
    mae   = mean_absolute_error(y_true, y_pred)
    rmse  = np.sqrt(mean_squared_error(y_true, y_pred))
    mape  = np.mean(np.abs((y_true - y_pred) / (y_true + EPSILON))) * 100
    smape = np.mean(2.0 * np.abs(y_true - y_pred) /
                    (np.abs(y_true) + np.abs(y_pred) + EPSILON)) * 100
    wape  = np.sum(np.abs(y_true - y_pred)) / (np.sum(np.abs(y_true)) + EPSILON) * 100
    r2    = r2_score(y_true, y_pred)
    return mae, rmse, mape, smape, wape, r2


# ============================================================
# 地图可视化（无改动）
# ============================================================
def visualize_metrics_to_shp(per_town_metrics, town_list, output_dir):
    """将 MAPE, SMAPE, WAPE, R2 可视化为 SHP 地图"""
    gdf = gpd.read_file(SHP_PATH)
    print("SHP文件ID类型:", gdf['ID'].dtype)
    print("town_list前几个ID:", town_list[:5])

    gdf['ID'] = gdf['ID'].astype(str)
    town_list_str = [str(tid) for tid in town_list]

    for col in ['MAPE', 'SMAPE', 'WAPE', 'R2']:
        gdf[col] = np.nan

    matched_ids, unmatched_ids = [], []
    for town_id in town_list_str:
        if town_id in gdf['ID'].values:
            key     = int(town_id) if town_id.isdigit() else town_id
            metrics = per_town_metrics.get(key, {})
            for col in ['MAPE', 'SMAPE', 'WAPE', 'R2']:
                gdf.loc[gdf['ID'] == town_id, col] = metrics.get(col, np.nan)
            matched_ids.append(town_id)
        else:
            unmatched_ids.append(town_id)

    print(f"匹配的ID数量: {len(matched_ids)}")
    print(f"未匹配的ID:   {unmatched_ids}")

    gdf.to_file(os.path.join(output_dir, '上海市_with_metrics.shp'))

    metrics_to_plot = [
        ('MAPE',  'Mean Absolute Percentage Error (%)'),
        ('SMAPE', 'Symmetric Mean Absolute Percentage Error (%)'),
        ('WAPE',  'Weighted Absolute Percentage Error (%)'),
        ('R2',    'R-squared'),
    ]
    for metric, title in metrics_to_plot:
        fig, ax = plt.subplots(figsize=(10, 10))
        gdf.plot(column=metric, ax=ax, legend=True, cmap=CMAP, alpha=ALPHA,
                 missing_kwds={'color': 'lightgrey', 'label': 'Missing Data'})
        ax.set_title(f"{MODEL_NAME} - {title}", fontsize=14)
        ax.set_xlabel('Longitude', fontsize=12)
        ax.set_ylabel('Latitude',  fontsize=12)
        plt.savefig(os.path.join(output_dir, f'{metric.lower()}_map.png'))
        plt.close()


# ============================================================
# 主训练 / 评估函数（无改动）
# ============================================================
def train_and_evaluate():
    """运行 SHA / ARIMA / HoltWinters 并输出评估结果"""

    # ---------- 数据加载 ----------
    charging_data, town_list, town_names = load_and_preprocess_data()

    available_columns = charging_data.columns.tolist()
    town_list_str     = [str(tid) for tid in town_list]
    matched_towns     = [col for col in town_list_str if col in available_columns]

    if not matched_towns:
        raise ValueError(
            "没有找到匹配的乡镇ID列，请检查 charging_energy_all.csv 的列名"
            "与 town_ids.csv 的ID是否一致"
        )

    matched_indices = [town_list_str.index(col) for col in matched_towns]
    town_list  = [town_list[i]  for i in matched_indices]
    town_names = [town_names[i] for i in matched_indices]
    print(f"匹配的乡镇数量: {len(matched_towns)}")
    if len(matched_towns) < len(town_list_str):
        print(f"警告: 只有 {len(matched_towns)} 个乡镇ID在充电数据中找到，其余忽略")

    # ---------- 归一化 ----------
    scaler      = MinMaxScaler()
    scaled_data = scaler.fit_transform(charging_data[matched_towns])

    # ---------- 数据集划分 ----------
    total_size = len(scaled_data)
    train_size = int(total_size * TRAIN_SPLIT)
    val_size   = int(total_size * VAL_SPLIT)

    train_data = scaled_data[:train_size]
    test_data  = scaled_data[train_size + val_size:]

    # ---------- 构建真实值 ----------
    y_test_scaled = create_test_targets(test_data, FORECAST_STEPS)
    # shape: (n_samples, FORECAST_STEPS, n_towns)

    # ---------- 模型预测 ----------
    print(f"\n===== 使用模型: {MODEL_NAME} =====")
    start_time = time.time()

    if MODEL_NAME == "SHA":
        y_pred_scaled = sha_predict(
            train_data, test_data, FORECAST_STEPS, period=SHA_PERIOD
        )
    elif MODEL_NAME == "ARIMA":
        y_pred_scaled = arima_predict(
            train_data, test_data, FORECAST_STEPS,
            order=ARIMA_ORDER, rolling=ARIMA_ROLLING
        )
    elif MODEL_NAME == "HoltWinters":
        y_pred_scaled = holtwinters_predict(
            train_data, test_data, FORECAST_STEPS,
            trend=HW_TREND, seasonal=HW_SEASONAL,
            period=HW_PERIOD, rolling=HW_ROLLING
        )
    else:
        raise ValueError(
            f"未知的 MODEL_NAME='{MODEL_NAME}'，请选择 'SHA' / 'ARIMA' / 'HoltWinters'"
        )

    run_time = time.time() - start_time
    print(f"预测完成，耗时: {run_time:.2f}s")

    # ---------- 反归一化 ----------
    n_samples, n_steps, n_towns = y_pred_scaled.shape

    y_pred    = scaler.inverse_transform(
        y_pred_scaled.reshape(-1, n_towns)
    ).reshape(n_samples, n_steps, n_towns)

    y_test_np = scaler.inverse_transform(
        y_test_scaled.reshape(-1, n_towns)
    ).reshape(n_samples, n_steps, n_towns)

    # ---------- 保存预测结果 ----------
    time_start = train_size + val_size + FORECAST_STEPS - 1
    time_index = charging_data.index[time_start: time_start + n_samples]

    pred_columns = [f"pred_{tid}_step{s + 1}"
                    for tid in town_list for s in range(FORECAST_STEPS)]
    true_columns = [f"true_{tid}_step{s + 1}"
                    for tid in town_list for s in range(FORECAST_STEPS)]

    pred_df   = pd.DataFrame(
        y_pred.reshape(n_samples, -1), columns=pred_columns, index=time_index
    )
    true_df   = pd.DataFrame(
        y_test_np.reshape(n_samples, -1), columns=true_columns, index=time_index
    )
    result_df = pd.concat([true_df, pred_df], axis=1)
    result_df.to_csv(os.path.join(OUTPUT_DIR, 'predictions.csv'))

    # ---------- 整体指标 ----------
    overall_metrics = calculate_metrics(y_test_np.flatten(), y_pred.flatten())
    print("\nOverall Metrics (across all towns & forecast steps):")
    print(f"  MAE:   {overall_metrics[0]:.4f}")
    print(f"  RMSE:  {overall_metrics[1]:.4f}")
    print(f"  MAPE:  {overall_metrics[2]:.4f}%")
    print(f"  SMAPE: {overall_metrics[3]:.4f}%")
    print(f"  WAPE:  {overall_metrics[4]:.4f}%")
    print(f"  R2:    {overall_metrics[5]:.4f}")
    print(f"  Time:  {run_time:.2f}s")

    # ---------- 逐乡镇指标 ----------
    per_town_metrics = {}
    for i, town_id in enumerate(town_list):
        town_name   = town_names[i]
        town_y_test = y_test_np[:, :, i]
        town_y_pred = y_pred[:, :, i]
        metrics     = calculate_metrics(town_y_test.flatten(), town_y_pred.flatten())
        per_town_metrics[town_id] = {
            '名称':  town_name,
            'MAE':   metrics[0],
            'RMSE':  metrics[1],
            'MAPE':  metrics[2],
            'SMAPE': metrics[3],
            'WAPE':  metrics[4],
            'R2':    metrics[5],
            'Time':  run_time,
        }
        print(f"\nTownship {town_name} (ID: {town_id}):")
        print(f"  MAE={metrics[0]:.4f}  RMSE={metrics[1]:.4f}  "
              f"MAPE={metrics[2]:.4f}%  SMAPE={metrics[3]:.4f}%  "
              f"WAPE={metrics[4]:.4f}%  R2={metrics[5]:.4f}")

    # ---------- 保存指标表 ----------
    metrics_df = pd.DataFrame.from_dict(per_town_metrics, orient='index')
    metrics_df.to_csv(os.path.join(OUTPUT_DIR, 'town_metrics.csv'))

    # ---------- 地图可视化 ----------
    visualize_metrics_to_shp(per_town_metrics, town_list, OUTPUT_DIR)

    return overall_metrics, per_town_metrics


# ============================================================
# 入口
# ============================================================
if __name__ == "__main__":
    overall_metrics, per_town_metrics = train_and_evaluate()