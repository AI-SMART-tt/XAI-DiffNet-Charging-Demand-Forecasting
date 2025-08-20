# XAI-DiffNet: A Trustworthy AI Framework for EV Charging Demand Forecasting
—— Multi-Scale Explainable Graph Learning for Electric Vehicle Charging Demand Prediction in Urban Energy Systems

<img width="944" height="587" alt="1" src="https://github.com/user-attachments/assets/c66be68b-8d2c-45e6-92cb-29d12cb39555" />

---

**XAI-DiffNet** is a deep learning framework designed to deliver **trustworthy, accurate, and multi-scale interpretable forecasts** for electric vehicle (EV) charging demand in urban energy systems. The framework is composed of two core, intrinsically linked modules:

1.  **Forecasting Module**: A novel **Dual-Channel Diffusion Graph Recurrent Network (DC-DGRN)** that achieves state-of-the-art accuracy by capturing complex spatiotemporal dependencies. It operates on a sophisticated graph that models real-world urban dynamics.
2.  **Interpretability Module**: A **Multi-Scale Interpretation Generator** that opens the "black box" of the deep learning model. It provides hierarchical explanations to reveal *why* a prediction was made, building trust and providing actionable insights for grid operators and urban planners.

This system is specifically designed to address the trust deficit in AI for critical infrastructure, moving beyond simple prediction accuracy to provide transparent, verifiable, and decision-focused intelligence.

---
## Model Details

### 1\. Forecasting Module

The forecasting engine is built upon a novel graph learning architecture that excels at modeling the complex, non-Euclidean relationships in an urban charging network.

  * **Multi-Semantic Fused Small-World Network**: We move beyond simplistic, proximity-based graphs. Our model constructs a graph that integrates multiple layers of urban semantics:
      * Physical Adjacency
      * Geographic Distance
      * Functional Similarity (from Points of Interest)
      * Mobility Patterns (from Origin-Destination flows)
      * Historical Demand Correlation
  * **Dual-Channel Diffusion Graph Recurrent Network (DC-DGRN)**: This is a sophisticated encoder-decoder architecture that captures spatiotemporal patterns. Its dual-channel design allows it to simultaneously model and disentangle different types of spatial influence (e.g., local geographic vs. long-range semantic).

### 2\. Multi-Scale Interpretation Generator

The interpretability module makes the model's decisions transparent by providing hierarchical explanations, which are crucial for building operator trust.

  * **Microscopic Level**: Identifies the specific historical time steps and spatial connections (neighboring regions) that were most influential for a given prediction.
  * **Mesoscopic Level**: Quantifies how dependent a region's forecast is on its local neighborhood, revealing areas with robust, self-reliant patterns versus those that are highly coupled.
  * **Macroscopic Level**: Assesses the systemic importance of each region to the entire network's forecast accuracy, identifying critical hubs and systemic vulnerabilities.

---
## Project Structure

```
XAI-DiffNet/
├── main.py                     # Main script to run the entire workflow
├── config.py                   # Central configuration file for all parameters
├── model.py                    # Contains the PyTorch model definitions (DC-DGRN, etc.)
├── data_loader.py              # Handles data loading, preprocessing, and splitting
├── trainer.py                  # Manages the model training and evaluation loops
├── interpreter.py              # Runs the multi-scale interpretability analysis
├── visualization.py            # Generates all plots, including maps and charts
├── data/
│   ├── data_graph/             # Graph-related data
│   │   └── sw_fused_graph.csv  # Example of a fused multi-semantic graph
│   ├── data_timeseries/
│   │   └── charging_demand.csv # Example time-series charging demand data
│   └── data_geo/
│       └── shanghai_districts.shp # Geospatial data for map visualizations
├── outputs/
│   ├── checkpoints/            # Saved model weights (e.g., best_model.pth)
│   ├── figures/                # All generated figures and visualizations
│   │   ├── microscopic/        # Micro-level interpretation maps
│   │   ├── mesoscopic/         # Meso-level dependency maps
│   │   └── macroscopic/        # Macro-level systemic importance maps
│   ├── logs/                   # Log files for training and analysis
│   └── results/                # CSV files with performance metrics and interpretations
└── requirements.txt            # Required Python packages
```

---
## Environment Requirements

### Installing Dependencies

```bash
pip install -r requirements.txt
```

Or manually install the following core dependencies:

```bash
pip install torch numpy pandas scikit-learn matplotlib geopandas seaborn
```

---
## Usage Instructions

The entire workflow can be run and configured via the `main.py` and `config.py` files.

### 1\. Data Preparation

1.  Place data files in the appropriate subdirectories within the `data/` folder.
2.  Update the file paths in `config.py` to point to your dataset.
      * `ADJACENCY_MATRIX_FILE`: Path to the fused graph CSV.
      * `SOC_DATA_FILE`: Path to the time-series demand data CSV.
      * `SHANGHAI_SHP_FILE`: Path to the shapefile for map visualizations.

### 2\. Configuration

Adjust key parameters in `config.py` to customize the experiment:

  * **Model Parameters**: `SEQ_LEN`, `PRED_LEN`, `RNN_UNITS`, `NUM_RNN_LAYERS`.
  * **Training Parameters**: `NUM_EPOCHS_GNN`, `BATCH_SIZE`, `LEARNING_RATE_GNN`.
  * **Interpretability Parameters**: `NUM_EPOCHS_MASK`, `LEARNING_RATE_MASK`.

### 3\. Running the Workflow

Execute the main script from your terminal. The script will automatically handle training, evaluation, and interpretability analysis based on the settings in `config.py`.

```bash
python main.py
```

The script will perform the following steps:

1.  **Data Loading & Preprocessing**: Loads and prepares the graph and time-series data.
2.  **Model Training**: Trains the DC-DGRN forecasting model and saves the best-performing weights in `outputs/checkpoints/`.
3.  **Model Evaluation**: Evaluates the trained model on the test set and saves performance metrics.
4.  **Interpretability Mask Training**: Trains the learnable perturbation masks to explain the model's behavior.
5.  **Multi-Scale Analysis & Visualization**: Generates and saves all interpretation results and figures to the `outputs/` directory.

---
## Baseline Models Reference

1. Graph Neural Network (GNN) Based Baseline Models
  * **Reference Link:** [https://github.com/AIcharon-stt/Traffic-prediction-models-GNN](https://github.com/AIcharon-stt/Traffic-prediction-models-GNN)

---
## Output Results

### Forecasting Outputs

  * **Performance Metrics**:
      * `outputs/results/metrics_original.csv`: Performance metrics (MAE, RMSE, etc.) of the base model.
      * `outputs/results/metrics_interpreted.csv`: Performance after applying the interpretation module.
  * **Loss Curves**:
      * `outputs/figures/gnn_loss_curve.png`: Training and validation loss for the forecasting model.

### Interpretability Outputs

  * **Mask Statistics**:
      * `outputs/results/mask_statistics.csv`: Statistical summary of the learned spatial and temporal masks.
  * **Interpretation Maps**:
      * `outputs/figures/microscopic/`: Maps showing influential neighbors for specific regions.
      * `outputs/figures/mesoscopic/dependency_map.png`: A city-wide map showing the local dependency of each region.
      * `outputs/figures/macroscopic/systemic_importance_map.png`: A city-wide map highlighting the most systemically critical hubs.

---

## Citation

If you use the XAI-DiffNet model or ideas from this research in your work, please cite our paper:


```
To be added
```

For any questions, please contact us at [ttshi3514@163.com] or [1765309248@qq.com].
