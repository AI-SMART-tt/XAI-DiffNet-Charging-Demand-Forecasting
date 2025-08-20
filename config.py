# XAI-DiffNet/config.py

import torch
import numpy as np
import matplotlib.pyplot as plt

class Config:
    """
    Configuration class to centralize all hyperparameters and path settings.
    Modifying parameters here allows for easy adjustment of the entire experimental workflow.
    """
    # ------------------- Data Paths -------------------
    ADJACENCY_MATRIX_FILE = './data/data_graph/sw_fused_graph.csv'
    SOC_DATA_FILE = './data/data_timeseries/charging_demand.csv'
    SHANGHAI_SHP_FILE = './data/data_geo/shanghai_districts.shp'
    NODE_COORDS_FILE = './data/data_graph/node_coordinates.csv'
    MAP_ID_COLUMN = 'ID'

    # ------------------- Model Parameters -------------------
    SEQ_LEN = 12
    PRED_LEN = 3
    RNN_UNITS = 64
    NUM_RNN_LAYERS = 2
    MAX_DIFFUSION_STEP = 2

    # ------------------- Training Parameters -------------------
    NUM_EPOCHS_GNN = 100
    NUM_EPOCHS_MASK = 50
    BATCH_SIZE = 64
    LEARNING_RATE_GNN = 0.001
    LEARNING_RATE_MASK = 0.005

    # ------------------- Interpretability & Analysis Parameters -------------------
    IMPORTANCE_THRESHOLD = 0.1
    TOP_N_NODES_ANALYSIS = 10
    FIDELITY_SPARSITY_LEVELS = np.linspace(0.1, 0.9, 9)[::-1].tolist()
    IG_STEPS = 50  # Number of steps for Integrated Gradients

    # ------------------- Visualization Parameters -------------------
    DEFAULT_NEUTRAL_COLOR_AREA = 'gainsboro'
    UNMODELED_AREA_COLOR = 'whitesmoke'
    AREA_EDGE_COLOR_DEFAULT = 'gray'
    AREA_EDGE_COLOR_HIGHLIGHT = 'black'
    AREA_LINEWIDTH_DEFAULT = 0.5
    AREA_LINEWIDTH_HIGHLIGHT = 1.2
    AREA_ALPHA_DEFAULT = 0.6
    AREA_ALPHA_HIGHLIGHTED_NEIGHBOR = 0.6
    AREA_ALPHA_FOCUSED = 0.95
    AREA_ALPHA_UNMODELED = 0.3
    FOCUSED_NODE_COLOR = 'blue'
    ENHANCED_NEIGHBOR_CMAP = plt.cm.Reds
    WEAKENED_NEIGHBOR_CMAP = plt.cm.Greens_r
    ENHANCED_NEIGHBOR_EDGE_COLOR = 'darkred'
    WEAKENED_NEIGHBOR_EDGE_COLOR = 'darkgreen'

    # ------------------- Other Configurations -------------------
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUTPUT_DIR = './outputs'
    TRAIN_RATIO = 0.6
    VAL_RATIO = 0.2
    TEST_RATIO = 0.2