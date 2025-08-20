
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import Patch
import geopandas as gpd
from pathlib import Path
from config import Config


def plot_and_save_losses(train_losses, val_losses, title, filename, output_dir):
    """Plots and saves training and validation loss curves."""
    save_path = Path(output_dir) / 'figures' / filename
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 5))
    if train_losses: plt.plot(train_losses, label='Training Loss', color='royalblue')
    if val_losses: plt.plot(val_losses, label='Validation Loss', linestyle='--', color='darkorange')
    plt.title(title, fontsize=14)
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Loss curve saved: {save_path}")


def plot_fidelity_curves(df, output_dir):
    """Plots and saves the Fidelity vs. Sparsity curves."""
    save_path = Path(output_dir) / 'figures' / 'fidelity_sparsity_curve.png'
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 6))
    plt.plot(df["S_Actual"], df["Fidelity_Absolute"], marker='o', label='Absolute Fidelity (RMSE Change)')
    plt.xlabel("Sparsity (Fraction of Edges Remaining)")
    plt.ylabel("Absolute Fidelity (Increase in RMSE)")
    plt.title("Fidelity vs. Sparsity Curve")
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    if len(df["S_Actual"]) > 1 and df["S_Actual"].iloc[0] > df["S_Actual"].iloc[-1]:
        plt.gca().invert_xaxis()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Fidelity curve saved: {save_path}")


class GeoVisualizer:
    """Handles the geographical visualization of analysis results on a map."""

    def __init__(self, config: Config, node_ids: list):
        self.config = config
        self.node_ids_ordered = node_ids
        try:
            self.shanghai_map = gpd.read_file(config.SHANGHAI_SHP_FILE).to_crs(epsg=4326)
            if config.MAP_ID_COLUMN in self.shanghai_map.columns:
                self.shanghai_map[config.MAP_ID_COLUMN] = self.shanghai_map[config.MAP_ID_COLUMN].astype(str)
            else:
                raise ValueError(f"ID column '{config.MAP_ID_COLUMN}' not found in shapefile.")
        except Exception as e:
            self.shanghai_map = None
            print(f"Warning: Could not load shapefile. Geospatial visualizations will be disabled. Error: {e}")

    def generate_focused_area_maps_for_top_n(self, spatial_mask, top_n_ids, mask_name_prefix="spatial_influence"):
        """Generates and saves maps showing spatial influence for a list of focus nodes."""
        if self.shanghai_map is None: return
        print(f"Generating maps for top {len(top_n_ids)} nodes...")

        node_id_to_idx = {nid: i for i, nid in enumerate(self.node_ids_ordered)}
        cfg = self.config
        thr = cfg.IMPORTANCE_THRESHOLD

        norm_enh = mcolors.Normalize(vmin=1 + thr, vmax=max(2.0, spatial_mask.max()))
        norm_weak = mcolors.Normalize(vmin=min(0.0, spatial_mask.min()), vmax=1 - thr)

        for focus_id in top_n_ids:
            if focus_id not in node_id_to_idx:
                print(f"Skipping node {focus_id}: not in model's node list.")
                continue

            fig, ax = plt.subplots(1, 1, figsize=(12, 10))
            divider = make_axes_locatable(ax)
            focus_idx = node_id_to_idx[focus_id]
            conn_values = spatial_mask[:, focus_idx]

            for _, poly_row in self.shanghai_map.iterrows():
                geom_id = str(poly_row[cfg.MAP_ID_COLUMN])
                color, alpha, ec, lw = cfg.DEFAULT_NEUTRAL_COLOR_AREA, cfg.AREA_ALPHA_DEFAULT, cfg.AREA_EDGE_COLOR_DEFAULT, cfg.AREA_LINEWIDTH_DEFAULT

                if geom_id == focus_id:
                    color, alpha, ec, lw = cfg.FOCUSED_NODE_COLOR, cfg.AREA_ALPHA_FOCUSED, cfg.AREA_EDGE_COLOR_HIGHLIGHT, cfg.AREA_LINEWIDTH_HIGHLIGHT
                elif geom_id in node_id_to_idx:
                    mask_val = conn_values[node_id_to_idx[geom_id]]
                    if mask_val > 1 + thr:
                        color, alpha, ec = cfg.ENHANCED_NEIGHBOR_CMAP(
                            norm_enh(mask_val)), cfg.AREA_ALPHA_HIGHLIGHTED_NEIGHBOR, cfg.ENHANCED_NEIGHBOR_EDGE_COLOR
                    elif mask_val < 1 - thr:
                        color, alpha, ec = cfg.WEAKENED_NEIGHBOR_CMAP(
                            norm_weak(mask_val)), cfg.AREA_ALPHA_HIGHLIGHTED_NEIGHBOR, cfg.WEAKENED_NEIGHBOR_EDGE_COLOR
                else:
                    color, alpha = cfg.UNMODELED_AREA_COLOR, cfg.AREA_ALPHA_UNMODELED

                gpd.GeoSeries([poly_row.geometry]).plot(ax=ax, color=color, alpha=alpha, edgecolor=ec, linewidth=lw)

            ax.set_title(f"Spatial Influence Analysis: Sources of Influence for Node {focus_id}", fontsize=16)
            ax.axis('off')

            cax_e = divider.append_axes("right", size="5%", pad=0.1)
            cb_e = plt.colorbar(plt.cm.ScalarMappable(norm=norm_enh, cmap=cfg.ENHANCED_NEIGHBOR_CMAP), cax=cax_e)
            cb_e.set_label(f'Enhancing Influence (> {1 + thr})', fontsize=10)

            legend_elements = [
                Patch(facecolor=cfg.FOCUSED_NODE_COLOR, label=f'Focus Node: {focus_id}'),
                Patch(facecolor=cfg.DEFAULT_NEUTRAL_COLOR_AREA, label='Neutral Influence'),
                Patch(facecolor=cfg.ENHANCED_NEIGHBOR_CMAP(0.75), label='Enhancing Influence'),
                Patch(facecolor=cfg.WEAKENED_NEIGHBOR_CMAP(0.25), label='Weakening Influence'),
                Patch(facecolor=cfg.UNMODELED_AREA_COLOR, label='Unmodeled Area')
            ]
            ax.legend(handles=legend_elements, loc='lower left', fontsize=10)

            save_path = Path(cfg.OUTPUT_DIR) / 'figures' / f'{mask_name_prefix}_node_{focus_id}.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"  - Map saved for node {focus_id}: {save_path}")