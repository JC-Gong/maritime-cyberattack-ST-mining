import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pyinform.transferentropy as te
from ..utils.visualizer import AcademicVisualizer
import logging
import os

class TemporalAnalysis:
    def __init__(self, visualizer: AcademicVisualizer, config: dict):
        self.viz = visualizer
        self.config = config['analysis']['temporal']
        self.logger = logging.getLogger("MC-STM").getChild(self.__class__.__name__)

    def run(self, df: pd.DataFrame):
        """Run all temporal analysis tasks."""
        self.logger.info("Starting temporal analysis...")
        
        # 2. Transfer Entropy Analysis
        self.analyze_transfer_entropy(df)
        
        # 3. Categorical Temporal Evolution
        self.plot_categorical_temporal_evolution(df)
        
        self.logger.info("Temporal analysis completed.")


    def analyze_transfer_entropy(self, df):
        """Analyze information transfer between categories following user snippet."""
        # 1. Prepare categorical time series (2010-2025 counts)
        columns = ['Target', 'Asset exploited', 'Cyber threats', 'Consequence severity', 
                   'Consequence type', 'Intent', 'Origin']
        years = np.arange(2010, 2026)
        data_filtered = df[(df['Year'] >= 2010) & (df['Year'] <= 2025)].copy()
        
        series_dict = {}
        for col in columns:
            if col not in data_filtered.columns:
                continue
            # Use all categories as per Untitled-1.py
            categories = data_filtered[col].value_counts().index
            for cat in categories:
                series_dict[f"{col}-{cat}"] = (
                    data_filtered[data_filtered[col] == cat]['Year']
                    .value_counts()
                    .sort_index()
                    .reindex(years, fill_value=0)
                )
        
        ts_df = pd.DataFrame(series_dict)
        if ts_df.empty:
            return
            
        # Include 'All' column as per Untitled-1.py
        ts_df['All'] = data_filtered['Year'].value_counts().sort_index().reindex(years, fill_value=0)

        def transfer_entropy_matrix(data, k=1):
            cols = data.columns
            n = len(cols)
            te_mat = np.zeros((n, n))
            for i, xi in enumerate(cols):
                for j, yj in enumerate(cols):
                    if i != j:
                        x = data[xi].values.astype(int)
                        y = data[yj].values.astype(int)
                        # Ensure data is not all zeros to avoid pyinform errors
                        if np.any(x) and np.any(y):
                            try:
                                te_mat[i, j] = te.transfer_entropy(x, y, k=k)
                            except:
                                te_mat[i, j] = 0
            return pd.DataFrame(te_mat, index=cols, columns=cols)

        def monte_carlo_test(data, k=1, num_permutations=500):
            cols = data.columns
            n = len(cols)
            p_value_mat = np.zeros((n, n))
            te_mat = transfer_entropy_matrix(data, k=k)
            
            for i, xi in enumerate(cols):
                for j, yj in enumerate(cols):
                    if i != j:
                        te_original = te_mat.iloc[i, j]
                        if te_original == 0:
                            p_value_mat[i, j] = 1.0
                            continue
                            
                        te_permuted = np.zeros(num_permutations)
                        x_vals = data[xi].values.astype(int)
                        y_vals = data[yj].values.astype(int)
                        for t in range(num_permutations):
                            y_perm = np.random.permutation(y_vals)
                            try:
                                te_permuted[t] = te.transfer_entropy(x_vals, y_perm.astype(int), k=k)
                            except:
                                te_permuted[t] = 0
                        p_value_mat[i, j] = np.mean(te_permuted >= te_original)
            return pd.DataFrame(p_value_mat, index=cols, columns=cols)

        self.logger.info(f"Computing Transfer Entropy matrix and Monte Carlo test ({len(ts_df.columns)} nodes)...")
        np.random.seed(0)
        te_df = transfer_entropy_matrix(ts_df, k=1)
        # Using 1000 permutations for complete reporting as per user requirement
        num_perms = 1000
        p_value_df = monte_carlo_test(ts_df, k=1, num_permutations=num_perms)
        
        # Count significant edges (p < 0.05)
        mask = ~np.eye(p_value_df.shape[0], dtype=bool)
        sig_edges = (p_value_df < 0.05).values[mask].sum()
        
        self.logger.info(f"Transfer Entropy Significance Results ({num_perms} permutations):")
        self.logger.info(f"- Significant edges (p < 0.05): {sig_edges}")
        
        # Save matrices
        te_df.round(3).to_csv(os.path.join("output", "transfer_entropy_matrix.csv"))
        p_value_df.round(3).to_csv(os.path.join("output", "transfer_entropy_p_values.csv"))
        
        # Detailed Reporting CSV
        detailed_te = []
        for i, src in enumerate(ts_df.columns):
            for j, dest in enumerate(ts_df.columns):
                if i != j:
                    detailed_te.append({
                        "Source": src,
                        "Destination": dest,
                        "TE": round(te_df.iloc[i, j], 3),
                        "p_value": round(p_value_df.iloc[i, j], 3),
                        "Significant": p_value_df.iloc[i, j] < 0.05
                    })
        pd.DataFrame(detailed_te).to_csv(os.path.join("output", "transfer_entropy_detailed_report.csv"), index=False)

        # Plot following user snippet exactly
        plt.rcParams['font.size'] = 10
        fig = plt.figure(figsize=(15, 8))
        
        im = plt.imshow(te_df, aspect='auto', cmap='viridis')
        plt.colorbar(im, label='Transfer Entropy')
        
        plt.xticks(ticks=np.arange(0, te_df.shape[0]), labels=np.arange(0, te_df.shape[0]) + 1)
        plt.yticks(ticks=np.arange(0, te_df.shape[0]), labels=np.arange(0, te_df.shape[0]) + 1)
        
        plt.xlabel('Target (To)')
        plt.ylabel('Source (From)')
        
        handles = [plt.Line2D([0], [0], color="white", marker='s', linestyle="", 
                               label=f"{i+1}: {lab}") for i, lab in enumerate(ts_df.columns)]
        
        plt.legend(handles=handles, title="Label", 
                   bbox_to_anchor=(-0.7, 1), loc="upper left")
        
        plt.tight_layout()
        self.viz.save_plot(fig, "Transfer Entropy", dpi=300)
        
        # 2. Network Visualization following user snippet exactly
        G = nx.DiGraph()
        node_mapping = {i+1: name for i, name in enumerate(ts_df.columns)}
        for idx in node_mapping.keys():
            G.add_node(idx)
            
        for i in range(len(te_df)):
            for j in range(len(te_df)):
                weight = te_df.iloc[i, j]
                # Use raw p-values for the network graph
                if i != j and p_value_df.iloc[i, j] < 0.05:
                    G.add_edge(i+1, j+1, weight=weight)
        
        if len(G.edges) > 0:
            # 2a. Network Visualization following user snippet exactly
            fig_net = plt.figure(figsize=(12, 8))
            pos = nx.circular_layout(G)
            
            # Draw edges
            nx.draw_networkx_edges(
                G, pos,
                arrowstyle="->", arrowsize=20,
                edge_color="gray",
                width=[d['weight'] for _, _, d in G.edges(data=True)]
            )
            
            # Draw nodes (with digital ID)
            nx.draw_networkx_nodes(G, pos, node_size=500, node_color="skyblue", alpha=0.7)
            nx.draw_networkx_labels(
                G, pos,
                font_size=12,
                font_family=self.viz.config['font_family'],
                font_weight='bold',
                font_color='black',
            )
            
            # Legend at bottom
            handles_net = [plt.Line2D([0], [0], color="white", marker='s', linestyle="", 
                                      label=f"{i}: {lab}") for i, lab in node_mapping.items()]
            
            plt.legend(
                handles=handles_net,
                title="Label",
                loc='upper center',
                bbox_to_anchor=(0.5, -0.0),
                ncol=3,
                fontsize=10,
                title_fontsize=12,
                frameon=True,
                borderaxespad=0.0
            )
            
            plt.axis("off")
            plt.tight_layout(rect=[0, 0, 1, 1])
            self.viz.save_plot(fig_net, "Transfer Entropy Graph", dpi=900)

            # 2b. Radar Chart following user snippet exactly
            def _series_to_radar(ax, s, title="", normalize=True):
                s = s.sort_index()
                labels = s.index.astype(str).tolist()
                values = s.values.astype(float)

                if normalize:
                    vmin, vmax = float(np.min(values)), float(np.max(values))
                    if vmax > vmin:
                        values = (values - vmin) / (vmax - vmin)
                    else:
                        values = np.zeros_like(values)
                    r_ticks = [0.25, 0.5, 0.75, 1]
                    r_ticklabels = ["0.25","0.50","0.75",""]
                    r_lim = (0, 1)
                else:
                    upper = float(np.max(values))
                    r_lim = (0, upper * 1.05 if upper > 0 else 1.0)
                    r_ticks = np.linspace(r_lim[0], r_lim[1], 5)[1:]
                    r_ticklabels = [f"{t:.2g}" for t in r_ticks]
                    r_ticklabels[-1] = ""

                N = len(labels)
                angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
                angles += angles[:1]
                vals_closed = np.concatenate([values, values[:1]])

                ax.set_theta_offset(np.pi / 2)
                ax.set_theta_direction(-1)
                ax.set_xticks(angles[:-1])
                ax.set_xticklabels(labels, fontsize=12)
                ax.set_ylim(*r_lim)
                ax.set_yticks(r_ticks)
                ax.set_yticklabels(r_ticklabels, fontsize=12)
                ax.plot(angles, vals_closed, linewidth=2)
                ax.fill(angles, vals_closed, alpha=0.10)
                ax.set_title(title, fontsize=12, pad=20, loc="center", y=-0.25)
                plt.tight_layout()

            out_strength = dict(G.out_degree(weight="weight"))
            in_strength = dict(G.in_degree(weight="weight"))
            betweenness = nx.betweenness_centrality(G, weight="weight")

            out_s = pd.Series(out_strength).sort_index()
            in_s = pd.Series(in_strength).sort_index()
            betweenness_s = pd.Series(betweenness).sort_index()

            fig_radar = plt.figure(figsize=(12, 4))
            ax1 = plt.subplot(1, 3, 1, polar=True)
            _series_to_radar(ax1, out_s, title="(a) Out-strength", normalize=False)
            ax2 = plt.subplot(1, 3, 2, polar=True)
            _series_to_radar(ax2, in_s, title="(b) In-strength", normalize=False)
            ax3 = plt.subplot(1, 3, 3, polar=True)
            _series_to_radar(ax3, betweenness_s, title="(c) Betweenness", normalize=False)
            
            plt.tight_layout()
            self.viz.save_plot(fig_radar, "Transfer Entropy Graph Radar", dpi=300)

    def plot_categorical_temporal_evolution(self, df):
        """Plot temporal evolution for categorical features following user snippet."""
        columns = ['Target', 'Asset exploited', 'Cyber threats', 'Consequence severity', 
                   'Consequence type', 'Intent', 'Origin']
        
        # Filter data for 2010-2025 as per snippet
        years = np.arange(2010, 2026)
        data = df[(df['Year'] >= 2010) & (df['Year'] <= 2025)].copy()
        
        all_year = data['Year'].value_counts().sort_index().reindex(years, fill_value=0)
        col_year = {}
        
        import matplotlib
        ncols = 3
        nrows = int(np.ceil(len(columns) / ncols))
        
        fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
        axes = axes.flatten()
        
        for idx, col in enumerate(columns):
            if col not in data.columns:
                continue
                
            ax = axes[idx]
            categories = data[col].value_counts().index
            
            # Use tab20b or fallback to tab20
            try:
                cmap = matplotlib.colormaps["tab20b"].resampled(len(categories))
            except:
                cmap = matplotlib.colormaps["tab20"].resampled(len(categories))

            for j, cat in enumerate(categories):
                series = (
                    data[data[col] == cat]['Year']
                    .value_counts()
                    .sort_index()
                    .reindex(years, fill_value=0)
                )
                col_year[col + '-' + str(cat)] = series
                
                ax.plot([str(_) for _ in years], 
                        series.values, 
                        marker='o', 
                        linewidth=1.5, 
                        color=cmap(j),
                        label=cat)
            
            ax.tick_params(axis='x', rotation=45)
            ax.set_xlabel('Year')
            ax.set_ylabel('Counts')
            # Adjust legend font size to be smaller for complex plots
            ax.legend(loc="upper left", title=col, fontsize=8, title_fontsize=9)
            self.viz.apply_academic_spines(ax)

        # Remove extra subplots
        for k in range(len(columns), len(axes)):
            fig.delaxes(axes[k])
            
        plt.tight_layout()
        self.viz.save_plot(fig, "time", dpi=1200)
        
        # Save results to CSV
        col_year['All'] = all_year
        res_df = pd.DataFrame(col_year)
        output_csv = os.path.join("output", "categorical_temporal_counts.csv")
        res_df.to_csv(output_csv)
        self.logger.info(f"Categorical temporal counts saved to {output_csv}")
