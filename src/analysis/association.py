import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules
import networkx as nx
from ..utils.visualizer import AcademicVisualizer
import logging
import os

class AssociationAnalysis:
    def __init__(self, visualizer: AcademicVisualizer, config: dict):
        self.viz = visualizer
        self.config = config['analysis']['association']
        self.logger = logging.getLogger("MC-STM").getChild(self.__class__.__name__)

    def run(self, df: pd.DataFrame):
        """Run association rule mining and visualization."""
        self.logger.info("Starting association analysis...")
        
        # 1. Prepare data (One-hot encoding)
        cols = ['Target', 'Asset exploited', 'Cyber threats', 'Consequence severity', 
                'Consequence type', 'Intent', 'Origin']
        transactions = df[cols].astype(str)
        df_onehot = pd.get_dummies(transactions, prefix_sep='-')
        
        # 2. Apriori Sensitivity Analysis
        self.analyze_apriori_sensitivity(df_onehot)
        
        # 3. Standard Apriori Run
        try:
            frequent_itemsets = apriori(
                df_onehot, 
                min_support=self.config.get('min_support', 0.1), 
                use_colnames=True, 
                max_len=2
            )
            
            if frequent_itemsets.empty:
                self.logger.warning("No frequent itemsets found. Adjust min_support.")
                return

            rules = association_rules(
                frequent_itemsets, 
                metric="lift", 
                min_threshold=self.config.get('association_comparison_baseline', 1.0)
            )
            # Apply thresholds from config
            min_conf = self.config.get('min_confidence', 0.5)
            min_lift = self.config.get('min_lift', 1.0)
            rules = rules[(rules['lift'] >= min_lift) & (rules['confidence'] >= min_conf)]
            
            if rules.empty:
                self.logger.warning("No association rules found. Adjust thresholds.")
                return

            # Format rules for display/save
            rules['antecedents'] = rules['antecedents'].apply(lambda x: list(x)[0] if len(x)==1 else ','.join(list(x)))
            rules['consequents'] = rules['consequents'].apply(lambda x: list(x)[0] if len(x)==1 else ','.join(list(x)))
            
            # Round metrics (Untitled-1.py uses 2, but we follow user preference 3)
            rules['support'] = rules['support'].round(3)
            rules['confidence'] = rules['confidence'].round(3)
            rules['lift'] = rules['lift'].round(3)
            
            # Save CSV (sorted by lift)
            rules_path = os.path.join("output", "apriori.csv")
            rules[['antecedents','consequents','support','confidence','lift']].sort_values(by='lift', ascending=False).to_csv(rules_path, index=False)
            self.logger.info(f"Saved {len(rules)} rules to {rules_path}")

            # Console Outputs (Following Untitled-1.py)
            print("\n频繁项集 (Top 10 by Support):")
            print(frequent_itemsets.sort_values(by='support', ascending=False).head(10))
            
            print("\n关联规则 (Top 10 by Support):")
            print(rules[['antecedents','consequents','support','confidence','lift']].sort_values(by='support', ascending=False).head(10))

            # 3. Build Graph and Node Mapping once (Following Untitled-1.py)
            G = nx.DiGraph()
            node_mapping = {}
            node_counter = 1
            for _, row in rules.iterrows():
                ant, con = row['antecedents'], row['consequents']
                if ant not in node_mapping:
                    node_mapping[ant] = node_counter
                    node_counter += 1
                if con not in node_mapping:
                    node_mapping[con] = node_counter
                    node_counter += 1
                G.add_edge(node_mapping[ant], node_mapping[con], weight=row['confidence'])

            # 3.1 Define RIF colors once for consistent use in all plots
            rif_categories = ['Target', 'Asset exploited', 'Cyber threats', 'Consequence severity', 
                              'Consequence type', 'Intent', 'Origin']
            custom_colors = ['#1f77b4', '#2ca02c', '#bcbd22', '#9467bd', '#ff7f0e', '#17becf', '#e377c2']
            rif_color_map = dict(zip(rif_categories, custom_colors))
            
            # Map node IDs to colors
            node_colors_dict = {}
            rev_map = {v: k for k, v in node_mapping.items()}
            for node_id in G.nodes():
                original_label = rev_map[node_id]
                category = original_label.split('-')[0]
                node_colors_dict[node_id] = rif_color_map.get(category, "skyblue")

            # 4. Network Analysis & Visualization
            self.plot_rules_network(rules, G, node_mapping, node_colors_dict, rif_color_map)
            out_s, in_s, bet_s_weighted = self.plot_centrality_radar(G, node_colors_dict)
            
            # 5. Console Outputs & Table (Following Untitled-1.py)
            # Note: Untitled-1.py uses unweighted betweenness for the table but weighted for radar
            bet_s_unweighted = pd.Series(nx.betweenness_centrality(G)).sort_index()
            
            print("\n节点中心性 (Node Centrality - Unweighted Betweenness):")
            node_data = {
                "Node": list(node_mapping.keys()),
                "Out": [out_s.get(i, 0) for i in range(1, node_counter)],
                "In": [in_s.get(i, 0) for i in range(1, node_counter)],
                "Betweenness": [bet_s_unweighted.get(i, 0) for i in range(1, node_counter)]
            }
            print(pd.DataFrame(node_data))

            # 6. Log/Print Edge Betweenness Centrality
            self._log_edge_centrality(G, node_mapping)
            
        except Exception as e:
            self.logger.error(f"Association analysis failed: {str(e)}")

    def analyze_apriori_sensitivity(self, df_onehot):
        """Analyze sensitivity of Apriori results to support and confidence thresholds."""
        self.logger.info("Starting Apriori sensitivity analysis...")
        
        support_range = [0.05, 0.1, 0.15, 0.2]
        confidence_range = [0.3, 0.4, 0.5, 0.6, 0.7]
        
        results = []
        for supp in support_range:
            frequent_itemsets = apriori(df_onehot, min_support=supp, use_colnames=True, max_len=2)
            if frequent_itemsets.empty:
                for conf in confidence_range:
                    results.append({"Support": supp, "Confidence": conf, "RuleCount": 0, "AvgLift": 0})
                continue
                
            for conf in confidence_range:
                try:
                    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=conf)
                    results.append({
                        "Support": supp,
                        "Confidence": conf,
                        "RuleCount": len(rules),
                        "AvgLift": rules['lift'].mean() if not rules.empty else 0
                    })
                except:
                    results.append({"Support": supp, "Confidence": conf, "RuleCount": 0, "AvgLift": 0})
        
        res_df = pd.DataFrame(results)
        res_df.round(3).to_csv(os.path.join("output", "apriori_sensitivity_metrics.csv"), index=False)
        
        # Plot Sensitivity
        fig, ax = plt.subplots(figsize=(10, 6))
        for supp in support_range:
            subset = res_df[res_df["Support"] == supp]
            ax.plot(subset["Confidence"], subset["RuleCount"], marker='o', label=f"Support={supp}")
        
        ax.set_xlabel("Minimum Confidence")
        ax.set_ylabel("Number of Rules Found")
        ax.set_title("Sensitivity of Apriori Rules to Thresholds")
        ax.legend()
        self.viz.apply_academic_spines(ax)
        self.viz.save_plot(fig, "apriori_sensitivity_analysis", dpi=300)
        
        self.logger.info(f"Apriori sensitivity analysis completed. Results saved to output/apriori_sensitivity_metrics.csv")

    def plot_rules_network(self, rules, G, node_mapping, node_colors_dict, rif_color_map):
        """Visualize association rules as a directed network with RIF category color-coding."""
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111)
        # Use circular_layout as per user preference
        pos = nx.circular_layout(G)
        
        # 1. Map nodes to colors
        node_colors = [node_colors_dict[node] for node in G.nodes()]
        
        # 2. Draw edges with confidence-based width
        nx.draw_networkx_edges(
            G, pos,
            arrowstyle="->", arrowsize=20,
            edge_color="gray",
            width=[d['weight'] * 2.5 for _, _, d in G.edges(data=True)], # Scaled for visibility
            alpha=0.6,
            ax=ax
        )
        
        # 4. Draw nodes with category colors and bold numbers
        nx.draw_networkx_nodes(G, pos, node_size=900, node_color=node_colors, alpha=0.9, 
                               edgecolors='black', linewidths=1.5, ax=ax)
        
        # 5. Draw labels (Node IDs)
        nx.draw_networkx_labels(G, pos, labels={n: n for n in G.nodes()},
                                font_size=12, font_weight='bold',
                                font_family=self.viz.config['font_family'], ax=ax)
        
        ax.set_axis_off()
        
        # 6. Dual Legends (Academic style: bottom multi-column)
        # Legend A: RIF Categories
        rif_handles = [plt.Line2D([0], [0], color="white", marker='o', markerfacecolor=color, 
                                  markeredgecolor='black', markersize=10, linestyle="", label=cat) 
                       for cat, color in rif_color_map.items()]
        
        # Legend B: Node Mapping
        mapping_handles = [plt.Line2D([0], [0], color="white", marker='s', markerfacecolor='lightgray', 
                                      linestyle="", label=f"{i+1}: {lab.split('-', 1)[-1]}") 
                           for i, lab in enumerate(node_mapping.keys())]
        
        # Combine legends or place them carefully at the bottom
        # First legend for RIF
        leg1 = ax.legend(
            handles=rif_handles,
            title="RIF Categories",
            loc='upper center',
            bbox_to_anchor=(0.5, -0.02),
            ncol=4,
            fontsize=10,
            title_fontsize=11,
            frameon=True,
            shadow=True
        )
        ax.add_artist(leg1)
        
        # Second legend for Node Mapping (Multi-column at bottom)
        ax.legend(
            handles=mapping_handles,
            title="Node Mapping",
            loc='upper center',
            bbox_to_anchor=(0.5, -0.12),
            ncol=3,
            fontsize=9,
            title_fontsize=11,
            frameon=True,
            shadow=True
        )
        
        plt.tight_layout(rect=[0, 0.1, 1, 1]) # Adjust for bottom legends
        self.viz.save_plot(fig, "association_rules_network", dpi=1200)

    def plot_centrality_radar(self, G, node_colors_dict):
        """Radar plot for node centrality with RIF-based colors."""
        out_s = pd.Series(dict(G.out_degree(weight="weight"))).sort_index()
        in_s = pd.Series(dict(G.in_degree(weight="weight"))).sort_index()
        bet_s = pd.Series(nx.betweenness_centrality(G, weight="weight")).sort_index()
        
        def _series_to_radar(ax, s, title="", normalize=False):
            s = s.sort_index()
            labels = s.index.astype(str).tolist()
            values = s.values.astype(float)
            
            # Use specific color for each bar/point in radar if possible
            # Standard radar plot in matplotlib usually uses one color for the polygon.
            # To show different RIF categories, we will color the x-axis tick labels
            # and potentially the markers if we plot them individually.
            
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
            
            # Plot the main polygon in blue (as per user preference)
            angles_closed = angles + angles[:1]
            vals_closed = np.concatenate([values, values[:1]])
            primary_blue = '#1f77b4'
            ax.plot(angles_closed, vals_closed, linewidth=1.5, color=primary_blue, alpha=0.8)
            ax.fill(angles_closed, vals_closed, alpha=0.15, color=primary_blue)

            # Plot individual points with RIF colors
            for i in range(N):
                node_id = int(labels[i])
                color = node_colors_dict.get(node_id, 'skyblue')
                ax.plot(angles[i], values[i], marker='o', markersize=8, color=color, markeredgecolor='black')

            ax.set_theta_offset(np.pi / 2)
            ax.set_theta_direction(-1)
            ax.set_xticks(angles)
            ax.set_xticklabels(labels, fontsize=12)
            
            # Color the tick labels to match RIF
            for label, angle in zip(ax.get_xticklabels(), angles):
                node_id = int(label.get_text())
                label.set_color(node_colors_dict.get(node_id, 'black'))
                label.set_fontweight('bold')

            ax.set_ylim(*r_lim)
            ax.set_yticks(r_ticks)
            ax.set_yticklabels(r_ticklabels, fontsize=12)
            ax.set_title(title, fontsize=12, pad=30, loc="center", y=-0.25)

        fig = plt.figure(figsize=(15, 6))
        ax1 = plt.subplot(1, 3, 1, polar=True)
        _series_to_radar(ax1, out_s, title="(a) Out-strength", normalize=False)
        ax2 = plt.subplot(1, 3, 2, polar=True)
        _series_to_radar(ax2, in_s, title="(b) In-strength", normalize=False)
        ax3 = plt.subplot(1, 3, 3, polar=True)
        _series_to_radar(ax3, bet_s, title="(c) Betweenness", normalize=False)
        
        plt.tight_layout()
        self.viz.save_plot(fig, "association_centrality_radar", dpi=300)
        return out_s, in_s, bet_s

    def _log_edge_centrality(self, G, node_mapping):
        """Calculate and log top 20 edge betweenness centrality."""
        ebc = nx.edge_betweenness_centrality(G, weight='weight')
        sorted_ebc = sorted(ebc.items(), key=lambda x: x[1], reverse=True)[:20]
        
        print("\n边中介中心性 (Edge Betweenness Centrality):")
        rev_map = {v: k for k, v in node_mapping.items()}
        for (u, v), val in sorted_ebc:
            u_label = rev_map[u]
            v_label = rev_map[v]
            print(f"边 {u}→{v} ({u_label} → {v_label}): {val:.3f}")
            self.logger.info(f"Edge {u}->{v} ({u_label} -> {v_label}): {val:.3f}")
