import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import HDBSCAN, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
from mlxtend.frequent_patterns import apriori, association_rules
from scipy.stats import chi2_contingency
from ..utils.visualizer import AcademicVisualizer
import logging
import os

class ComparisonAnalysis:
    def __init__(self, visualizer: AcademicVisualizer, config: dict):
        self.viz = visualizer
        self.config = config
        self.logger = logging.getLogger("MC-STM").getChild(self.__class__.__name__)

    def run(self, df: pd.DataFrame):
        """Run all comparison analyses."""
        self.logger.info("Starting baseline comparison analysis...")
        
        # 1. Clustering Baseline (HDBSCAN vs DBSCAN)
        self.compare_clustering(df)
        
        # 2. Association Baseline (Apriori vs Chi-Square)
        self.compare_association(df)
        
        self.logger.info("Comparison analysis completed.")

    def compare_clustering(self, df: pd.DataFrame):
        """Compare HDBSCAN with standard DBSCAN using Haversine metric."""
        # Prepare coordinates in radians [lat, lon] for Haversine
        df_clean = df.dropna(subset=['Longitude', 'Latitude']).copy()
        X = np.radians(np.c_[df_clean['Latitude'].values, df_clean['Longitude'].values])
        
        if len(X) < 50:
            self.logger.warning("Insufficient data for clustering comparison.")
            return

        # HDBSCAN (Proposed)
        hdb = HDBSCAN(
            min_cluster_size=self.config['analysis']['clustering'].get('min_cluster_size', 20),
            min_samples=self.config['analysis']['clustering'].get('min_samples', 20),
            metric='haversine'
        )
        hdb_labels = hdb.fit_predict(X)
        
        # DBSCAN (Baseline)
        # Try different eps to find a reasonable baseline (approx 500km = 0.08 rad)
        eps_rad = 0.08
        dbs = DBSCAN(eps=eps_rad, min_samples=10, metric='haversine') 
        dbs_labels = dbs.fit_predict(X)
        
        # Calculate metrics
        def get_metrics(labels, name):
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            noise_ratio = np.sum(labels == -1) / len(labels)
            
            # Silhouette only for non-noise points if there are at least 2 clusters
            mask = labels != -1
            if n_clusters > 1 and np.sum(mask) > n_clusters:
                sil = silhouette_score(X[mask], labels[mask])
                db_idx = davies_bouldin_score(X[mask], labels[mask])
            else:
                sil, db_idx = 0, np.nan
                
            return {
                "Method": name,
                "Clusters": n_clusters,
                "Noise Ratio": noise_ratio,
                "Silhouette Score": sil,
                "Davies-Bouldin Index": db_idx
            }

        results = [
            get_metrics(hdb_labels, "HDBSCAN (Proposed)"),
            get_metrics(dbs_labels, "DBSCAN (Baseline)")
        ]
        res_df = pd.DataFrame(results)
        
        # Log results
        self.logger.info("\nClustering Comparison Results:")
        self.logger.info(res_df.to_string())
        
        # Visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # HDBSCAN Plot
        ax1.scatter(X[:, 0], X[:, 1], c=hdb_labels, cmap='tab20', s=10, alpha=0.6)
        ax1.set_title(f"HDBSCAN (Clusters: {results[0]['Clusters']})", fontweight='bold')
        self.viz.apply_academic_spines(ax1)
        
        # DBSCAN Plot
        ax2.scatter(X[:, 0], X[:, 1], c=dbs_labels, cmap='tab20', s=10, alpha=0.6)
        ax2.set_title(f"DBSCAN (Clusters: {results[1]['Clusters']})", fontweight='bold')
        self.viz.apply_academic_spines(ax2)
        
        plt.tight_layout()
        self.viz.save_plot(fig, "clustering_comparison_baseline")
        
        # Save metrics to CSV
        res_df.to_csv(os.path.join("output", "clustering_comparison_metrics.csv"), index=False)

    def compare_association(self, df: pd.DataFrame):
        """Compare Apriori with Chi-square association test."""
        cols = ['Target', 'Asset exploited', 'Cyber threats', 'Consequence severity', 
                'Consequence type', 'Intent', 'Origin']
        
        # 1. Apriori (Proposed)
        df_onehot = pd.get_dummies(df[cols].astype(str), prefix_sep='-')
        min_support = self.config['analysis']['association'].get('min_support', 0.1)
        
        frequent_itemsets = apriori(df_onehot, min_support=min_support, use_colnames=True, max_len=2)
        min_threshold = self.config['analysis']['association'].get('association_comparison_baseline', 1.0)
        rules = association_rules(frequent_itemsets, metric="lift", min_threshold=min_threshold)
        
        min_conf = self.config['analysis']['association'].get('min_confidence', 0.5)
        min_lift = self.config['analysis']['association'].get('min_lift', 1.0)
        rules = rules[(rules['lift'] >= min_lift) & (rules['confidence'] >= min_conf)]
        
        n_rules = len(rules)
        
        # 2. Chi-square (Baseline)
        # Test all pairs of columns
        significant_pairs = 0
        total_pairs = 0
        chi_results = []
        
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                total_pairs += 1
                contingency = pd.crosstab(df[cols[i]], df[cols[j]])
                try:
                    chi2, p, dof, ex = chi2_contingency(contingency)
                    if p < 0.05:
                        significant_pairs += 1
                    chi_results.append({"Pair": f"{cols[i]}-{cols[j]}", "p-value": p})
                except:
                    continue
        
        # Summary
        self.logger.info(f"\nAssociation Comparison:")
        self.logger.info(f"Apriori (Proposed) found {n_rules} rules (Supp > {min_support}, Conf > {min_conf}, Lift > {min_lift}).")
        self.logger.info(f"Chi-square (Baseline) found {significant_pairs} significant column pairs (p < 0.05) out of {total_pairs} total pairs.")
        
        # Visualization: Bar Chart
        fig, ax = plt.subplots(figsize=(8, 6))
        methods = ['Apriori Rules', 'Chi-square Significant Pairs']
        counts = [n_rules, significant_pairs]
        
        sns.barplot(x=methods, y=counts, palette='viridis', ax=ax)
        ax.set_ylabel('Number of Significant Discoveries')
        ax.set_title(f'Association Analysis Method Comparison\n(Supp > {min_support}, Conf > {min_conf}, Lift > {min_lift})')
        
        # Add text labels on top of bars
        for i, v in enumerate(counts):
            ax.text(i, v + 0.1, str(v), ha='center', fontweight='bold')
            
        self.viz.apply_academic_spines(ax)
        self.viz.save_plot(fig, "association_comparison_baseline")
        
        # Save comparison summary
        summary = pd.DataFrame({
            "Method": ["Apriori", "Chi-square"],
            "Count": [n_rules, significant_pairs],
            "Metric": [f"Rules (Supp>{min_support}, Conf>{min_conf}, Lift>{min_lift})", "Significant Pairs (p < 0.05)"]
        })
        summary.to_csv(os.path.join("output", "association_comparison_summary.csv"), index=False)
