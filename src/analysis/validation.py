import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.cluster import HDBSCAN
from ..utils.visualizer import AcademicVisualizer
import logging
import os

class ValidationAnalysis:
    def __init__(self, visualizer: AcademicVisualizer, config: dict):
        self.viz = visualizer
        self.config = config
        self.logger = logging.getLogger("MC-STM").getChild(self.__class__.__name__)

    def run(self, df: pd.DataFrame):
        """Run temporal and hold-out validation."""
        self.logger.info("Starting validation analysis...")
        
        # 1. Temporal Hold-out Validation (2010-2020 vs 2021-2025)
        self.validate_temporal_consistency(df)
        
        self.logger.info("Validation analysis completed.")

    def validate_temporal_consistency(self, df: pd.DataFrame):
        """Compare patterns identified in 2010-2020 with 2021-2025."""
        # Split point
        split_year = 2020
        df_a = df[df['Year'] <= split_year].copy()
        df_b = df[df['Year'] > split_year].copy()
        
        if len(df_b) < 10:
            self.logger.warning("Insufficient data in the validation period (2021-2025) for temporal validation.")
            return

        self.logger.info(f"Temporal Split: Period A (2010-{split_year}, N={len(df_a)}) | Period B ({split_year+1}-2025, N={len(df_b)})")
        
        # 1. Association Rule Consistency
        self._validate_association_consistency(df_a, df_b)
        
        # 2. Spatial Cluster Consistency
        self._validate_spatial_consistency(df_a, df_b)

    def _validate_association_consistency(self, df_a, df_b):
        """Check if top association rules persist across periods using metrics comparison."""
        cols = ['Target', 'Asset exploited', 'Cyber threats', 'Consequence severity', 
                'Consequence type', 'Intent', 'Origin']
        
        def get_detailed_rules(df_sub):
            df_onehot = pd.get_dummies(df_sub[cols].astype(str), prefix_sep='-')
            frequent_itemsets = apriori(df_onehot, min_support=0.05, use_colnames=True, max_len=2)
            if frequent_itemsets.empty: return pd.DataFrame()
            rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
            if rules.empty: return pd.DataFrame()
            
            rules['rule_id'] = rules.apply(lambda r: f"{sorted(list(r['antecedents']))[0]} -> {sorted(list(r['consequents']))[0]}", axis=1)
            return rules[['rule_id', 'support', 'confidence', 'lift']]

        rules_a = get_detailed_rules(df_a)
        rules_b = get_detailed_rules(df_b)
        
        if rules_a.empty or rules_b.empty:
            self.logger.warning("Insufficient rules found in one of the periods for detailed validation.")
            return

        # Merge rules to see how metrics change
        comparison = pd.merge(rules_a, rules_b, on='rule_id', suffixes=('_A', '_B'))
        
        # Stability Analysis: Correlation of Lift between periods
        lift_corr = comparison['lift_A'].corr(comparison['lift_B'])
        persistence_rate = len(comparison) / len(rules_a)
        
        self.logger.info(f"Association Rule Validation (Apriori Persistence):")
        self.logger.info(f"- Rules found in Period A: {len(rules_a)}")
        self.logger.info(f"- Rules from A also found in B: {len(comparison)} ({persistence_rate:.2%})")
        self.logger.info(f"- Correlation of Lift values: {lift_corr:.3f}")
        
        # Identify "Robust Rules" (Lift > 1.2 in both periods)
        robust_rules = comparison[(comparison['lift_A'] > 1.2) & (comparison['lift_B'] > 1.2)]
        self.logger.info(f"- Number of highly robust rules (Lift > 1.2 in both): {len(robust_rules)}")
        
        # Save validation results
        comparison.round(3).to_csv(os.path.join("output", "association_temporal_validation.csv"), index=False)
        
        # Visualization: Lift Comparison Scatter
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(comparison['lift_A'], comparison['lift_B'], alpha=0.6, color='#1f77b4')
        
        # Add diagonal line
        lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),
            np.max([ax.get_xlim(), ax.get_ylim()]),
        ]
        ax.plot(lims, lims, 'k--', alpha=0.5, zorder=0)
        
        ax.set_xlabel("Lift (Period A: 2010-2020)")
        ax.set_ylabel("Lift (Period B: 2021-2025)")
        ax.set_title(f"Association Rule Stability (Corr: {lift_corr:.3f})")
        self.viz.apply_academic_spines(ax)
        self.viz.save_plot(fig, "association_validation_scatter", dpi=300)

    def _validate_spatial_consistency(self, df_a, df_b):
        """Compare spatial density distribution between periods."""
        # Using simple mean center comparison for simplicity in validation
        coords_a = df_a[['Latitude', 'Longitude']].dropna()
        coords_b = df_b[['Latitude', 'Longitude']].dropna()
        
        center_a = coords_a.mean()
        center_b = coords_b.mean()
        
        dist = np.sqrt((center_a['Latitude'] - center_b['Latitude'])**2 + 
                       (center_a['Longitude'] - center_b['Longitude'])**2)
        
        self.logger.info(f"Spatial Validation:")
        self.logger.info(f"- Mean Center Period A: Lat {center_a['Latitude']:.2f}, Lon {center_a['Longitude']:.2f}")
        self.logger.info(f"- Mean Center Period B: Lat {center_b['Latitude']:.2f}, Lon {center_b['Longitude']:.2f}")
        self.logger.info(f"- Euclidean Shift: {dist:.2f} degrees")
        
        # Plot distribution comparison
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(coords_a['Longitude'], coords_a['Latitude'], alpha=0.3, label='2010-2020', color='blue', s=10)
        ax.scatter(coords_b['Longitude'], coords_b['Latitude'], alpha=0.6, label='2021-2025', color='#D81B60', s=15, edgecolors='white')
        
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title("Spatial Distribution Comparison (Validation)")
        ax.legend()
        self.viz.apply_academic_spines(ax)
        self.viz.save_plot(fig, "spatial_temporal_validation", dpi=300)
