import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
import logging


class ValidationAnalysis:

    def __init__(self, config: dict):
        self.config = config
        self.logger = logging.getLogger('MC-STM').getChild(self.__class__.__name__)

    def run(self, df: pd.DataFrame):
        return self.validate_temporal_consistency(df)

    def validate_temporal_consistency(self, df: pd.DataFrame):
        split_year = 2020
        df_a = df[df['Year'] <= split_year].copy()
        df_b = df[df['Year'] > split_year].copy()
        if len(df_b) < 10:
            self.logger.warning('Insufficient data in the validation period (2021-2025) for temporal validation.')
            return None
        association = self._validate_association_consistency(df_a, df_b)
        spatial = self._validate_spatial_consistency(df_a, df_b)
        return {'association': association, 'spatial': spatial}

    def _validate_association_consistency(self, df_a, df_b):
        cols = ['Target', 'Asset exploited', 'Cyber threats', 'Consequence severity', 'Consequence type', 'Intent', 'Origin']

        def get_detailed_rules(df_sub):
            df_onehot = pd.get_dummies(df_sub[cols].astype(str), prefix_sep='-')
            frequent_itemsets = apriori(df_onehot, min_support=0.05, use_colname=True, max_len=2)
            if frequent_itemsets.empty:
                return pd.DataFrame()
            rules = association_rules(frequent_itemsets, metric='lift', min_threshold=1.0)
            if rules.empty:
                return pd.DataFrame()
            rules['rule_id'] = rules.apply(lambda r: f"{sorted(list(r['antecedents']))[0]} -> {sorted(list(r['consequents']))[0]}", axis=1)
            return rules[['rule_id', 'support', 'confidence', 'lift']]
        rules_a = get_detailed_rules(df_a)
        rules_b = get_detailed_rules(df_b)
        if rules_a.empty or rules_b.empty:
            self.logger.warning('Insufficient rules found in one of the periods for detailed validation.')
            return None
        comparison = pd.merge(rules_a, rules_b, on='rule_id', suffixes=('_A', '_B'))
        lift_corr = comparison['lift_A'].corr(comparison['lift_B'])
        persistence_rate = len(comparison) / len(rules_a)
        robust_rules = comparison[(comparison['lift_A'] > 1.2) & (comparison['lift_B'] > 1.2)]
        return {
            'comparison': comparison,
            'lift_correlation': lift_corr,
            'persistence_rate': persistence_rate,
            'robust_rules': robust_rules,
        }

    def _validate_spatial_consistency(self, df_a, df_b):
        coords_a = df_a[['Latitude', 'Longitude']].dropna()
        coords_b = df_b[['Latitude', 'Longitude']].dropna()
        center_a = coords_a.mean()
        center_b = coords_b.mean()
        dist = np.sqrt((center_a['Latitude'] - center_b['Latitude']) ** 2 + (center_a['Longitude'] - center_b['Longitude']) ** 2)
        return {
            'center_a': center_a,
            'center_b': center_b,
            'euclidean_shift': dist,
        }
