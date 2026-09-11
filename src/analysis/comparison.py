import pandas as pd
import numpy as np
from sklearn.cluster import HDBSCAN, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
from mlxtend.frequent_patterns import apriori, association_rules
from scipy.stats import chi2_contingency
import logging


class ComparisonAnalysis:

    def __init__(self, config: dict):
        self.config = config
        self.logger = logging.getLogger('MC-STM').getChild(self.__class__.__name__)

    def run(self, df: pd.DataFrame):
        return {
            'clustering': self.compare_clustering(df),
            'association': self.compare_association(df),
        }

    def compare_clustering(self, df: pd.DataFrame):
        df_clean = df.dropna(subset=['Longitude', 'Latitude']).copy()
        X = np.radians(np.c_[df_clean['Latitude'].values, df_clean['Longitude'].values])
        if len(X) < 50:
            self.logger.warning('Insufficient data for clustering comparison.')
            return None

        def get_metrics(labels, name):
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            noise_ratio = np.sum(labels == -1) / len(labels)
            mask = labels != -1
            if n_clusters > 1 and np.sum(mask) > n_clusters:
                sil = silhouette_score(X[mask], labels[mask])
                db_idx = davies_bouldin_score(X[mask], labels[mask])
            else:
                sil, db_idx = (0, np.nan)
            return {'Method': name, 'Clusters': n_clusters, 'Noise Ratio': noise_ratio, 'Silhouette Score': sil, 'Davies-Bouldin Index': db_idx}
        hdb = HDBSCAN(min_cluster_size=self.config['analysis']['clustering'].get('min_cluster_size', 20), min_samples=self.config['analysis']['clustering'].get('min_samples', 20), metric='haversine')
        hdb_labels = hdb.fit_predict(X)
        dbs = DBSCAN(eps=0.08, min_samples=10, metric='haversine')
        dbs_labels = dbs.fit_predict(X)
        results = [get_metrics(hdb_labels, 'HDBSCAN (Proposed)'), get_metrics(dbs_labels, 'DBSCAN (Baseline)')]
        return pd.DataFrame(results)

    def compare_association(self, df: pd.DataFrame):
        cols = ['Target', 'Asset exploited', 'Cyber threats', 'Consequence severity', 'Consequence type', 'Intent', 'Origin']
        df_onehot = pd.get_dummies(df[cols].astype(str), prefix_sep='-')
        min_support = self.config['analysis']['association'].get('min_support', 0.1)
        frequent_itemsets = apriori(df_onehot, min_support=min_support, use_colname=True, max_len=2)
        min_threshold = self.config['analysis']['association'].get('association_comparison_baseline', 1.0)
        rules = association_rules(frequent_itemsets, metric='lift', min_threshold=min_threshold)
        min_conf = self.config['analysis']['association'].get('min_confidence', 0.5)
        min_lift = self.config['analysis']['association'].get('min_lift', 1.0)
        rules = rules[(rules['lift'] >= min_lift) & (rules['confidence'] >= min_conf)]
        n_rules = len(rules)
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
                    chi_results.append({'Pair': f'{cols[i]}-{cols[j]}', 'p-value': p})
                except Exception:
                    continue
        summary = pd.DataFrame({
            'Method': ['Apriori', 'Chi-square'],
            'Count': [n_rules, significant_pairs],
            'Metric': [f'Rules (Supp>{min_support}, Conf>{min_conf}, Lift>{min_lift})', 'Significant Pairs (p < 0.05)'],
        })
        return {'n_rules': n_rules, 'significant_pairs': significant_pairs, 'chi_results': pd.DataFrame(chi_results), 'summary': summary}
