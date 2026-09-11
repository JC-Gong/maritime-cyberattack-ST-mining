import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
import networkx as nx
import logging


class AssociationAnalysis:

    def __init__(self, config: dict):
        self.config = config['analysis']['association']
        self.logger = logging.getLogger('MC-STM').getChild(self.__class__.__name__)

    def run(self, df: pd.DataFrame):
        cols = ['Target', 'Asset exploited', 'Cyber threats', 'Consequence severity', 'Consequence type', 'Intent', 'Origin']
        transactions = df[cols].astype(str)
        df_onehot = pd.get_dummies(transactions, prefix_sep='-')
        try:
            frequent_itemsets = apriori(df_onehot, min_support=self.config.get('min_support', 0.1), use_colnames=True, max_len=2)
            if frequent_itemsets.empty:
                self.logger.warning('No frequent itemsets found. Adjust min_support.')
                return None
            rules = association_rules(frequent_itemsets, metric='lift', min_threshold=self.config.get('association_comparison_baseline', 1.0))
            min_conf = self.config.get('min_confidence', 0.5)
            min_lift = self.config.get('min_lift', 1.0)
            rules = rules[(rules['lift'] >= min_lift) & (rules['confidence'] >= min_conf)]
            if rules.empty:
                self.logger.warning('No association rules found. Adjust thresholds.')
                return None
            rules['antecedents'] = rules['antecedents'].apply(lambda x: list(x)[0] if len(x) == 1 else ','.join(list(x)))
            rules['consequents'] = rules['consequents'].apply(lambda x: list(x)[0] if len(x) == 1 else ','.join(list(x)))
            rules['support'] = rules['support'].round(3)
            rules['confidence'] = rules['confidence'].round(3)
            rules['lift'] = rules['lift'].round(3)
            G = nx.DiGraph()
            node_mapping = {}
            node_counter = 1
            for _, row in rules.iterrows():
                ant, con = (row['antecedents'], row['consequents'])
                if ant not in node_mapping:
                    node_mapping[ant] = node_counter
                    node_counter += 1
                if con not in node_mapping:
                    node_mapping[con] = node_counter
                    node_counter += 1
                G.add_edge(node_mapping[ant], node_mapping[con], weight=row['confidence'])
            out_s = pd.Series(dict(G.out_degree(weight='weight'))).sort_index()
            in_s = pd.Series(dict(G.in_degree(weight='weight'))).sort_index()
            betweenness_s = pd.Series(nx.betweenness_centrality(G, weight='weight')).sort_index()
            edge_betweenness = nx.edge_betweenness_centrality(G, weight='weight')
            top_edges = sorted(edge_betweenness.items(), key=lambda x: x[1], reverse=True)[:20]
            return {
                'rules': rules,
                'frequent_itemsets': frequent_itemsets,
                'graph': G,
                'node_mapping': node_mapping,
                'centrality': pd.DataFrame({
                    'Node': out_s.index,
                    'Label': [node_mapping.get(i, i) for i in out_s.index],
                    'Out_strength': out_s.values,
                    'In_strength': in_s.values,
                    'Betweenness': betweenness_s.values,
                }),
                'top_edge_betweenness': top_edges,
            }
        except Exception as e:
            self.logger.error(f'Association analysis failed: {str(e)}')
            return None
