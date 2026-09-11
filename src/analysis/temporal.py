import pandas as pd
import numpy as np
import statsmodels.api as sm
import networkx as nx
import pyinform.transferentropy as te
import logging
import os


def build_te_series(df):
    columns = ['Target', 'Asset exploited', 'Cyber threats', 'Consequence severity', 'Consequence type', 'Intent', 'Origin']
    years = np.arange(2010, 2026)
    data_filtered = df[(df['Year'] >= 2010) & (df['Year'] <= 2025)].copy()
    series_dict = {}
    for col in columns:
        if col not in data_filtered.columns:
            continue
        categories = data_filtered[col].value_counts().index
        for cat in categories:
            series_dict[f'{col}-{cat}'] = data_filtered[data_filtered[col] == cat]['Year'].value_counts().sort_index().reindex(years, fill_value=0)
    ts_df = pd.DataFrame(series_dict)
    if ts_df.empty:
        return ts_df
    ts_df['All'] = data_filtered['Year'].value_counts().sort_index().reindex(years, fill_value=0)
    return ts_df


def transfer_entropy_matrix(data, k=1):
    cols = data.columns
    n = len(cols)
    te_mat = np.zeros((n, n))
    for i, xi in enumerate(cols):
        for j, yj in enumerate(cols):
            if i != j:
                x = data[xi].values.astype(int)
                y = data[yj].values.astype(int)
                if np.any(x) and np.any(y):
                    try:
                        te_mat[i, j] = te.transfer_entropy(x, y, k=k)
                    except Exception:
                        te_mat[i, j] = 0
    return pd.DataFrame(te_mat, index=cols, columns=cols)


def monte_carlo_test(data, k=1, num_permutations=500, seed=0):
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
                rng = np.random.default_rng(seed)
                for t in range(num_permutations):
                    y_perm = rng.permutation(y_vals)
                    try:
                        te_permuted[t] = te.transfer_entropy(x_vals, y_perm.astype(int), k=k)
                    except Exception:
                        te_permuted[t] = 0
                p_value_mat[i, j] = np.mean(te_permuted >= te_original)
    return pd.DataFrame(p_value_mat, index=cols, columns=cols)


def _run_mc_config(ts_df, perms, seed, k=1):
    p = monte_carlo_test(ts_df, k=k, num_permutations=perms, seed=seed)
    return (perms, seed, p)


class TemporalAnalysis:

    def __init__(self, config: dict):
        self.config = config['analysis']['temporal']
        self.logger = logging.getLogger('MC-STM').getChild(self.__class__.__name__)
        self._stable_edges = None

    def run(self, df: pd.DataFrame):
        return {
            'poisson_forecast': self.poisson_glm_forecast(df),
            'te_stability': self.validate_te_stability(df),
            'transfer_entropy': self.analyze_transfer_entropy(df),
        }

    def poisson_glm_forecast(self, df):
        df_clean = df.copy()
        df_clean['MonthDate'] = pd.to_datetime(dict(year=df_clean['Year'].astype(int), month=df_clean['Month'].astype(int), day=1), errors='coerce')
        df_clean = df_clean.dropna(subset=['MonthDate'])
        df_monthly = df_clean.groupby('MonthDate').size().rename('Count').reset_index()
        df_monthly = df_monthly.rename(columns={'MonthDate': 'Month'})
        df_monthly = df_monthly.set_index('Month').asfreq('MS', fill_value=0).reset_index()
        t0 = df_monthly['Month'].min()
        df_monthly['t'] = (df_monthly['Month'].dt.year - t0.year) * 12 + (df_monthly['Month'].dt.month - t0.month)
        X = sm.add_constant(df_monthly[['t']])
        y = df_monthly['Count'].astype(int)
        try:
            model = sm.GLM(y, X, family=sm.families.Poisson()).fit()
            df_monthly['fitted_mean'] = model.predict(X)
            forecast_len = self.config.get('forecast_months', 6)
            last_month = df_monthly['Month'].max()
            future_months = pd.date_range(last_month + pd.offsets.MonthBegin(1), periods=forecast_len, freq='MS')
            df_f = pd.DataFrame({'Month': future_months})
            df_f['t'] = (df_f['Month'].dt.year - t0.year) * 12 + (df_f['Month'].dt.month - t0.month)
            Xf = sm.add_constant(df_f[['t']])
            pred_res = model.get_prediction(Xf)
            pred_sf = pred_res.summary_frame(alpha=0.05)
            df_f['pred_mean'] = pred_sf['mean'].values
            df_f['pred_lower'] = pred_sf['mean_ci_lower'].values
            df_f['pred_upper'] = pred_sf['mean_ci_upper'].values
            return {'model': model, 'monthly': df_monthly, 'forecast': df_f}
        except Exception as e:
            self.logger.error(f'Poisson GLM failed: {str(e)}')
            return None

    def analyze_transfer_entropy(self, df):
        ts_df = build_te_series(df)
        if ts_df.empty:
            return None
        te_df = transfer_entropy_matrix(ts_df, k=1)
        num_perms = 500
        p_value_df = monte_carlo_test(ts_df, k=1, num_permutations=num_perms, seed=0)
        detailed_te = []
        for i, src in enumerate(ts_df.columns):
            for j, dest in enumerate(ts_df.columns):
                if i != j:
                    detailed_te.append({'Source': src, 'Destination': dest, 'TE': round(te_df.iloc[i, j], 3), 'p_value': round(p_value_df.iloc[i, j], 3), 'Significant': p_value_df.iloc[i, j] < 0.05})
        G = nx.DiGraph()
        node_mapping = {i + 1: name for i, name in enumerate(ts_df.columns)}
        for idx in node_mapping.keys():
            G.add_node(idx)
        for i in range(len(te_df)):
            for j in range(len(te_df)):
                weight = te_df.iloc[i, j]
                if i != j and p_value_df.iloc[i, j] < 0.05:
                    if self._stable_edges is not None and (i, j) not in self._stable_edges:
                        continue
                    G.add_edge(i + 1, j + 1, weight=weight)
        out_s = pd.Series(dict(G.out_degree(weight='weight'))).sort_index()
        in_s = pd.Series(dict(G.in_degree(weight='weight'))).sort_index()
        betweenness_s = pd.Series(nx.betweenness_centrality(G, weight='weight')).sort_index()
        return {
            'te_matrix': te_df,
            'p_values': p_value_df,
            'edge_list': pd.DataFrame(detailed_te),
            'graph': G,
            'node_mapping': node_mapping,
            'centrality': pd.DataFrame({
                'Node': out_s.index,
                'Label': [node_mapping.get(i, i) for i in out_s.index],
                'Out_strength': out_s.values,
                'In_strength': in_s.values,
                'Betweenness': betweenness_s.values,
            }),
        }

    def validate_te_stability(self, df, alpha=0.05):
        ts_df = build_te_series(df)
        if ts_df.empty:
            return None
        seeds = self.config.get('stability_seeds', [0, 1, 42, 123, 2025])
        perm_counts = self.config.get('stability_permutations', [200, 500, 1000])
        reference_perms = self.config.get('reference_permutations', 1000)
        n = len(ts_df.columns)
        off_diag = [(i, j) for i in range(n) for j in range(n) if i != j]
        ref_p = monte_carlo_test(ts_df, k=1, num_permutations=reference_perms, seed=0)
        ref_sig = {(i, j) for i, j in off_diag if ref_p.iloc[i, j] < alpha}
        configs = [(perms, seed) for perms in perm_counts for seed in seeds]
        results = {}
        try:
            from concurrent.futures import ProcessPoolExecutor
            with ProcessPoolExecutor(max_workers=min(len(configs), os.cpu_count() or 1)) as ex:
                futures = {ex.submit(_run_mc_config, ts_df, p, s): (p, s) for p, s in configs}
                for fut in futures:
                    p, s = futures[fut]
                    results[p, s] = fut.result()[2]
        except Exception as e:
            self.logger.warning(f'Parallel stability run failed ({e}); falling back to sequential.')
            results = {cfg: monte_carlo_test(ts_df, k=1, num_permutations=cfg[0], seed=cfg[1]) for cfg in configs}
        records = []
        jaccard_rows = []
        for (perms, seed), p in results.items():
            sig = {(i, j) for i, j in off_diag if p.iloc[i, j] < alpha}
            jac = len(sig & ref_sig) / len(sig | ref_sig) if sig | ref_sig else 1.0
            jaccard_rows.append({'permutations': perms, 'seed': seed, 'n_significant': len(sig), 'jaccard_vs_reference': round(jac, 4)})
            for i, j in off_diag:
                records.append({'permutations': perms, 'seed': seed, 'source': ts_df.columns[i], 'target': ts_df.columns[j], 'p_value': round(p.iloc[i, j], 4), 'significant': p.iloc[i, j] < alpha})
        detail_df = pd.DataFrame(records)
        jaccard_df = pd.DataFrame(jaccard_rows)
        stab = detail_df.groupby(['source', 'target'])['significant'].agg(['sum', 'count'])
        stab['stability'] = (stab['sum'] / stab['count']).round(4)
        stab = stab.reset_index().rename(columns={'sum': 'times_significant', 'count': 'times_tested'})
        stab = stab.sort_values(['stability', 'times_significant'], ascending=False).reset_index(drop=True)
        jac_summary = jaccard_df.groupby('permutations')['jaccard_vs_reference'].agg(['mean', 'min', 'max']).round(4).reset_index()
        self._stable_edges = set()
        for _, row in stab[stab['stability'] == 1.0].iterrows():
            si = ts_df.columns.get_loc(row['source'])
            sj = ts_df.columns.get_loc(row['target'])
            self._stable_edges.add((si, sj))
        return {
            'detail': detail_df,
            'jaccard': jaccard_df,
            'edge_stability': stab,
            'jaccard_summary': jac_summary,
            'avg_jaccard': jaccard_df['jaccard_vs_reference'].mean(),
            'stable_edges': self._stable_edges,
        }
