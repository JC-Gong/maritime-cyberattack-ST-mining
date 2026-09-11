import pandas as pd
import numpy as np
import geopandas as gpd
import libpysal
from esda import Moran, Moran_Local
from sklearn.cluster import HDBSCAN
import logging


class SpatialAnalysis:

    def __init__(self, config: dict):
        self.config = config['analysis']['spatial']
        self.cluster_config = config['analysis']['clustering']
        self.logger = logging.getLogger('MC-STM').getChild(self.__class__.__name__)

    def run(self, df: pd.DataFrame):
        gdf = self._prepare_gdf(df)
        return {
            'moran': self.analyze_moran(gdf),
            'hdbscan': self.analyze_hdbscan(gdf),
        }

    def _prepare_gdf(self, df):
        df_clean = df.dropna(subset=['Longitude', 'Latitude']).copy()
        df_clean = df_clean[df_clean['Longitude'].between(-180, 180) & df_clean['Latitude'].between(-90, 90)].copy()
        if len(df_clean) < len(df):
            self.logger.warning(f'Dropped {len(df) - len(df_clean)} records with invalid coordinates for spatial analysis.')
        gdf = gpd.GeoDataFrame(df_clean, geometry=gpd.points_from_xy(df_clean['Longitude'], df_clean['Latitude']), crs='EPSG:4326')
        return gdf

    def analyze_moran(self, gdf):
        try:
            results = []
            for year, df_year in gdf.groupby('Year'):
                if len(df_year) < 10:
                    continue
                gdf_year = gpd.GeoDataFrame(df_year, geometry=gpd.points_from_xy(df_year['Longitude'], df_year['Latitude']))
                w = libpysal.weights.KNN.from_dataframe(gdf_year, k=9)
                w.transform = 'R'
                y = np.where(df_year['Consequence severity'] == 'Major', 1, 0.5)
                mi = Moran(y, w, permutations=999)
                results.append({'Year': year, 'Moran_I': mi.I, 'p_value': mi.p_norm, 'Z-score': mi.z_norm})
            global_moran = pd.DataFrame(results).sort_values('Year') if results else pd.DataFrame()
            w = libpysal.weights.KNN.from_dataframe(gdf, k=9)
            w.transform = 'R'
            y = np.where(gdf['Consequence severity'] == 'Major', 1, 0.5)
            ml = Moran_Local(y, w, permutations=999)
            gdf['quadrant'] = ml.q
            gdf['p_value'] = ml.p_sim
            gdf['is_significant'] = gdf['p_value'] < 0.05
            label_dict = {1: 'High-High', 2: 'Low-High', 3: 'Low-Low', 4: 'High-Low'}
            gdf['cluster_type'] = gdf['quadrant'].map(label_dict)
            sig_counts = gdf[gdf['is_significant']]['cluster_type'].value_counts()
            lisa = gdf[['quadrant', 'p_value', 'is_significant', 'cluster_type']].copy()
            return {'global_moran': global_moran, 'lisa': lisa, 'significant_counts': sig_counts}
        except Exception as e:
            self.logger.error(f'Moran analysis failed: {str(e)}')
            return None

    def analyze_hdbscan(self, gdf):
        X = np.radians(np.c_[gdf.geometry.y.values, gdf.geometry.x.values])
        clusterer = HDBSCAN(min_cluster_size=self.cluster_config.get('min_cluster_size', 20), min_samples=self.cluster_config.get('min_samples', 20), metric='haversine', copy=True)
        labels = clusterer.fit_predict(X)
        gdf['cluster'] = labels
        gdf['prob'] = clusterer.probabilities_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        cluster_counts = pd.Series(labels).value_counts().sort_index()
        counts_df = pd.DataFrame({'Cluster': [f'Cluster {k + 1}' if k != -1 else 'Noise' for k in cluster_counts.index], 'Incident counts': cluster_counts.values})
        years = list(range(2010, 2026))
        unique_clusters = [l for l in sorted(set(labels)) if l != -1]
        evolution = {}
        for lab in unique_clusters:
            sel = gdf['cluster'] == lab
            counts = gdf[sel]['Year'].value_counts().reindex(years, fill_value=0).sort_index()
            evolution[f'Cluster {lab + 1}'] = counts.values
        temporal_evolution = pd.DataFrame(evolution, index=years)
        temporal_evolution.index.name = 'Year'
        return {
            'labels': labels,
            'probabilities': clusterer.probabilities_,
            'n_clusters': n_clusters,
            'cluster_counts': counts_df,
            'temporal_evolution': temporal_evolution,
        }
