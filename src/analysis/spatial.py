import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
from shapely.geometry import Point
import libpysal
from esda import Moran, Moran_Local
from scipy.stats import gaussian_kde
from sklearn.cluster import HDBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
from ..utils.visualizer import AcademicVisualizer
import logging
import os

class SpatialAnalysis:
    def __init__(self, visualizer: AcademicVisualizer, config: dict):
        self.viz = visualizer
        self.config = config['analysis']['spatial']
        self.cluster_config = config['analysis']['clustering']
        self.logger = logging.getLogger("MC-STM").getChild(self.__class__.__name__)
        # Shapefile path following Untitled-1.py structure
        self.shape_path = "../shape/ne_110m_admin_0_countries.shp"

    def run(self, df: pd.DataFrame):
        """Run all spatial analysis tasks."""
        self.logger.info("Starting spatial analysis...")
        
        gdf = self._prepare_gdf(df)
        world = self._load_world()
        
        # 0. Global Point Map (Space.png)
        self.plot_global_map(gdf, world)
        
        # 1. Moran's I Analysis
        self.analyze_moran(gdf, world)
        
        # 2. Moran's I Sensitivity to K
        self.analyze_moran_k_sensitivity(gdf)
        
        # 3. HDBSCAN Clustering (HDBSCAN.png, trees, parameter analysis, temporal)
        self.analyze_hdbscan(gdf, world)
        
        # 4. HDBSCAN Sensitivity & Quality Analysis
        self.analyze_hdbscan_sensitivity(gdf)
        
        # 5. Spatial Feature Maps (Space_Feature.png)
        self.plot_feature_maps(gdf, world)

        # 6. Sensitivity Analysis for Severity Encoding
        self.analyze_severity_sensitivity(gdf)
        
        self.logger.info("Spatial analysis completed.")

    def _prepare_gdf(self, df):
        # Filter for valid coordinates as per Untitled-1.py
        df_clean = df.dropna(subset=['Longitude', 'Latitude']).copy()
        df_clean = df_clean[
            (df_clean['Longitude'].between(-180, 180)) & 
            (df_clean['Latitude'].between(-90, 90))
        ].copy()
        
        if len(df_clean) < len(df):
            self.logger.warning(f"Dropped {len(df) - len(df_clean)} records with invalid coordinates for spatial analysis.")

        gdf = gpd.GeoDataFrame(
            df_clean, 
            geometry=gpd.points_from_xy(df_clean['Longitude'], df_clean['Latitude']),
            crs="EPSG:4326"
        )
        return gdf

    def _load_world(self):
        if os.path.exists(self.shape_path):
            return gpd.read_file(self.shape_path)
        else:
            self.logger.warning(f"Shapefile not found at {self.shape_path}. Using built-in lowres.")
            return gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

    def plot_global_map(self, gdf, world):
        """Generate Space.png as per Untitled-1.py."""
        fig, ax = plt.subplots(figsize=(12, 8))
        world.plot(ax=ax, color='lightgrey')
        gdf.plot(ax=ax, color='blue', markersize=5)
        ax.set_axis_off()
        plt.tight_layout()
        self.viz.save_plot(fig, "Space", dpi=300)

    def analyze_moran(self, gdf, world):
        """Global and Local Moran's I analysis following Untitled-1.py."""
        try:
            # 1. Global Moran's I by year
            results = []
            for year, df_year in gdf.groupby("Year"):
                if len(df_year) < 10: continue
                
                gdf_year = gpd.GeoDataFrame(
                    df_year,
                    geometry=gpd.points_from_xy(df_year['Longitude'], df_year['Latitude'])
                )
                
                # Use k=9 as per Untitled-1.py
                w = libpysal.weights.KNN.from_dataframe(gdf_year, k=9)
                w.transform = 'R'
                
                y = np.where(df_year['Consequence severity'] == 'Major', 1, 0.5)
                # Explicitly set permutations=999 to match academic description
                mi = Moran(y, w, permutations=999)
                results.append({
                    "Year": year, 
                    "Moran_I": mi.I, 
                    "p_value": mi.p_norm,
                    "Z-score": mi.z_norm
                })
            
            if results:
                df_mi = pd.DataFrame(results).sort_values("Year")
                self.logger.info("\nGlobal Moran's I Yearly Significance Results:")
                self.logger.info(df_mi.to_string(index=False))
                
                # Save Global Moran Results to CSV
                df_mi.to_csv(os.path.join("output", "global_moran_yearly.csv"), index=False)
                
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.plot(df_mi["Year"], df_mi["Moran_I"], marker="o", label="Moran's I")
                ax.axhline(0, color="grey", linestyle="--")
                
                # Styling as per Untitled-1.py
                years = df_mi["Year"].unique()
                if len(years) > 0:
                    ax.set_xticks(range(int(min(years)), int(max(years)) + 1))
                
                ax.set_xlabel("Year")
                ax.set_ylabel("Moran's I")
                ax.legend()
                self.viz.apply_academic_spines(ax)
                self.viz.save_plot(fig, "Moran-Year", dpi=300)

            # 2. Local Moran (LISA)
            w = libpysal.weights.KNN.from_dataframe(gdf, k=9)
            w.transform = 'R'
            y = np.where(gdf['Consequence severity'] == 'Major', 1, 0.5)
            # Explicitly set permutations=999 to match academic description
            ml = Moran_Local(y, w, permutations=999)
            
            gdf['quadrant'] = ml.q
            gdf['p_value'] = ml.p_sim
            gdf['is_significant'] = gdf['p_value'] < 0.05
            
            label_dict = {
                1: 'High-High',
                2: 'Low-High',
                3: 'Low-Low',
                4: 'High-Low'
            }
            gdf['cluster_type'] = gdf['quadrant'].map(label_dict)
            
            # Log results
            sig_counts = gdf[gdf['is_significant']]['cluster_type'].value_counts()
            self.logger.info("\nLocal Moran's I (LISA) Significant Cluster Counts (p < 0.05):")
            if not sig_counts.empty:
                self.logger.info(sig_counts.to_string())
            else:
                self.logger.info("No significant clusters found.")
            
            # Save Local Moran Results
            lisa_results = pd.DataFrame(gdf.drop(columns='geometry'))
            lisa_results.to_csv(os.path.join("output", "local_moran_lisa.csv"), index=False)
            
            fig, ax = plt.subplots(figsize=(10, 8))
            world.plot(ax=ax, color='#F9F9F9', edgecolor='#CCCCCC', linewidth=0.3)
            
            colors = {
                'High-High': '#D81B60', 
                'Low-Low': '#1E88E5',  
                'High-Low': '#FFB300', 
                'Low-High': '#43A047', 
                'Not Significant': '#757575' 
            }
            
            # 1. Plot Not Significant points
            non_sig = ~gdf['is_significant']
            if non_sig.any():
                gdf[non_sig].plot(ax=ax, color=colors['Not Significant'], markersize=10, 
                                   label='Not Significant', alpha=0.6)
            
            # 2. Plot Significant Clusters
            gdf_sig = gdf[gdf['is_significant']]
            for cluster, color in colors.items():
                if cluster == 'Not Significant': continue
                subset = gdf_sig[gdf_sig['cluster_type'] == cluster]
                if not subset.empty:
                    subset.plot(ax=ax, color=color, markersize=25, label=cluster,
                                edgecolors='white', linewidths=0.5)
            
            ax.legend(title="LISA Clusters", loc='lower left', frameon=True, shadow=True, 
                      fontsize=10, title_fontsize=10)
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            # Academic spines usually not applied to maps, but keeping visualizer consistency if needed
            # For map, we usually keep axes but can hide them if preferred. Untitled-1.py keeps them.
            self.viz.save_plot(fig, "Moran", dpi=300)
            
        except Exception as e:
            self.logger.error(f"Moran analysis failed: {str(e)}")

    def analyze_hdbscan(self, gdf, world):
        """Comprehensive HDBSCAN analysis using Haversine metric."""
        # 1. Clustering
        # Haversine metric requires input in radians [lat, lon]
        X = np.radians(np.c_[gdf.geometry.y.values, gdf.geometry.x.values])
        
        clusterer = HDBSCAN(
            min_cluster_size=self.cluster_config.get('min_cluster_size', 20),
            min_samples=self.cluster_config.get('min_samples', 20),
            metric='haversine',
            copy=True
        )
        labels = clusterer.fit_predict(X)
        gdf['cluster'] = labels
        gdf['prob'] = clusterer.probabilities_
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        self.logger.info(f"HDBSCAN found {n_clusters} clusters.")
        
        # 2. HDBSCAN.png
        fig, ax = plt.subplots(figsize=(12, 8))
        # Use a very light background for higher contrast
        world.plot(ax=ax, color='#F9F9F9', edgecolor='#CCCCCC', linewidth=0.5)
        
        unique_labels = sorted(set(labels))
        # Use Dark2 palette which provides distinct academic colors
        cmap = plt.cm.Dark2
        
        for idx, lab in enumerate(unique_labels):
            sel = gdf['cluster'] == lab
            if lab == -1:
                # Noise: Darker grey for better contrast
                color = '#78909C'
                label = 'Noise'
                size = 12
                alpha = 0.5
            else:
                # Clusters: Vivid colors from Dark2
                color = cmap(idx % 8)
                label = f'Cluster {lab+1}'
                size = 35
                alpha = 0.9
                
            ax.scatter(gdf.loc[sel, 'geometry'].x, gdf.loc[sel, 'geometry'].y,
                       s=size, color=color, label=label, alpha=alpha,
                       edgecolors='white' if lab != -1 else 'none', linewidths=0.8)
        
        handles, labels_ = ax.get_legend_handles_labels()
        # Ensure noise is last in legend
        if 'Noise' in labels_:
            idx_noise = labels_.index('Noise')
            h_noise = handles.pop(idx_noise)
            l_noise = labels_.pop(idx_noise)
            handles.append(h_noise)
            labels_.append(l_noise)
            
        ax.legend(handles, labels_, loc='lower left', title='HDBSCAN Clusters', 
                    fontsize=12, title_fontsize=14, frameon=True, shadow=True)
        ax.set_axis_off()
        plt.tight_layout()
        self.viz.save_plot(fig, "HDBSCAN", dpi=300)
        
        # 3. Trees and Parameter Analysis (Simplified for sklearn)
        # Note: sklearn's tree objects don't have built-in .plot()
        self._plot_hdbscan_parameter_analysis(X)
        self._plot_cluster_temporal_evolution(gdf)

    def _plot_hdbscan_trees(self, clusterer):
        """Generate trees - Skipped for sklearn compatibility."""
        pass

    def _plot_hdbscan_parameter_analysis(self, X):
        """Generate hdbscan_parameter_analysis.png using Haversine metric."""
        sizes_to_test = range(10, 100, 10)
        noise_ratios = []
        num_clusters = []
        
        for size in sizes_to_test:
            temp = HDBSCAN(min_cluster_size=size, metric='haversine', copy=True)
            temp.fit(X)
            noise_ratios.append(np.sum(temp.labels_ == -1) / len(X))
            num_clusters.append(temp.labels_.max() + 1 if temp.labels_.max() >= 0 else 0)
            
        fig, ax1 = plt.subplots(figsize=(12, 6))
        c2 = '#1E88E5'
        ax1.set_xlabel('Min cluster size', fontsize=15)
        ax1.set_ylabel('Noise Ratio', color=c2, fontsize=15)
        ax1.plot(sizes_to_test, noise_ratios, color=c2, linestyle='--', marker='s', linewidth=2, label='Noise Ratio')
        ax1.tick_params(axis='y', labelcolor=c2, labelsize=15)
        ax1.set_ylim(-0.05, 1.05)
        ax1.grid(alpha=0.2)
        
        ax2 = ax1.twinx()
        c3 = '#8E24AA'
        ax2.set_ylabel('Number of Clusters', color=c3, fontsize=15)
        ax2.plot(sizes_to_test, num_clusters, color=c3, linestyle=':', marker='^', linewidth=2, label='Cluster Count')
        ax2.tick_params(axis='y', labelcolor=c3, labelsize=15)
        
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=15)
        
        fig.tight_layout()
        self.viz.save_plot(fig, "hdbscan_parameter_analysis", dpi=300)

    def _plot_cluster_temporal_evolution(self, gdf):
        """Generate DBSCAN-Time.png."""
        unique_labels = sorted(gdf['cluster'].unique())
        # Filter out noise for the subplots but can include it if needed
        unique_clusters = [l for l in unique_labels if l != -1]
        n_clusters = len(unique_clusters)
        
        if n_clusters == 0: return
        
        ncols = 2
        nrows = int(np.ceil(n_clusters / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3*nrows))
        axes = axes.flatten()
        
        colors = plt.cm.Accent(np.linspace(0, 1, n_clusters))
        years = range(2010, 2026)
        
        for idx, lab in enumerate(unique_clusters):
            ax = axes[idx]
            sel = gdf['cluster'] == lab
            counts = gdf[sel]['Year'].value_counts().reindex(years, fill_value=0).sort_index()
            
            ax.plot(counts.index, counts.values, marker='o', linewidth=2, color=colors[idx], 
                    alpha=0.8, label=f'Cluster {lab+1}')
            ax.set_xlabel("Year")
            ax.set_ylabel("Counts")
            ax.set_xticks(years)
            ax.set_xticklabels(years, rotation=45)
            ax.legend()
            
        for k in range(n_clusters, len(axes)):
            fig.delaxes(axes[k])
            
        plt.tight_layout()
        self.viz.save_plot(fig, "DBSCAN-Time", dpi=300)

    def plot_feature_maps(self, gdf, world):
        """Generate Space_Feature.png as per Untitled-1.py."""
        features = ['Target', 'Asset exploited', 'Cyber threats', 'Consequence severity', 
                    'Intent', 'Origin', 'Consequence type']
        
        rows, cols = 4, 2
        fig, axes = plt.subplots(rows, cols, figsize=(9, 13))
        axes = axes.flatten()
        
        for i, feat in enumerate(features):
            if feat not in gdf.columns: continue
            ax = axes[i]
            
            # Prepare categories: Top-8, others as Others
            counts = gdf[feat].value_counts(dropna=False)
            top_cats = counts.index[:8].tolist()
            
            gdf_plot = gdf.copy()
            gdf_plot['_cat'] = gdf_plot[feat].apply(lambda x: x if x in top_cats else 'Others')
            
            # Colors: High-contrast Dark2 for top categories, avoiding red
            cmap = plt.cm.Dark2
            
            # Very light grey for world map to make points pop
            world.plot(ax=ax, color='#F5F5F5', edgecolor='#CCCCCC', linewidth=0.2)
            
            for j, c in enumerate(top_cats):
                sel = gdf_plot['_cat'] == c
                # Use Dark2 colors (8 distinct colors)
                color = cmap(j % 8)
                ax.scatter(gdf_plot.loc[sel, 'geometry'].x, gdf_plot.loc[sel, 'geometry'].y,
                           s=12, color=color, label=str(c), alpha=0.9, 
                           edgecolors='white', linewidths=0.3)
            
            # Others: Distinct but low-key
            sel_others = gdf_plot['_cat'] == 'Others'
            if sel_others.any():
                ax.scatter(gdf_plot.loc[sel_others, 'geometry'].x, gdf_plot.loc[sel_others, 'geometry'].y,
                           s=8, color='#AAAAAA', label='Others', alpha=0.4)
            
            ax.set_axis_off()
            
            # Academic-style legend
            handles, labels = ax.get_legend_handles_labels()
            ncol = 2
            ax.legend(
                handles, labels,
                title=f"{feat}",
                loc='upper center',
                bbox_to_anchor=(0.5, 0.05),
                ncol=ncol,
                fontsize=8,
                title_fontsize=10,
                frameon=True,
                shadow=False,
                edgecolor='none'
            )

        for k in range(len(features), len(axes)):
            fig.delaxes(axes[k])
            
        plt.tight_layout()
        self.viz.save_plot(fig, "Space_Feature", dpi=900)

    def analyze_moran_k_sensitivity(self, gdf):
        """Analyze sensitivity of Moran's I to the choice of K in spatial weight matrix."""
        self.logger.info("Starting Moran's I K-sensitivity analysis...")
        k_values = [4, 6, 8, 9, 12, 16]
        results = []
        
        y = np.where(gdf['Consequence severity'] == 'Major', 1, 0.5)
        
        for k in k_values:
            try:
                w = libpysal.weights.KNN.from_dataframe(gdf, k=k)
                w.transform = 'R'
                mi = Moran(y, w, permutations=999)
                results.append({
                    "K": k,
                    "Moran_I": mi.I,
                    "p_value": mi.p_sim,
                    "Z-score": mi.z_sim
                })
            except Exception as e:
                self.logger.warning(f"Failed to compute Moran's I for k={k}: {e}")
        
        if results:
            res_df = pd.DataFrame(results)
            self.logger.info("\nMoran's I K-Sensitivity Results:")
            self.logger.info(res_df.to_string(index=False))
            res_df.round(3).to_csv(os.path.join("output", "moran_k_sensitivity.csv"), index=False)
            
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(res_df["K"], res_df["Moran_I"], marker='o', color='#D81B60', linewidth=2)
            ax.set_xlabel("Number of Neighbors (K)")
            ax.set_ylabel("Global Moran's I")
            ax.set_title("Sensitivity of Moran's I to Spatial Weight (K)")
            self.viz.apply_academic_spines(ax)
            self.viz.save_plot(fig, "moran_k_sensitivity", dpi=300)

    def analyze_hdbscan_sensitivity(self, gdf):
        """Analyze HDBSCAN sensitivity to parameters and report quality measures."""
        self.logger.info("Starting HDBSCAN sensitivity and quality analysis...")
        
        # 1. Parameter Sensitivity (min_cluster_size vs min_samples)
        X = np.radians(np.c_[gdf.geometry.y.values, gdf.geometry.x.values])
        
        cluster_sizes = [10, 20, 30, 40, 50]
        sample_sizes = [5, 10, 20, 30]
        
        sensitivity_results = []
        
        for mcs in cluster_sizes:
            for ms in sample_sizes:
                clusterer = HDBSCAN(min_cluster_size=mcs, min_samples=ms, metric='haversine')
                labels = clusterer.fit_predict(X)
                
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                noise_ratio = np.sum(labels == -1) / len(labels)
                
                # Quality Metrics (Silhouette and Davies-Bouldin)
                mask = labels != -1
                if n_clusters > 1 and np.sum(mask) > n_clusters:
                    sil = silhouette_score(X[mask], labels[mask])
                    db_idx = davies_bouldin_score(X[mask], labels[mask])
                else:
                    sil, db_idx = 0, np.nan
                
                sensitivity_results.append({
                    "min_cluster_size": mcs,
                    "min_samples": ms,
                    "n_clusters": n_clusters,
                    "noise_ratio": noise_ratio,
                    "silhouette_score": sil,
                    "davies_bouldin_index": db_idx
                })
        
        res_df = pd.DataFrame(sensitivity_results)
        res_df = res_df.round(3)
        res_df.to_csv(os.path.join("output", "hdbscan_sensitivity_metrics.csv"), index=False)
        
        # Plot Heatmap of Noise Ratio
        pivot_noise = res_df.pivot(index="min_cluster_size", columns="min_samples", values="noise_ratio")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(pivot_noise, annot=True, cmap="YlGnBu", ax=ax)
        ax.set_title("HDBSCAN Noise Ratio Sensitivity")
        self.viz.save_plot(fig, "hdbscan_sensitivity_noise", dpi=300)
        
        # Plot Heatmap of Cluster Count
        pivot_clusters = res_df.pivot(index="min_cluster_size", columns="min_samples", values="n_clusters")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(pivot_clusters, annot=True, cmap="OrRd", ax=ax)
        ax.set_title("HDBSCAN Cluster Count Sensitivity")
        self.viz.save_plot(fig, "hdbscan_sensitivity_clusters", dpi=300)

        # Plot Heatmap of Silhouette Score
        pivot_sil = res_df.pivot(index="min_cluster_size", columns="min_samples", values="silhouette_score")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(pivot_sil, annot=True, cmap="YlOrBr", fmt=".3f", ax=ax)
        ax.set_title("HDBSCAN Silhouette Score Sensitivity")
        self.viz.save_plot(fig, "hdbscan_sensitivity_silhouette", dpi=300)

        # Plot Heatmap of Davies-Bouldin Index
        pivot_db = res_df.pivot(index="min_cluster_size", columns="min_samples", values="davies_bouldin_index")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(pivot_db, annot=True, cmap="YlOrBr_r", fmt=".3f", ax=ax) # _r because lower is better
        ax.set_title("HDBSCAN Davies-Bouldin Index Sensitivity")
        self.viz.save_plot(fig, "hdbscan_sensitivity_davies_bouldin", dpi=300)

        # 2. Distance Metric Sensitivity
        metrics = ['haversine', 'euclidean']
        dist_results = []
        for m in metrics:
            # For euclidean, use raw coordinates
            data = X if m == 'haversine' else np.c_[gdf.geometry.y.values, gdf.geometry.x.values]
            clusterer = HDBSCAN(min_cluster_size=20, min_samples=20, metric=m)
            labels = clusterer.fit_predict(data)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            noise_ratio = np.sum(labels == -1) / len(labels)
            dist_results.append({"Metric": m, "Clusters": n_clusters, "Noise Ratio": noise_ratio})
        
        dist_df = pd.DataFrame(dist_results)
        self.logger.info("\nHDBSCAN Distance Metric Sensitivity:")
        self.logger.info(dist_df.to_string(index=False))

    def analyze_severity_sensitivity(self, gdf):
        """Perform sensitivity analysis on severity encoding for Moran's I."""
        self.logger.info("Starting severity encoding sensitivity analysis...")
        
        # Define encoding schemes (Major, Minor)
        schemes = {
            "Original (1.0, 0.5)": (1.0, 0.5),
            "Binary (1.0, 0.0)": (1.0, 0.0),
            "High Contrast (1.0, 0.1)": (1.0, 0.1),
            "Ordinal (2.0, 1.0)": (2.0, 1.0)
        }
        
        results = []
        
        # Use a fixed spatial weight matrix for consistency
        try:
            w = libpysal.weights.KNN.from_dataframe(gdf, k=9)
            w.transform = 'R'
            
            for name, (major_val, minor_val) in schemes.items():
                y = np.where(gdf['Consequence severity'] == 'Major', major_val, minor_val)
                mi = Moran(y, w)
                results.append({
                    "Scheme": name,
                    "Moran_I": mi.I,
                    "p_value": mi.p_norm,
                    "Z-score": mi.z_norm
                })
            
            res_df = pd.DataFrame(results)
            self.logger.info("\nSeverity Encoding Sensitivity Results:")
            self.logger.info(res_df.to_string())
            
            # Save results
            res_df.round(3).to_csv(os.path.join("output", "severity_sensitivity_metrics.csv"), index=False)
            
            # Visualization
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x="Scheme", y="Moran_I", data=res_df, palette="Blues_d", ax=ax)
            
            # Add Z-score labels on top
            for i, row in res_df.iterrows():
                ax.text(i, row['Moran_I'] + 0.01, f"Z={row['Z-score']:.2f}", ha='center', fontweight='bold')
            
            ax.set_title("Sensitivity Analysis: Moran's I vs. Severity Encoding", fontweight='bold')
            ax.set_ylabel("Global Moran's I")
            self.viz.apply_academic_spines(ax)
            self.viz.save_plot(fig, "severity_sensitivity_analysis")
            
        except Exception as e:
            self.logger.error(f"Sensitivity analysis failed: {str(e)}")
