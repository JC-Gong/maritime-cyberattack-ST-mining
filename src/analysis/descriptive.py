import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import numpy as np
from pyecharts.charts import Pie, Grid
from pyecharts import options as opts
from ..utils.visualizer import AcademicVisualizer
import logging

class DescriptiveAnalysis:
    def __init__(self, visualizer: AcademicVisualizer):
        self.viz = visualizer
        self.logger = logging.getLogger(self.__class__.__name__)
        self.categorical_cols = [
            'Target', 'Asset exploited', 'Cyber threats', 
            'Consequence severity', 'Consequence type', 
            'Intent', 'Origin', 'Region', 'Attack countries', 'Victim countries'
        ]

    def run(self, df: pd.DataFrame):
        """Run all descriptive analysis tasks."""
        self.logger.info("Starting descriptive analysis...")
        self.plot_yearly_counts(df)
        self.generate_interactive_pie_charts(df)
        self.logger.info("Descriptive analysis completed.")

    def plot_yearly_counts(self, df):
        """Plot the number of cyberattack incidents per year."""
        year_counts = df['Year'].value_counts().sort_index()
        min_year, max_year = year_counts.index.min(), year_counts.index.max()
        continuous_years = np.arange(min_year, max_year + 1)
        filled_series = year_counts.reindex(continuous_years).fillna(0)

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(filled_series.index, filled_series.values, 
                marker='o', linestyle='-', color='#1f77b4', 
                linewidth=2, markersize=8)
        
        ax.set_xlabel('Year')
        ax.set_ylabel('Number of cyberattack incidents')
        ax.set_xticks(continuous_years)
        plt.xticks(rotation=45)
        
        self.viz.apply_academic_spines(ax)
        self.viz.save_plot(fig, "yearly_incident_counts")

    def generate_interactive_pie_charts(self, df):
        """Generate interactive pie charts using pyecharts based on user-provided template."""
        # 1. Filter data (Year >= 2010)
        data = df[df['Year'] >= 2010].copy()
        
        # 2. Columns to plot (Total 10)
        columns = [
            'Asset exploited', 'Target', 'Consequence severity', 
            'Consequence type', 'Intent', 'Cyber threats',  
            'Origin', 'Attack countries', 'Region', 'Victim countries'
        ]
        
        # 3. Grid coordinates (5x2)
        rows, cols = 5, 2 
        pos = [] 
        x_step = 100 // (cols + 1)   # 横向间距 (约33.3) 
        y_step = 100 // (rows + 1)   # 纵向间距 (约16.6) 
        for r in range(rows): 
            for c in range(cols): 
                pos.append(( (c+1)*x_step , (r+1)*y_step )) 
        
        # 4. Initialize Grid
        grid = Grid(init_opts=opts.InitOpts(width="2000px", height="3000px", renderer="svg")) 
        
        # 5. Color setup
        import matplotlib
        cmap = matplotlib.colormaps["YlGnBu"]
            
        n_colors = 10 
        colors = [plt.cm.colors.to_hex(cmap(i/n_colors)) for i in range(n_colors)] 
        
        for i, col in enumerate(columns):
            if col not in data.columns: continue
            
            counts = data[col].value_counts()
            
            # Victim countries specific logic
            if col == 'Victim countries':
                others_mask = counts.index.str.lower() == 'others' 
                original_others = counts[others_mask].sum() 
                top_8 = counts[~others_mask].head(6) 
                rest_sum = counts[~others_mask].iloc[6:].sum() 
                final_other = original_others + rest_sum 
                
                if final_other > 0: 
                    counts = pd.concat([top_8, pd.Series({'Other': final_other})]) 
                else: 
                    counts = top_8 
            else:
                # Default grouping for other columns
                if len(counts) > 10:
                    top_9 = counts.head(9)
                    other_sum = counts.iloc[9:].sum()
                    counts = pd.concat([top_9, pd.Series({'Other': other_sum})])

            data_pair = [(str(k), int(v)) for k, v in counts.items()] 
            current_n = counts.shape[0] 
            
            pie = ( 
                Pie() 
                .add( 
                    series_name=col, 
                    data_pair=data_pair, 
                    radius=["10%", "20%"], 
                    center=[f"{pos[i][0]}%", f"{pos[i][1]}%"],
                    label_opts=opts.LabelOpts( 
                        formatter="{b}: \n{d}%", 
                        font_size=30, 
                        font_weight="bold", 
                        font_family='Times New Roman', 
                    ), 
                ) 
                .set_global_opts( 
                    legend_opts=opts.LegendOpts(is_show=False) 
                ) 
                .set_colors(colors[-current_n:]) 
            ) 
            
            # Center Label
            pie.add( 
                series_name="center", 
                data_pair=[(col, 1)], 
                radius=["0%", "0%"],  
                center=[f"{pos[i][0]}%", f"{pos[i][1]}%"], 
                label_opts=opts.LabelOpts( 
                    is_show=True, 
                    position="center", 
                    formatter=col, 
                    font_size=30, 
                    font_weight="bold", 
                    font_family="Times New Roman" 
                ), 
            ) 
        
            grid.add(pie, grid_opts=opts.GridOpts())

        output_path = "output/interactive_distributions.html"
        grid.render(output_path)
        self.logger.info(f"Interactive charts saved to {output_path}")
