import pandas as pd
import numpy as np
import logging


class DescriptiveAnalysis:

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.categorical_cols = ['Target', 'Asset exploited', 'Cyber threats', 'Consequence severity', 'Consequence type', 'Intent', 'Origin', 'Region', 'Attack countries', 'Victim countries']

    def run(self, df: pd.DataFrame):
        return {
            'yearly_counts': self.yearly_counts(df),
            'category_distribution': self.category_distribution(df),
        }

    def yearly_counts(self, df):
        year_counts = df['Year'].value_counts().sort_index()
        if year_counts.empty:
            return pd.DataFrame(columns=['Year', 'Incident counts'])
        min_year, max_year = (year_counts.index.min(), year_counts.index.max())
        continuous_years = np.arange(min_year, max_year + 1)
        filled_series = year_counts.reindex(continuous_years).fillna(0)
        year_df = filled_series.reset_index()
        year_df.columns = ['Year', 'Incident counts']
        return year_df

    def category_distribution(self, df):
        return {col: df[col].value_counts() for col in self.categorical_cols if col in df.columns}
