import pandas as pd
import numpy as np
import yaml
import os
import logging

class DataLoader:

    def __init__(self, config_path='config/config.yaml'):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)['data']
        self.logger = logging.getLogger(self.__class__.__name__)

    def load_data(self, file_path=None):
        """Load and preprocess maritime cyber incident data."""
        if file_path is None:
            potential_paths = [self.config['raw_path'], '../data/output.csv', 'data/output.csv']
            for p in potential_paths:
                if os.path.exists(p):
                    file_path = p
                    break
            if file_path is None:
                raise FileNotFoundError('Could not find raw data file (output.csv).')
        try:
            df = pd.read_csv(file_path)
            rename_map = {'Lat': 'Latitude', 'Lon': 'Longitude'}
            df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
            if 'Year' in df.columns:
                df = df[df['Year'] >= self.config['start_year']].copy()
            df = self._normalize_geography(df)
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].round(3)
            return df
        except Exception as e:
            self.logger.error(f'Error loading data: {str(e)}')
            raise

    def _normalize_geography(self, df):
        """Merge HK, Macau, Taiwan into China as per user preference."""
        merge_list = [c.title() for c in self.config['geographic_normalization']['merge_china']]
        target_name = self.config['geographic_normalization']['target_name']
        geo_cols = ['Country', 'Attack countries', 'Victim countries']
        for col in geo_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.title()
                mask = df[col].isin(merge_list)
                if mask.any():
                    df.loc[mask, col] = target_name
        return df

    def save_processed(self, df):
        """Save the cleaned and processed data."""
        output_path = self.config['processed_path']
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
