import argparse
import yaml
import os
import logging
import warnings
from .utils.logger import setup_logger
from .utils.data_loader import DataLoader
from .utils.visualizer import AcademicVisualizer
from .analysis.descriptive import DescriptiveAnalysis
from .analysis.temporal import TemporalAnalysis
from .analysis.spatial import SpatialAnalysis
from .analysis.association import AssociationAnalysis
from .analysis.comparison import ComparisonAnalysis
from .analysis.validation import ValidationAnalysis

def main():
    # Suppress specific library warnings for clean CLI output
    warnings.filterwarnings("ignore", category=UserWarning, module="libpysal")
    warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
    
    parser = argparse.ArgumentParser(description="Maritime Cybersecurity Spatio-Temporal Modeling (MC-STM)")
    parser.add_argument("--config", default="config/config.yaml", help="Path to config file")
    parser.add_argument("--data", help="Path to raw data CSV (overrides config)")
    parser.add_argument("--tasks", nargs="+", choices=["all", "describe", "temporal", "spatial", "association", "comparison", "validation"], 
                        default=["all"], help="Analysis tasks to run")
    args = parser.parse_args()

    # 1. Setup
    if not os.path.exists(args.config):
        print(f"Error: Config file not found at {args.config}")
        return

    logger = setup_logger()
    logger.info("Starting MC-STM workflow...")

    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 2. Data Loading
    loader = DataLoader(args.config)
    try:
        df = loader.load_data(args.data)
        loader.save_processed(df)
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return

    # 3. Analysis
    viz = AcademicVisualizer(args.config)
    
    tasks = args.tasks
    if "all" in tasks:
        tasks = ["describe", "temporal", "spatial", "association", "comparison", "validation"]

    if "describe" in tasks:
        DescriptiveAnalysis(viz).run(df)
    
    # Filter for advanced analyses (2010 onwards)
    df_2010 = df[df['Year'] >= 2010].copy()
    
    if "temporal" in tasks:
        TemporalAnalysis(viz, config).run(df_2010)
        
    if "spatial" in tasks:
        SpatialAnalysis(viz, config).run(df_2010)
        
    if "association" in tasks:
        AssociationAnalysis(viz, config).run(df_2010)

    if "comparison" in tasks:
        ComparisonAnalysis(viz, config).run(df_2010)

    if "validation" in tasks:
        ValidationAnalysis(viz, config).run(df_2010)

    logger.info("MC-STM workflow finished successfully.")

if __name__ == "__main__":
    main()
