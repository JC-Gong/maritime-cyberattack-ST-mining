# A data-driven framework for spatiotemporal pattern mining of cyber incidents in digital maritime transport systems

<p align="center">
    <img src="https://img.shields.io/static/v1?label=Python&message=3.11&color=e83e8c"/>
</p>

Code for [A data-driven framework for spatiotemporal pattern mining of cyber incidents in digital maritime transport systems
](https://www.sciencedirect.com/science/article/pii/S0029801826052662)

## Abstract
The rapid digitalisation of maritime transport systems has increased the exposure of vessels, ports, and offshore infrastructures to cyber threats, posing growing risks to operational safety, service continuity, and system resilience. Despite growing concern over maritime cybersecurity, limited research has examined the spatial and temporal patterns of cyber incidents at a system level. This study develops a data-driven Maritime Cyberattack Spatiotemporal Mining (MC-STM) framework to explore geographical concentrations, temporal associations, and co-occurrence patterns in reported maritime cyber incidents. An enhanced Maritime Cyber Attack Database (MCAD++) is established, containing 465 reported incidents up to December 2025, with structured Risk Influential Factors (RIFs) representing different aspects of cyber incidents. The framework combines association rule mining to identify recurring RIF combinations, transfer entropy to examine directional associations between annual RIF time series, and HDBSCAN and Moran's I to investigate spatial clustering patterns. The results identify three geographical clusters of reported incidents in North America, Europe and Asia under the selected HDBSCAN configuration. Association rule mining reveals recurring combinations of RIFs involving system availability and information integrity. Transfer entropy suggests several directional associations before multiple-testing correction, but none remain statistically significant after Benjamini–Hochberg false discovery rate correction. Global spatial autocorrelation in recorded consequence severity is weak and sensitive to the neighbourhood specification. Overall, the framework provides a structured analysis of geographical concentrations and recurring risk-factor combinations in reported maritime cyber incidents. The findings offer an empirical basis for maritime cybersecurity monitoring and the assessment of cyber resilience, while highlighting the need to account for reporting bias and uncertainty when interpreting historical incident data.



## Installation
```bash
pip install -r requirements.txt
```

## Usage
Run all analysis tasks:
```bash
python -m src.main
```

Run specific tasks:
```bash
python -m src.main --tasks describe temporal
```

Custom config and data:
```bash
python -m src.main --config config/my_config.yaml --data data/custom_output.csv
```

## Directory Structure
- `config/`: Configuration parameters (YAML).
- `data/`: Placeholder for input data.
- `output/`: Results, plots, and logs.
- `src/`: Source code.
  - `utils/`: Data loader, visualizer, and logger.
  - `analysis/`: Specialized analysis modules.

### Cite MC-STM
If you find this repo useful, please consider citing paper

```
@article{li2027data,
  title={A data-driven framework for spatiotemporal pattern mining of cyber incidents in digital maritime transport systems},
  author={Li, Huanhuan and Gong, Jincheng and Zhao, Yunfeng and Yang, Zaili},
  journal={Ocean Engineering},
  volume={371},
  pages={129432},
  year={2027},
  publisher={Elsevier}
}
```