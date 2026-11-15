import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import os

class AcademicVisualizer:
    def __init__(self, config_path="config/config.yaml"):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)['visualization']
        
        self._setup_style()

    def _setup_style(self):
        """Set up global matplotlib style based on config."""
        plt.rcParams['font.family'] = self.config.get('font_family', 'Times New Roman')
        plt.rcParams['font.size'] = self.config.get('font_size', 16)
        plt.rcParams['axes.labelweight'] = 'bold'
        plt.rcParams['axes.titleweight'] = 'bold'
        plt.rcParams['savefig.dpi'] = self.config.get('dpi', 300)

    def apply_academic_spines(self, ax):
        """Remove top and right spines from a matplotlib axis."""
        spines = self.config.get('show_spines', [False, False, True, True])
        ax.spines['top'].set_visible(spines[0])
        ax.spines['right'].set_visible(spines[1])
        ax.spines['bottom'].set_visible(spines[2])
        ax.spines['left'].set_visible(spines[3])

    def save_plot(self, fig, filename, output_dir="output/plots", dpi=None):
        """Save the figure with configured format and DPI."""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        ext = self.config.get('format', 'png')
        path = os.path.join(output_dir, f"{filename}.{ext}")
        
        save_dpi = dpi if dpi is not None else self.config.get('dpi', 300)
        fig.savefig(path, bbox_inches='tight', dpi=save_dpi)
        plt.close(fig)
        return path

    def get_color(self):
        """Return the primary theme color."""
        return self.config.get('color_primary', 'skyblue')
