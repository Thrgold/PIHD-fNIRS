"""Fig. 4: horizontally aligned panels with the seed legend outside axes."""
from pathlib import Path
import importlib.util
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location('paper_figures', Path(__file__).with_name('13_generate_paper_figures_final_order.py'))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

def save_aligned(fig, stem):
    left, right = fig.axes
    fig.set_size_inches(7.15, 2.55)
    # Identical bottom, height and width guarantee aligned data regions.
    left.set_position([0.08, 0.23, 0.34, 0.55])
    right.set_position([0.66, 0.23, 0.34, 0.55])
    for ax in fig.axes:
        ax.set_title('', loc='left')
        ax.set_title('', loc='center')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    left.legend(loc='lower left', bbox_to_anchor=(0, 1.025), ncol=3,
                frameon=False, fontsize=7.4, handlelength=1.3,
                columnspacing=0.6, borderaxespad=0)
    fig.text(0.08, 0.94, 'a  PIHD label-permutation null', fontsize=8, va='top')
    fig.text(0.66, 0.94, 'b  Subject-cluster bootstrap 95% CI', fontsize=8, va='top')
    assert left.get_position().y0 == right.get_position().y0
    assert left.get_position().y1 == right.get_position().y1
    out = ROOT / 'output' / 'pdf' / 'Fig4_permutation_validation_external_legend.pdf'
    fig.savefig(out, bbox_inches='tight', pad_inches=0.06)
    fig.savefig(ROOT / 'figures' / 'Fig4_permutation_validation_external_legend.png', dpi=300, bbox_inches='tight', pad_inches=0.06)
    plt.close(fig)
    print(out.name)

if __name__ == '__main__':
    results = {s: np.load(ROOT / 'results' / 'npy' / f'loso_permutation_9methods_seed{s}.npy', allow_pickle=True).item() for s in module.SEEDS}
    module.save_figure = save_aligned
    module.plot_permutation_validation(results)
