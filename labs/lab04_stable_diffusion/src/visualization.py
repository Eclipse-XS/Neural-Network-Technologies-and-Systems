import math
from io import BytesIO
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .config import LAB_ROOT
from .experiments import experiment_groups
from .generation import cached_result


def preview_bytes(path, max_width=960):
    """Compact notebook preview; original PNG remains lossless on disk."""
    with Image.open(path) as image:
        image = image.convert('RGB')
        image.thumbnail((max_width, max_width), Image.Resampling.LANCZOS)
        buffer = BytesIO()
        image.save(buffer, format='JPEG', quality=85, optimize=True)
        return buffer.getvalue()


def comparison_grid(records, titles, destination, columns=3, root=LAB_ROOT):
    rows = math.ceil(len(records)/columns)
    fig, axes = plt.subplots(rows, columns, figsize=(4*columns, 4.35*rows), squeeze=False)
    for ax in axes.flat:
        ax.axis('off')
    for ax, record, title in zip(axes.flat, records, titles):
        with Image.open(root / record['image_path']) as im:
            ax.imshow(im)
        ax.set_title(title, fontsize=12)
    fig.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=120, bbox_inches='tight')
    plt.close(fig)
    return destination


def build_figures(context, root=LAB_ROOT):
    groups = experiment_groups()
    figures = {}
    for name, specs in groups.items():
        if name == 'baseline':
            continue
        records = [cached_result(s, context, root) for s in specs]
        if name == 'seed': titles = [f'Seed = {s.seed}' for s in specs]
        elif name == 'steps': titles = [f'Steps = {s.num_inference_steps}' for s in specs]
        elif name == 'cfg': titles = [f'CFG = {s.guidance_scale}' for s in specs]
        elif name == 'styles': titles = [s.style for s in specs]
        else: titles = [f'Seed {s.seed}: '+('with negative' if s.negative_prompt else 'without negative') for s in specs]
        figures[name] = comparison_grid(records, titles, root/'outputs/figures'/f'{name}_comparison.png',
                                        2 if len(specs) in (2, 4) else 3, root)
    records = [cached_result(s, context, root) for s in groups['steps']]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([r['num_inference_steps'] for r in records], [r['generation_time_seconds'] for r in records], 'o-')
    ax.set(xlabel='Inference steps', ylabel='Wall-clock inference time (s)', xticks=[20, 30, 50])
    ax.grid(alpha=.25)
    fig.tight_layout()
    figures['runtime'] = root/'outputs/figures/runtime_by_steps.png'
    fig.savefig(figures['runtime'], dpi=120)
    plt.close(fig)
    return figures
