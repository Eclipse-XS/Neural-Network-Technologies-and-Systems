from dataclasses import asdict
import importlib.metadata
import platform
import shutil
import time
from .config import LAB_ROOT, ModelConfig


def environment():
    import torch
    import psutil
    packages = ['torch', 'diffusers', 'transformers', 'accelerate', 'safetensors',
                'Pillow', 'numpy', 'pandas', 'matplotlib', 'nbformat', 'nbclient', 'ipykernel', 'pytest']
    versions = {p: importlib.metadata.version(p) for p in packages}
    cuda = torch.cuda.is_available()
    return {'python_version': platform.python_version(), 'versions': versions,
            'torch_version': versions['torch'], 'diffusers_version': versions['diffusers'],
            'cuda_version': torch.version.cuda, 'cuda_available': cuda,
            'gpu_name': torch.cuda.get_device_name(0) if cuda else None,
            'gpu_total_bytes': torch.cuda.get_device_properties(0).total_memory if cuda else None,
            'gpu_free_bytes': torch.cuda.mem_get_info()[0] if cuda else None,
            'ram_total_bytes': psutil.virtual_memory().total,
            'ram_available_bytes': psutil.virtual_memory().available,
            'disk_free_bytes': shutil.disk_usage(LAB_ROOT).free}


def load_pipeline(config=ModelConfig()):
    import torch
    from diffusers import StableDiffusionPipeline
    env = environment()
    cuda = env['cuda_available']
    dtype = torch.float16 if cuda else torch.float32
    start = time.perf_counter()
    pipe = StableDiffusionPipeline.from_pretrained(
        config.model_id, revision=config.revision, variant='fp16' if cuda else None,
        dtype=dtype, use_safetensors=True, low_cpu_mem_usage=True)
    if cuda:
        if config.device_strategy == 'model_cpu_offload':
            pipe.enable_model_cpu_offload()
        elif config.device_strategy == 'sequential_cpu_offload':
            pipe.enable_sequential_cpu_offload()
        else:
            raise ValueError('Unknown device strategy')
    else:
        pipe.to('cpu')
    # PyTorch SDPA is already memory efficient: no attention slicing/xFormers.
    context = {**env, **asdict(config), 'dtype': str(dtype),
               'device_strategy': config.device_strategy if cuda else 'cpu',
               'pipeline_class': type(pipe).__name__, 'scheduler': type(pipe.scheduler).__name__,
               'scheduler_config': dict(pipe.scheduler.config),
               'attention_processors': sorted({type(p).__name__ for p in pipe.unet.attn_processors.values()}),
               'model_load_seconds': time.perf_counter() - start,
               'model_resolution_note': 'Runway model API redirects to the equivalent SD 1.5 repository.'}
    return pipe, context
