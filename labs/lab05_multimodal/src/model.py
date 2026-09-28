"""Lazy model loading and measured environment; never imported by CPU tests."""
import importlib.metadata
import platform
import shutil
import time
from .config import MODEL_ID, MODEL_REVISION, LAB_ROOT


def environment():
    import torch
    import psutil
    from huggingface_hub import try_to_load_from_cache
    versions = {p: importlib.metadata.version(p) for p in (
        'torch', 'transformers', 'accelerate', 'Pillow', 'safetensors', 'pandas',
        'numpy', 'matplotlib', 'pytest', 'nbformat', 'ipykernel', 'nbclient')}
    available = torch.cuda.is_available()
    cached = try_to_load_from_cache(MODEL_ID, 'model.safetensors', revision=MODEL_REVISION)
    return dict(python=platform.python_version(), platform=platform.system(), versions=versions,
                cuda_runtime=torch.version.cuda, cuda_available=available,
                gpu_name=torch.cuda.get_device_name(0) if available else None,
                vram_bytes=torch.cuda.get_device_properties(0).total_memory if available else None,
                free_vram_bytes=torch.cuda.mem_get_info()[0] if available else None,
                free_ram_bytes=psutil.virtual_memory().available,
                free_disk_bytes=shutil.disk_usage(LAB_ROOT).free,
                weights_cached=isinstance(cached, str))


def load_pipeline():
    import torch
    from transformers import pipeline, ImageTextToTextPipeline, GenerationConfig

    class MeasuredImageTextToTextPipeline(ImageTextToTextPipeline):
        """Keep standard decoding; expose actual continuation length and EOS."""
        def postprocess(self, model_outputs, **kwargs):
            records = super().postprocess(model_outputs, **kwargs)
            prefix_length = model_outputs['input_ids'].shape[-1]
            eos = self.generation_config.eos_token_id
            eos = {eos} if isinstance(eos, int) else set(eos or [])
            for record, sequence in zip(records, model_outputs['generated_sequence']):
                tokens = sequence[prefix_length:].tolist()
                record['generated_token_count'] = len(tokens)
                record['ended_with_eos'] = bool(tokens and tokens[-1] in eos)
            return records

    device = 0 if torch.cuda.is_available() else -1
    dtype = torch.float16 if device == 0 else torch.float32
    started = time.perf_counter()
    pipe = pipeline('image-text-to-text', model=MODEL_ID, revision=MODEL_REVISION,
                    dtype=dtype, device=device, batch_size=1,
                    model_kwargs={'attn_implementation': 'sdpa'},
                    pipeline_class=MeasuredImageTextToTextPipeline)
    pipe.model.eval()
    # Remove checkpoint/pipeline sampling defaults; case kwargs are authoritative.
    pipe.generation_config = GenerationConfig(
        eos_token_id=pipe.model.generation_config.eos_token_id,
        pad_token_id=pipe.processor.tokenizer.pad_token_id,
        bos_token_id=pipe.model.generation_config.bos_token_id,
        do_sample=False, max_new_tokens=64)
    config = pipe.model.config
    info = dict(model_id=MODEL_ID, revision=MODEL_REVISION,
                model_class=type(pipe.model).__name__, pipeline_class=type(pipe).__name__,
                pipeline_base_class=ImageTextToTextPipeline.__name__,
                dtype=str(dtype), device_strategy='cuda:0' if device == 0 else 'cpu',
                attention='sdpa', model_load_seconds=time.perf_counter() - started,
                parameter_count=sum(p.numel() for p in pipe.model.parameters()),
                vision_backbone=config.vision_config.model_type,
                language_backbone=config.text_config.model_type,
                image_size=config.vision_config.image_size,
                processor_class=type(pipe.processor).__name__,
                image_processor=pipe.processor.image_processor.to_dict(),
                model_config=config.to_dict(),
                generation_defaults=pipe.generation_config.to_dict())
    return pipe, info
