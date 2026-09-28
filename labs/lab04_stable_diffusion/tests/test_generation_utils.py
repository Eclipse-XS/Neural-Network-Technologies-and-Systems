from dataclasses import replace
import hashlib
import json
import re
from PIL import Image
from labs.lab04_stable_diffusion.src.config import GenerationSpec
from labs.lab04_stable_diffusion.src.generation import artifact_id, filename, identity, save_json, cached_result


def context():
    return dict(model_id='test', revision='abc', pipeline_class='Fake', scheduler='PNDM', scheduler_config={'skip_prk_steps': True},
                dtype='float16', device_strategy='cpu', torch_version='test', diffusers_version='test', cuda_version=None, gpu_name=None)


def test_filename_and_complete_identity():
    s, ctx = GenerationSpec(), context()
    key = artifact_id(s, ctx)
    assert re.fullmatch(r'[A-Za-z0-9_.-]+', filename(s, key))
    assert 'seed42_steps30_cfg7p5' in filename(s, key)
    for field, value in [('seed', 123), ('prompt', 'other scene'), ('negative_prompt', 'blur'),
                         ('num_inference_steps', 20), ('guidance_scale', 5.), ('width', 768)]:
        assert artifact_id(replace(s, **{field: value}), ctx) != key
    for field, value in [('model_id', 'other'), ('revision', 'other'), ('scheduler_config', {'skip_prk_steps': False})]:
        assert artifact_id(s, {**ctx, field: value}) != key


def test_sidecar_roundtrip_and_reuse_requires_integrity(tmp_path):
    s, ctx = GenerationSpec(), context()
    key = artifact_id(s, ctx)
    image = tmp_path/'image.png'
    Image.new('RGB', (512, 512)).save(image)
    record = {'identity': identity(s, ctx), 'image_path': 'image.png',
              'image_sha256': hashlib.sha256(image.read_bytes()).hexdigest()}
    sidecar = tmp_path/'outputs/metadata'/f'{key}.json'
    assert cached_result(s, ctx, tmp_path) is None
    save_json(sidecar, record)
    assert json.loads(sidecar.read_text()) == record
    assert cached_result(s, ctx, tmp_path) == record
    image.write_bytes(b'corrupt')
    assert cached_result(s, ctx, tmp_path) is None
