from dataclasses import replace
import hashlib
import json
import re
from PIL import Image
from labs.lab04_stable_diffusion.src.config import GenerationSpec
from labs.lab04_stable_diffusion.src.generation import artifact_id, filename, identity, save_json, cached_result, generate_image


def context():
    return dict(model_id='test', revision='abc', pipeline_class='Fake', scheduler='PNDM', scheduler_config={'skip_prk_steps': True},
                dtype='float16', device_strategy='cpu', torch_version='test', diffusers_version='test', cuda_version=None, gpu_name=None, cuda_available=False)


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


def test_generate_image_cache_semantics_and_force_regenerate(tmp_path):
    class FakeOutput:
        images = [Image.new('RGB', (512, 512))]
        nsfw_content_detected = [False]

    class FakePipe:
        def __call__(self, **kwargs):
            return FakeOutput()

    s, ctx = GenerationSpec(), context()
    pipe = FakePipe()

    r1 = generate_image(pipe, s, ctx, root=tmp_path, force_regenerate=False)
    assert r1['execution_status'] == 'LIVE_GENERATED'
    assert r1['cache_hit'] is False
    assert r1['generated_in_current_run'] is True

    r2 = generate_image(pipe, s, ctx, root=tmp_path, force_regenerate=False)
    assert r2['execution_status'] == 'CACHE_REUSED'
    assert r2['cache_hit'] is True
    assert r2['generated_in_current_run'] is False

    r3 = generate_image(pipe, s, ctx, root=tmp_path, force_regenerate=True)
    assert r3['execution_status'] == 'LIVE_GENERATED'
    assert r3['cache_hit'] is False
    assert r3['generated_in_current_run'] is True

