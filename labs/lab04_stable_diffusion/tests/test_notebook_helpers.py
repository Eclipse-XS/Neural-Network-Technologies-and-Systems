from pathlib import Path

from labs.lab04_stable_diffusion.src.evaluation import load_generation_manifest
from labs.lab04_stable_diffusion.src.experiments import experiment_groups
from labs.lab04_stable_diffusion.src.notebook_helpers import STATUS_REPLAY, resolve_group


ROOT = Path(__file__).resolve().parents[1]


def test_replay_resolver_checks_every_committed_generation():
    manifest = load_generation_manifest(ROOT / 'artifacts/portfolio/generations.csv')
    specs = []
    seen = set()
    for group in experiment_groups().values():
        for spec in group:
            if spec.case_id not in seen:
                specs.append(spec)
                seen.add(spec.case_id)
    records = resolve_group(specs, mode='replay', manifest=manifest)
    assert len(records) == 13
    assert all(record['status'] == STATUS_REPLAY for record in records)
    assert all(record['sha256_verified'] for record in records)
