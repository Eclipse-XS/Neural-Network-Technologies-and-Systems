"""Create compact, versioned Lab 5 evidence from an existing verified run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('source_outputs', type=Path)
    parser.add_argument('target_evidence', type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.source_outputs / 'metadata/manifest.json').read_text(encoding='utf-8'))
    unique = {row['run_id']: row for row in manifest}
    runs = []
    for row in unique.values():
        generation = {key: row[key] for key in
                      ('do_sample', 'max_new_tokens', 'temperature', 'top_p', 'seed')}
        keep = {key: row[key] for key in (
            'run_id', 'question_id', 'question_type', 'question', 'answerable',
            'expected_facts', 'answer', 'model_id', 'revision', 'image_path',
            'image_sha256', 'grounding_instruction', 'runtime_seconds',
            'generated_token_count', 'ended_with_eos', 'reached_token_limit',
            'answer_word_count', 'answer_character_count', 'created_at',
            'execution_context', 'pipeline_output_schema')}
        keep['generation_config'] = generation
        runs.append(keep)
    if len(runs) != 17:
        raise SystemExit(f'Expected 17 unique runs, found {len(runs)}')
    args.target_evidence.mkdir(parents=True, exist_ok=True)
    payload = {
        'schema_version': 1,
        'provenance': {
            'kind': 'historical_gpu_inference',
            'source_manifest': 'outputs/metadata/manifest.json',
            'note': 'Copied losslessly from the locally preserved run; runtimes are historical, not replay timings.'
        },
        'runs': runs,
    }
    (args.target_evidence / 'verified_runs.json').write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    for source_name, target_name in [('model.json', 'model_metadata.json'),
                                     ('environment.json', 'environment.json')]:
        value = json.loads((args.source_outputs / 'metadata' / source_name).read_text(encoding='utf-8'))
        (args.target_evidence / target_name).write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
