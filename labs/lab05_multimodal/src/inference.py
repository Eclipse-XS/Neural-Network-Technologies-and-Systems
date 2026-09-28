"""Grounded inference, content-addressed resume, atomic result persistence."""
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import time
from .config import (MODEL_ID, MODEL_REVISION, GROUNDING_INSTRUCTION,
                     INPUT_IMAGE_PATH, LAB_ROOT, OUTPUT_ROOT)


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def image_sha():
    return hashlib.sha256(INPUT_IMAGE_PATH.read_bytes()).hexdigest()


def signature(case, context):
    payload = dict(image_sha256=image_sha(), model_id=MODEL_ID, revision=MODEL_REVISION,
                   question=case.question['question'], grounding_instruction=GROUNDING_INSTRUCTION,
                   generation_config=asdict(case.generation), execution_context=context)
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def cached_result(case, context):
    run_id = signature(case, context)
    path = OUTPUT_ROOT / f'responses/{run_id}.json'
    if not path.exists():
        return None
    try:
        record = json.loads(path.read_text(encoding='utf-8'))
    except (ValueError, OSError):
        return None
    if record.get('run_id') != run_id or not record.get('answer', '').strip():
        return None
    return record


def extract_answer(output):
    """Accept text or assistant chat payload; reject user-only/unrecognized output."""
    if not isinstance(output, list) or len(output) != 1 or not isinstance(output[0], dict):
        raise ValueError('Expected one pipeline result')
    generated = output[0].get('generated_text')
    if isinstance(generated, str):
        answer = generated.strip()
    elif isinstance(generated, list):
        messages = [m for m in generated if isinstance(m, dict) and m.get('role') == 'assistant']
        if not messages:
            raise ValueError('No assistant message in pipeline output')
        content = messages[-1]['content']
        answer = content if isinstance(content, str) else '\n'.join(
            part['text'] for part in content if part.get('type') == 'text')
        answer = answer.strip()
    else:
        raise ValueError('Unknown generated_text schema')
    if not answer:
        raise ValueError('Empty assistant answer')
    return answer


def answer_question(pipe, image, case, context):
    record = cached_result(case, context)
    if record is not None:
        return record
    if pipe is None:
        raise RuntimeError('Missing inference result; load LLaVA before generation')
    import torch
    from transformers import set_seed
    if case.generation.do_sample:
        set_seed(case.generation.seed)
    prompt = GROUNDING_INSTRUCTION + '\n\nQuestion: ' + case.question['question']
    messages = [{'role': 'user', 'content': [
        {'type': 'image', 'image': image}, {'type': 'text', 'text': prompt}]}]
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    started = time.perf_counter()
    output = pipe(text=messages, return_full_text=False, generate_kwargs=case.generation.kwargs())
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    answer = extract_answer(output)
    count = output[0].get('generated_token_count')
    eos = output[0].get('ended_with_eos')
    record = dict(run_id=signature(case, context), experiment=case.experiment,
                  **case.question, answer=answer, model_id=MODEL_ID, revision=MODEL_REVISION,
                  image_path=INPUT_IMAGE_PATH.relative_to(LAB_ROOT).as_posix(),
                  image_sha256=image_sha(), grounding_instruction=GROUNDING_INSTRUCTION,
                  **asdict(case.generation), runtime_seconds=elapsed,
                  generated_token_count=count, ended_with_eos=eos,
                  reached_token_limit=(count == case.generation.max_new_tokens and not eos),
                  answer_word_count=len(answer.split()), answer_character_count=len(answer),
                  created_at=datetime.now(timezone.utc).isoformat(), execution_context=context,
                  pipeline_output_schema={k: type(v).__name__ for k, v in output[0].items()})
    save_json(OUTPUT_ROOT / f"responses/{record['run_id']}.json", record)
    print(f"{case.experiment}/{case.case_id}: {elapsed:.2f}s | {answer}", flush=True)
    return record
