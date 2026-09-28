import hashlib
import json
from collections import Counter
from labs.lab05_multimodal.src.config import INPUT_IMAGE_PATH
from labs.lab05_multimodal.src.experiments import load_questions


def test_question_taxonomy_and_references():
    questions = load_questions()
    counts = Counter(q['question_type'] for q in questions)
    assert counts == dict(descriptive=3, factual=3, analytical=3, unanswerable=1)
    assert len({q['question_id'] for q in questions}) == len(questions)
    for q in questions:
        assert q['question'].strip() and q['evaluation_notes'].strip()
        assert q['answerable'] == (q['question_type'] != 'unanswerable')
        if q['question_type'] == 'factual':
            assert q['expected_facts']
    assert questions[-1]['expected_facts'] == []


def test_image_provenance():
    source = json.loads(INPUT_IMAGE_PATH.with_name('source.json').read_text())
    assert source['sha256'] == hashlib.sha256(INPUT_IMAGE_PATH.read_bytes()).hexdigest()
    from PIL import Image
    with Image.open(INPUT_IMAGE_PATH) as image:
        assert image.size == (source['width'], source['height'])
