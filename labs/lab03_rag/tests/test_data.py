import csv
import json
import pytest
from labs.lab03_rag.src.data import csv_files, inspect_csv, load_csv_documents
from labs.lab03_rag.src.documents import build_documents, serialize_record, number


def write_csv(path, rows):
    with path.open('w', encoding='utf-8', newline='') as stream:
        csv.writer(stream).writerows(rows)


@pytest.mark.parametrize('count', [0, 1, 4])
def test_invalid_file_count(tmp_path, count):
    for i in range(count):
        write_csv(tmp_path / f'{i}.csv', [['name'], ['example']])
    with pytest.raises(ValueError, match='2–3'):
        csv_files(tmp_path)


@pytest.mark.parametrize('rows', [[['name','name'],['x','y']], [['name','price'],['x']], [['name']], [['name'],['']]])
def test_unusable_schema(tmp_path, rows):
    p = tmp_path / 'broken.csv'
    write_csv(p, rows)
    with pytest.raises(ValueError, match='Unusable CSV'):
        inspect_csv(p)


def test_actual_csvloader_preserves_multiline_and_provenance(tmp_path):
    for name in ['a.csv', 'b.csv']:
        write_csv(tmp_path / name, [['product_id','title','product_description','final_price','discount'],
                                   ['7','Brand','line one\nline two: with colon','"₹1,299.00"','']])
    loaded = load_csv_documents(tmp_path)
    docs = build_documents(loaded)
    assert len(docs) == 2
    assert 'line one\nline two: with colon' in docs[0].page_content
    assert '"₹1,299.00"' in docs[0].page_content
    assert docs[0].metadata['row_id'] == 0
    assert docs[0].metadata['record_id'] == 'a.csv:0'
    assert docs[0].metadata['product_id'] == '7'
    assert docs[0].metadata['discount'] is None
    assert docs[0].metadata['final_price'] == 1299
    assert inspect_csv(tmp_path / 'a.csv')['missing']['discount'] == 1


def test_configurable_schema_and_nested_fields():
    assert 'name: Widget' in serialize_record({'name':'Widget','price':'0.00'},fields=['name','price'])
    assert 'price: 0.00' in serialize_record({'name':'Widget','price':'0.00'},fields=['name','price'])
    row = dict(title='Brand',product_specifications=json.dumps([
        {'specification_name':'Fabric','specification_value':'Cotton'},
        {'specification_name':'Material'}, {'specification_name':'Closure','specification_value':'NA'}]))
    text = serialize_record(row)
    assert 'Fabric: Cotton' in text and 'Closure:' not in text and 'Material:' not in text
    with pytest.raises(ValueError):
        serialize_record({'irrelevant':'x'})


def test_missing_not_zero_and_invalid_number():
    assert number('') is None and number('0') == 0
    with pytest.raises(ValueError):
        number('not a price')
