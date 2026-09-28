from unittest.mock import Mock
import numpy as np
import pytest
from langchain_core.documents import Document
from labs.lab03_rag.src.retrieval import retrieve, result_table, make_retriever
from labs.lab03_rag.src.evaluation import relevance_metrics
from labs.lab03_rag.src.embeddings import CachedEmbeddings, validate_vectors
from labs.lab03_rag.src.redis_store import corpus_fingerprint, index_info


def test_redis_call_rank_score_and_filter():
    doc = Document(page_content='test fixture',metadata=dict(source_file='a.csv',row_id=1,
                    record_id='a.csv:1', product_id='7',category='earrings'))
    store = Mock()
    store.similarity_search_with_score.return_value = [(doc,.25)]
    pairs, elapsed = retrieve(store,'question',5,'earrings')
    assert store.similarity_search_with_score.call_args.kwargs['k'] == 5
    assert 'earrings' in str(store.similarity_search_with_score.call_args.kwargs['filter'])
    row = result_table('question',pairs,elapsed).iloc[0]
    assert row['rank'] == 1 and row['cosine_distance'] == .25 and row['source_file'] == 'a.csv'
    make_retriever(store,5)
    assert store.as_retriever.call_args.kwargs['search_type'] == 'similarity'


def test_metrics_do_not_score_unsupported_and_expose_missing_rows():
    assert relevance_metrics(['wrong','a'],['a','b']) == dict(hit_at_k=1,first_relevant_rank=2,
                                                            reciprocal_rank=.5,recall_at_k=.5)
    assert relevance_metrics(['a'],[])['hit_at_k'] is None
    assert relevance_metrics([],['a'])['hit_at_k'] == 0


def test_embedding_cache_and_shape(tmp_path):
    backend = Mock()
    vector = np.zeros(384); vector[0] = 1
    backend.embed_documents.return_value = [vector.tolist()]
    cache = CachedEmbeddings(backend,tmp_path)
    assert cache.embed_documents(['a','a']) == [vector.tolist()]*2
    cache.embed_query('a')
    backend.embed_documents.assert_called_once_with(['a'])
    with pytest.raises(ValueError):
        validate_vectors([[float('nan')]*384])
    with pytest.raises(ValueError):
        validate_vectors([[0]*383])


def test_fingerprint_includes_content_and_metadata():
    a = Document(page_content='A',metadata={'record_id':'a:0'})
    b = Document(page_content='B',metadata={'record_id':'a:0'})
    assert corpus_fingerprint([a]) != corpus_fingerprint([b])


def test_reject_wrong_redis_dimensions():
    client = Mock()
    client.execute_command.return_value = [b'attributes', [[b'type',b'VECTOR',b'dim',128,
                                                          b'distance_metric',b'COSINE']]]
    with pytest.raises(ValueError,match='schema'):
        index_info(client,'test')
