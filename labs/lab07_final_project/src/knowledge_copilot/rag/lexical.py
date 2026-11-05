"""Okapi BM25 Lexical search engine with bilingual stemming and structure-aware weighting."""
import math
import re
from collections import Counter
from typing import List, Optional, Tuple

from nltk.stem.snowball import SnowballStemmer

from ..domain.enums import EvidenceQuality
from ..domain.models import Chunk

_STEMMER_EN = SnowballStemmer("english")
_UKR_STEM_PATTERN = re.compile(
    r"(?:ів|ей|ями|ами|ях|ах|ому|ими|ого|ою|ею|ові|єві|их|ість|ості|ння|нь|івка|івки|я|а|е|є|и|і|о|у|ю)$"
)


def stem_word(w: str) -> str:
    """Stem English and Ukrainian technical tokens."""
    w = w.casefold()
    if re.match(r"^[a-z]+$", w):
        return _STEMMER_EN.stem(w)
    if re.match(r"^[а-яіїєґ]+$", w) and len(w) > 4:
        s = _UKR_STEM_PATTERN.sub("", w)
        if len(s) >= 3:
            return s
    return w


STOPWORDS = {
    "what", "is", "the", "of", "in", "and", "a", "an", "for", "to", "how", "why", "which",
    "are", "do", "does", "did", "from", "with", "between", "when", "where", "that", "this",
    "що", "як", "яка", "який", "яке", "які", "яку", "яких", "якому", "у", "в", "на", "та",
    "і", "із", "з", "до", "про", "для", "чи", "це", "було", "була", "був", "були", "між",
}


class BM25Retriever:
    """Enhanced BM25 Retriever supporting bilingual stemming and structural field boosts."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.chunks: List[Chunk] = []
        self.doc_terms: List[List[str]] = []
        self.df: Counter = Counter()
        self.avg_dl: float = 0.0

    @staticmethod
    def tokenize(text: str) -> List[str]:
        words = re.findall(r"[0-9A-Za-zА-Яа-яІіЇїЄєҐґ_./+-]+", text.casefold())
        tokens = []
        for w in words:
            tokens.append(w)
            sw = stem_word(w)
            if sw != w:
                tokens.append(sw)
        return tokens

    def index(self, chunks: List[Chunk]) -> None:
        """Index chunks with double weighting on section headers and filename context."""
        self.chunks = chunks
        self.doc_terms = []
        for c in chunks:
            fname = c.metadata.get("filename", "") if c.metadata else ""
            sec = c.section or ""
            # Boost section name (2x) and filename for improved structural relevance
            doc_str = f"{fname} {sec} {sec} {c.text}"
            self.doc_terms.append(self.tokenize(doc_str))

        self.df = Counter(term for terms in self.doc_terms for term in set(terms))
        total_tokens = sum(len(terms) for terms in self.doc_terms)
        self.avg_dl = total_tokens / max(1, len(chunks))

    def search(
        self,
        query: str,
        k: int = 5,
        document_id: Optional[str] = None,
        filter_quoted_outputs: bool = True,
    ) -> List[Tuple[Chunk, float]]:
        if not self.chunks:
            return []

        all_q_terms = self.tokenize(query)
        q_terms = [t for t in all_q_terms if t not in STOPWORDS] or all_q_terms
        if not q_terms:
            return []

        n_docs = len(self.chunks)
        scored: List[Tuple[Chunk, float]] = []

        for chunk, terms in zip(self.chunks, self.doc_terms):
            if document_id and chunk.document_id != document_id:
                continue
            if filter_quoted_outputs and chunk.evidence_quality == EvidenceQuality.QUOTED_MODEL_OUTPUT:
                continue

            tf = Counter(terms)
            doc_len = len(terms)
            score = 0.0

            for term in q_terms:
                freq = tf[term]
                if not freq:
                    continue
                df_term = self.df[term]
                # Standard Okapi BM25 formula with idf >= 0 safety
                idf = math.log(1.0 + (n_docs - df_term + 0.5) / (df_term + 0.5))
                denom = freq + self.k1 * (1.0 - self.b + self.b * (doc_len / max(1.0, self.avg_dl)))
                score += idf * (freq * (self.k1 + 1.0)) / denom

            if score > 0.0:
                scored.append((chunk, score))

        # Sort descending by BM25 score
        scored.sort(key=lambda item: (-item[1], item[0].chunk_id))
        return scored[:k]
