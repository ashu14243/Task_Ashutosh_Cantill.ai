"""Classifiers that produce a probability over taxonomy leaves.

ZeroShotLexical  - needs only the tree: weighted word overlap between the line and each leaf's
                   path, bridged by the generic procurement vocabulary (lexicon.py).
TextModel        - TF-IDF (char + word n-grams) logistic regression; trained on labels when a
                   company has them, or on the zero-shot model's confident predictions (self-training).
"""
from __future__ import annotations

import logging
import math
import re
from collections import defaultdict

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize as sk_normalize

from .lexicon import STOPWORDS, SYNONYMS, stem, strip_accents
from .taxonomy import Taxonomy
from .text import expand, tokens

log = logging.getLogger(__name__)


def _softmax(x: np.ndarray, temp: float) -> np.ndarray:
    z = x / max(temp, 1e-6)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def tree_acronyms(tax: Taxonomy, loose: bool = False) -> dict[str, list[str]]:
    """Acronyms of multi-word category names.

    strict (default): only acronyms the tree itself uses - if 'Continuous Miners' and 'CM Field
    Service' are both in the tree, 'cm' -> ['continuous', 'miner'].
    loose: every 3-4 letter acronym of a 3-4 word name ('Can Body Stock (coil)' -> 'cbs'), used to
    read invoice lines that abbreviate a category name."""
    names = sorted({n for lf in tax.leaves for n in lf})   # sorted: results must not depend on hash order
    phrases: dict[str, list[str]] = {}
    for n in names:
        variants = sorted({n, re.sub(r"\(.*?\)", " ", n)})
        for v in variants:
            words = [w for w in re.split(r"[^A-Za-z]+", strip_accents(v)) if w and w.lower() not in STOPWORDS]
            if 2 <= len(words) <= 4:
                acr = "".join(w[0] for w in words).lower()
                phrases.setdefault(acr, [stem(w.lower()) for w in words])
    if loose:
        return {a: p for a, p in phrases.items() if len(a) >= 3 and len(p) >= 3}
    used = set()
    for n in names:
        for w in re.split(r"[^A-Za-z]+", n):
            if 2 <= len(w) <= 4 and w.isupper():
                used.add(w.lower())
    return {a: p for a, p in phrases.items() if a in used}


class ZeroShotLexical:
    def __init__(self, tax: Taxonomy, temperature: float = 0.06, alpha: float = 1.0):
        self.tax = tax
        self.temperature = temperature
        self.alpha = alpha
        self.acr = tree_acronyms(tax)
        self.row_acr = {**tree_acronyms(tax, loose=True), **self.acr}
        self.extra_stop = frozenset()
        # the tree's own words become lemmatisation targets ('recycled' -> 'recycle')
        from .text import register_known
        register_known({stem(w.lower()) for lf in tax.leaves for n in lf
                        for w in re.split(r"[^A-Za-z]+", strip_accents(n)) if len(w) > 2})
        vocab: dict[str, int] = {}
        rows, cols, vals = [], [], []
        for i, lf in enumerate(tax.leaves):
            bag: dict[str, float] = defaultdict(float)
            d = len(lf)
            for k, name in enumerate(lf):
                w = 1.0 if k == d - 1 else (0.6 if k == d - 2 else 0.35)
                toks = list(self._cat_tokens(name))
                for t in toks:
                    bag[t] = max(bag[t], w)
                    for s in sorted(SYNONYMS.get(t, ())):
                        bag[s] = max(bag[s], 0.7 * w)
            desc = tax.descriptions.get(lf, "")
            for t in tokens(desc):
                bag[t] = max(bag[t], 0.5)
                for s in sorted(SYNONYMS.get(t, ())):
                    bag[s] = max(bag[s], 0.35)
            for t, v in sorted(bag.items()):
                j = vocab.setdefault(t, len(vocab))
                rows.append(i)
                cols.append(j)
                vals.append(v)
        self.vocab = vocab
        C = sp.csr_matrix((vals, (rows, cols)), shape=(tax.n, len(vocab)))
        df = np.asarray((C > 0).sum(axis=0)).ravel()
        self.idf = np.log((tax.n + 1.0) / (df + 0.5))
        Cw = C.multiply(self.idf).tocsr()
        # length normalisation: alpha=1 is cosine (penalises long category names), 0 is a raw dot product
        norms = np.sqrt(np.asarray(Cw.multiply(Cw).sum(axis=1)).ravel())
        self.C = sp.diags(1.0 / np.maximum(norms, 1e-9) ** self.alpha) @ Cw

    def _cat_tokens(self, name: str):
        out = []
        for w in re.split(r"[^A-Za-z0-9&]+", strip_accents(name)):
            if not w:
                continue
            lw = w.lower()
            if lw in self.acr:
                out.extend(self.acr[lw])
                continue
            out.extend(tokens(w))
        return out

    def row_matrix(self, field_texts: list[tuple[list[str], float]]) -> sp.csr_matrix:
        n = len(field_texts[0][0])
        rows, cols, vals = [], [], []
        V = self.vocab
        for texts, w in field_texts:
            for i, t in enumerate(texts):
                if not t:
                    continue
                acc: dict[int, float] = {}
                toks = tokens(t)
                for tok in toks:
                    if tok in self.row_acr:
                        wa = 0.9 if tok in self.acr else 0.5   # tree-defined acronym vs guessed one
                        exp = {x: wa for x in self.row_acr[tok]}
                    else:
                        exp = expand(tok)
                    for word, ww in exp.items():
                        j = V.get(word)
                        if j is not None:
                            acc[j] = acc.get(j, 0.0) + ww * w
                for j, v in sorted(acc.items()):
                    rows.append(i)
                    cols.append(j)
                    vals.append(math.sqrt(v))
        R = sp.csr_matrix((vals, (rows, cols)), shape=(n, len(V)))
        return sk_normalize(R.multiply(self.idf).tocsr())

    def predict(self, field_texts) -> tuple[np.ndarray, np.ndarray]:
        R = self.row_matrix(field_texts)
        S = (R @ self.C.T).toarray()
        if self.alpha != 1.0:   # rescale so the temperature means the same thing
            mx = S.max()
            S = S / mx if mx > 0 else S
        evidence = S.max(axis=1)
        P = _softmax(S, self.temperature)
        P[evidence <= 0] = 1.0 / self.tax.n
        return P, evidence


def _norm_for_model(t: str) -> str:
    return " ".join(tokens(t))


class TextModel:
    """TF-IDF char+word n-grams -> multinomial logistic regression over leaf indices."""

    def __init__(self, n_classes: int, C: float = 4.0):
        self.n_classes = n_classes
        self.C = C
        self.char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=1, sublinear_tf=True,
                                    max_features=200000, lowercase=True)
        self.word = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1, sublinear_tf=True,
                                    token_pattern=r"[a-z0-9]{2,}", max_features=100000)
        self.clf = None
        self.classes_ = None
        self.Xtrain = None

    def _raw(self, texts):
        return [t.lower() for t in texts]

    def _feat(self, texts, fit=False):
        raw = self._raw(texts)
        normd = [_norm_for_model(t) for t in texts]
        if fit:
            a = self.char.fit_transform(raw)
            b = self.word.fit_transform(normd)
        else:
            a = self.char.transform(raw)
            b = self.word.transform(normd)
        return sk_normalize(sp.hstack([a, b]).tocsr())

    def fit(self, texts: list[str], y: np.ndarray, weights: np.ndarray | None = None):
        X = self._feat(texts, fit=True)
        self.Xtrain = X
        classes = np.unique(y)
        self.classes_ = classes
        if len(classes) == 1:
            self.clf = None
            return self
        self.clf = LogisticRegression(C=self.C, max_iter=300, solver="lbfgs")
        self.clf.fit(X, y, sample_weight=weights)
        return self

    def predict_proba(self, texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """Return (P over all leaves, max cosine similarity to a training example)."""
        X = self._feat(texts)
        P = np.zeros((len(texts), self.n_classes))
        if self.clf is None:
            P[:, self.classes_[0]] = 1.0
        else:
            P[:, self.classes_] = self.clf.predict_proba(X)
        sim = np.zeros(len(texts))
        step = 2000
        XT = self.Xtrain.T.tocsc()
        for s in range(0, len(texts), step):
            block = (X[s:s + step] @ XT)
            sim[s:s + step] = block.max(axis=1).toarray().ravel()
        return P, sim
