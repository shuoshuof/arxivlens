import numpy as np
from datetime import datetime

import torch
from sentence_transformers import SentenceTransformer

from paper import ArxivPaper


def rerank_paper(
    candidate: list[ArxivPaper],
    corpus: list[dict],
    model: str = "avsolatorio/GIST-small-Embedding-v0",
) -> list[ArxivPaper]:
    encoder = SentenceTransformer(
        model, device="cuda" if torch.cuda.is_available() else "cpu"
    )
    corpus = sorted(
        corpus,
        key=lambda x: datetime.strptime(x["data"]["dateAdded"], "%Y-%m-%dT%H:%M:%SZ"),
        reverse=True,
    )
    time_decay_weight = 1 / (1 + np.log10(np.arange(len(corpus)) + 1))
    time_decay_weight = time_decay_weight / time_decay_weight.sum()
    corpus_feature = encoder.encode([paper["data"]["abstractNote"] for paper in corpus])
    candidate_feature = encoder.encode([paper.summary for paper in candidate])
    sim = encoder.similarity(candidate_feature, corpus_feature)
    scores = (sim * time_decay_weight).sum(axis=1) * 10
    for score, paper in zip(scores, candidate):
        paper.score = float(score)
    candidate = sorted(candidate, key=lambda x: x.score or 0.0, reverse=True)
    return candidate
