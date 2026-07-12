"""Semantic retriever.

Embeds a query with the same model used for the documents, searches the vector
store, and post-processes the hits with optional score thresholding and MMR
(Maximal Marginal Relevance) re-ranking for diversity.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from src.embeddings.embedder import EmbeddingModel
from src.retrieval.base_store import BaseVectorStore
from src.utils.config import AppConfig
from src.utils.exceptions import RetrievalError
from src.utils.logger import get_logger
from src.utils.schemas import RetrievedChunk

logger = get_logger(__name__)


class Retriever:
    """Embed a query, search the vector store, and post-process the results."""

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        vector_store: BaseVectorStore,
        config: Optional[AppConfig] = None,
    ) -> None:
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.config = config or AppConfig.load()

        retr = self.config.section("retrieval")
        self.default_top_k: int = retr.get("top_k", 4)
        self.default_threshold: float = retr.get("score_threshold", 0.0)
        self.relative_score_floor: float = retr.get("relative_score_floor", 0.0)
        mmr = retr.get("mmr", {})
        self.mmr_enabled: bool = mmr.get("enabled", False)
        self.mmr_lambda: float = mmr.get("lambda", 0.5)
        self.mmr_fetch_k: int = mmr.get("fetch_k", 20)

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
        use_mmr: Optional[bool] = None,
    ) -> List[RetrievedChunk]:
        """Return the most relevant chunks for ``query``."""
        if not query or not query.strip():
            raise RetrievalError("Query must be a non-empty string.")

        top_k = top_k if top_k is not None else self.default_top_k
        threshold = score_threshold if score_threshold is not None else self.default_threshold
        use_mmr = self.mmr_enabled if use_mmr is None else use_mmr

        query_vec = self.embedding_model.embed_query(query)

        if use_mmr:
            fetch_k = max(self.mmr_fetch_k, top_k)
            candidates = self.vector_store.search(query_vec, fetch_k)
            results = self._mmr_rerank(query_vec, candidates, top_k)
        else:
            results = self.vector_store.search(query_vec, top_k)

        results = self._apply_thresholds(results, threshold)

        # Re-number ranks after filtering/re-ranking so they stay contiguous.
        for rank, r in enumerate(results):
            r.rank = rank

        logger.info(
            "Retrieved %d chunk(s) for query %r (top_k=%d, mmr=%s, thr=%.2f)",
            len(results), query[:60], top_k, use_mmr, threshold,
        )
        return results

    def _apply_thresholds(
        self, results: List[RetrievedChunk], absolute_threshold: float
    ) -> List[RetrievedChunk]:
        """Filter results by the absolute floor and the query-adaptive relative floor."""
        if not results:
            return results

        # Absolute threshold is a hard filter: respect it even if nothing survives.
        if absolute_threshold > 0.0:
            results = [r for r in results if r.score >= absolute_threshold]
            if not results:
                return results

        # Relative floor: keep the best result plus any chunk within
        # ``relative_score_floor`` of it, dropping lone weak passages.
        if self.relative_score_floor > 0.0:
            floor = self.relative_score_floor * results[0].score
            results = [r for i, r in enumerate(results) if i == 0 or r.score >= floor]

        return results

    def _mmr_rerank(
        self, query_vec: np.ndarray, candidates: List[RetrievedChunk], top_k: int
    ) -> List[RetrievedChunk]:
        """Greedy Maximal Marginal Relevance selection over candidates."""
        if not candidates:
            return []
        if len(candidates) <= top_k:
            return candidates

        cand_vecs = self.embedding_model.embed_texts([c.content for c in candidates])
        query_sims = cand_vecs @ query_vec  # relevance to query

        selected: List[int] = []
        remaining = set(range(len(candidates)))
        lam = self.mmr_lambda

        while len(selected) < top_k and remaining:
            best_idx, best_score = None, -np.inf
            for i in remaining:
                if not selected:
                    score = query_sims[i]
                else:
                    max_sim_to_selected = max(
                        float(cand_vecs[i] @ cand_vecs[j]) for j in selected
                    )
                    score = lam * query_sims[i] - (1 - lam) * max_sim_to_selected
                if score > best_score:
                    best_idx, best_score = i, score
            selected.append(best_idx)
            remaining.discard(best_idx)

        # Preserve original similarity as the reported score; order by MMR selection.
        return [candidates[i] for i in selected]
