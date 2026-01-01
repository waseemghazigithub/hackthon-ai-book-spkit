"""Content retrieval service using semantic search.

This module handles semantic search for relevant book content,
including metadata filtering for selected-text queries.
"""
from typing import List, Optional, Dict, Any
from qdrant_client.models import ScoredPoint, Filter
from services.embedding_service import get_embedding_service
from services.qdrant_service import get_qdrant_service
from config import settings, get_settings
import logging
import re

logger = logging.getLogger(__name__)


class RetrievalService:
    """Service for retrieving relevant book content.

    Uses semantic search via Qdrant with optional metadata filtering
    for selected-text queries.
    """

    def __init__(self, settings_instance=None):
        """Initialize retrieval service.

        Args:
            settings_instance: Optional settings instance (uses global if None)
        """
        self.settings = settings_instance or get_settings()
        self.embedding_service = get_embedding_service()
        self.qdrant_service = get_qdrant_service()

    def retrieve_for_question(
        self,
        question: str,
        selected_text: Optional[str] = None,
        top_k: Optional[int] = None
    ) -> List[ScoredPoint]:
        """Retrieve relevant content for a question.

        Supports optional selected text for metadata filtering.
        Adjusts top_k based on question complexity if not specified.

        Args:
            question: User's question text
            selected_text: Optional highlighted text from book (for filtering)
            top_k: Number of results to return (default from settings, up to 10)

        Returns:
            List[ScoredPoint]: List of scored points with payloads

        Example:
            >>> service = RetrievalService()
            >>> results = service.retrieve_for_question("How does RAG work?")
            >>> for result in results:
            ...     print(result.score, result.payload['text'])
        """
        if top_k is None:
            top_k = self._determine_top_k(question)

        # Generate embedding for question
        question_embedding = self.embedding_service.generate_embedding(question)

        # Apply metadata filter if selected text provided
        filter_ = self._build_filter_for_selected_text(selected_text)

        # Search Qdrant
        try:
            results = self.qdrant_service.search(
                query_vector=question_embedding,
                limit=top_k,
                filter_=filter_
            )
            logger.info(f"Retrieved {len(results)} results for question: {question[:50]}...")
            return results
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            raise

    def _determine_top_k(self, question: str) -> int:
        """Determine optimal top_k based on question complexity.

        Args:
            question: User's question text

        Returns:
            int: Number of results to retrieve (between default_top_k and max_top_k)
        """
        # Simple heuristics for complexity
        complexity_indicators = [
            "why", "how", "explain", "describe", "compare",
            "difference", "relationship", "example", "versus"
        ]

        question_lower = question.lower()
        is_complex = any(indicator in question_lower for indicator in complexity_indicators)

        # Use higher top_k for complex questions
        if is_complex:
            return min(
                self.settings.max_top_k,
                self.settings.default_top_k + 5
            )

        return self.settings.default_top_k

    def _build_filter_for_selected_text(self, selected_text: Optional[str]) -> Optional[Filter]:
        """Build Qdrant filter for selected text metadata.

        If selected text is provided, attempts to extract chapter/section
        from the text and applies metadata filter to search.

        Args:
            selected_text: Optional highlighted text from book

        Returns:
            Optional[Filter]: Qdrant metadata filter (None if no selected text or metadata found)
        """
        if not selected_text:
            return None

        # Try to extract chapter/section from selected text
        metadata = self._extract_metadata_from_text(selected_text)

        if not metadata:
            logger.debug("No metadata found in selected text, using general search")
            return None

        # Build Qdrant filter
        must_conditions = []

        if metadata.get("chapter"):
            must_conditions.append({
                "key": "chapter",
                "match": {
                    "value": metadata["chapter"]
                }
            })

        if metadata.get("section"):
            must_conditions.append({
                "key": "section",
                "match": {
                    "value": metadata["section"]
                }
            })

        if must_conditions:
            filter_ = Filter(must=must_conditions)
            logger.debug(f"Built filter for selected text: {metadata}")
            return filter_

        return None

    def _extract_metadata_from_text(self, text: str) -> Dict[str, Any]:
        """Extract chapter/section metadata from text.

        Simple heuristic: looks for patterns like "Chapter 1", "Section 2.1", etc.

        Args:
            text: Selected text

        Returns:
            Dict: Extracted metadata {"chapter": str, "section": str} (may be empty)
        """
        metadata = {}

        # Pattern for "Chapter X" or "Chapter X: Title"
        chapter_pattern = r"chapter\s+(\d+)(?::\s*(.+)?)?"
        chapter_match = re.search(chapter_pattern, text, re.IGNORECASE)
        if chapter_match:
            chapter_num = chapter_match.group(1)
            chapter_title = chapter_match.group(2).strip() if chapter_match.group(2) else None
            metadata["chapter"] = f"Chapter {chapter_num}" + (f": {chapter_title}" if chapter_title else "")

        # Pattern for "Section X.Y" or "## Section Title"
        section_pattern = r"(?:section\s+(\d+(?:\.\d+)*)?|##\s*(.+?)$)"
        section_match = re.search(section_pattern, text, re.MULTILINE | re.IGNORECASE)
        if section_match:
            if section_match.group(1).startswith("section"):
                # "Section 2.1" format
                metadata["section"] = section_match.group(1)
            else:
                # "## Section Title" format
                section_title = section_match.group(1).strip()
                # Clean up markdown markers
                section_title = re.sub(r"^#+\s*", "", section_title).strip()
                metadata["section"] = section_title

        return metadata

    def retrieve_for_chunks(
        self,
        chunk_ids: List[int]
    ) -> List[ScoredPoint]:
        """Retrieve specific chunks by their IDs.

        Used for selected-text queries where chunk IDs are known.

        Args:
            chunk_ids: List of chunk IDs to retrieve

        Returns:
            List[ScoredPoint]: Retrieved points

        Example:
            >>> service = RetrievalService()
            >>> results = service.retrieve_for_chunks([42, 43, 44])
            >>> for result in results:
            ...     print(result.id, result.payload['text'])
        """
        if not chunk_ids:
            return []

        try:
            # Use Qdrant's retrieve method for exact ID lookup
            # Note: Qdrant's retrieve method doesn't support search, so we need
            # to use scroll or batch retrieve with filter
            # For simplicity, we'll use search with a dummy vector and filter by IDs
            # In production, consider using retrieve API or scroll API

            collection_name = self.settings.qdrant_collection_name
            client = self.qdrant_service.client

            # Use scroll with filter for exact IDs
            points = []
            for chunk_id in chunk_ids:
                try:
                    # Try to retrieve exact point by ID
                    # Qdrant doesn't have direct "get by ID" in client, so we scroll
                    result = client.scroll(
                        collection_name=collection_name,
                        scroll_filter={
                            "must": [
                                {
                                    "key": "chunk_id",
                                    "match": {"value": chunk_id}
                                }
                            ]
                        },
                        limit=1
                    )
                    if result[0]:  # result is (points, next_page_offset)
                        points.extend(result[0])
                except Exception as e:
                    logger.warning(f"Failed to retrieve chunk {chunk_id}: {e}")

            logger.info(f"Retrieved {len(points)} chunks by ID")
            return points

        except Exception as e:
            logger.error(f"Chunk retrieval failed: {e}")
            raise

    def get_context_for_answer(
        self,
        retrieved_points: List[ScoredPoint]
    ) -> str:
        """Format retrieved points into context string for LLM.

        Formats each point with metadata and text.

        Args:
            retrieved_points: List of retrieved ScoredPoint objects

        Returns:
            str: Formatted context string

        Example:
            >>> context = service.get_context_for_answer(results)
            >>> print(context)
            [Chapter 1, RAG Architecture]
            Text: RAG (Retrieval-Augmented Generation) combines...
        """
        if not retrieved_points:
            return "No relevant information found in the book."

        context_parts = []

        for idx, point in enumerate(retrieved_points, 1):
            payload = point.payload
            chapter = payload.get("chapter", "Unknown")
            section = payload.get("section", "")
            text = payload.get("text", "")

            # Format with chapter/section prefix
            if section:
                header = f"[{chapter}, {section}]\n"
            else:
                header = f"[{chapter}]\n"

            context_parts.append(f"{header}{text}")

        return "\n\n".join(context_parts)


# Singleton instance
_retrieval_service: Optional[RetrievalService] = None


def get_retrieval_service() -> RetrievalService:
    """Get singleton RetrievalService instance.

    Returns:
        RetrievalService: Singleton instance

    Example:
        >>> service = get_retrieval_service()
        >>> results = service.retrieve_for_question("How does RAG work?")
    """
    global _retrieval_service
    if _retrieval_service is None:
        _retrieval_service = RetrievalService()
    return _retrieval_service
