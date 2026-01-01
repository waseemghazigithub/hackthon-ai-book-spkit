"""RAG agent for answering questions about book content.

This module orchestrates retrieval and generation to provide
zero-hallucination answers with citations.
"""
from typing import List, Optional, Dict, Any
from qdrant_client.models import ScoredPoint
from openai import OpenAI

from models.question import Question
from models.answer import Answer, Citation
from services.retrieval_service import get_retrieval_service
from config import settings, get_settings
import logging

logger = logging.getLogger(__name__)


class RAGAgent:
    """RAG (Retrieval-Augmented Generation) agent.

    Combines semantic search with LLM generation to provide
    answers based on book content only.

    Attributes:
        settings: Application settings
        retrieval_service: Service for semantic search
        openai_client: OpenAI client for generation
        system_prompt: System prompt for LLM
    """

    def __init__(self, settings_instance=None):
        """Initialize RAG agent.

        Args:
            settings_instance: Optional settings instance (uses global if None)
        """
        self.settings = settings_instance or get_settings()
        self.retrieval_service = get_retrieval_service()

        # Initialize OpenAI client (supports OpenRouter via base_url)
        try:
            # Check if OpenRouter is configured
            if self.settings.openrouter_api_key and self.settings.router_base_url:
                self.openai_client = OpenAI(
                    api_key=self.settings.openrouter_api_key,
                    base_url=self.settings.router_base_url
                )
                self.model_name = self.settings.openrouter_model or self.settings.openai_model
                logger.info(f"OpenRouter client initialized with model: {self.model_name}")
            else:
                # Use standard OpenAI
                self.openai_client = OpenAI(api_key=self.settings.openai_api_key)
                self.model_name = self.settings.openai_model
                logger.info(f"OpenAI client initialized with model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize LLM client: {e}")
            raise

        # System prompt for zero-hallucination and citations
        self.system_prompt = """You are a helpful assistant answering questions about a technical book.

IMPORTANT RULES:
1. Answer ONLY using the provided book excerpts below. Do not use external knowledge.
2. Include citations in format [Chapter X, Section Y] for all factual claims.
3. If no relevant information is found in the excerpts, state this clearly.
4. Keep answers concise and focused on the user's question.
5. Explain technical concepts clearly but stay grounded in the provided text.
6. When selected text is provided, prioritize and reference that content heavily.

BOOK EXCERPTS:
{retrieved_context}

SELECTED TEXT (user highlighted passage):
{selected_text}

User Question: {user_question}

Answer:"""

    def generate_answer(
        self,
        question: str,
        selected_text: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> Answer:
        """Generate an answer for a given question.

        Retrieves relevant content from book and generates
        answer using OpenAI with citations.

        Args:
            question: User's question
            selected_text: Optional highlighted text (for context-aware queries)
            conversation_history: Optional previous messages for context

        Returns:
            Answer: Generated answer with citations and confidence

        Example:
            >>> agent = RAGAgent()
            >>> answer = agent.generate_answer("How does RAG work?")
            >>> print(answer.content)
            RAG retrieval works by...
            >>> print(answer.citations)
            [Citation(chapter='Chapter 1', section='RAG Architecture', ...)]
        """
        # Step 1: Retrieve relevant content
        try:
            retrieved_points = self.retrieval_service.retrieve_for_question(
                question=question,
                selected_text=selected_text
            )
            logger.info(f"Retrieved {len(retrieved_points)} points")
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            # Return error answer
            return self._generate_error_answer(question, f"Failed to retrieve relevant content: {str(e)}")

        # Step 2: Format retrieved content into context
        retrieved_context = self.retrieval_service.get_context_for_answer(retrieved_points)

        # Step 3: Generate answer using LLM
        try:
            response = self._call_openai(question, retrieved_context, selected_text, conversation_history)

            # Step 4: Parse response and build answer with citations
            answer = self._build_answer_from_response(
                response,
                retrieved_points,
                question,
                selected_text,
                conversation_history
            )

            return answer

        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            return self._generate_error_answer(question, f"Failed to generate answer: {str(e)}")

    def _call_openai(
        self,
        question: str,
        retrieved_context: str,
        selected_text: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """Call OpenAI API to generate answer.

        Args:
            question: User's question
            retrieved_context: Formatted retrieved book content
            selected_text: Optional highlighted text for context
            conversation_history: Optional previous messages

        Returns:
            str: Generated answer text
        """
        # Build system prompt with selected text context
        system_prompt = self.system_prompt.format(
            retrieved_context=retrieved_context,
            user_question=question,
            selected_text=selected_text or "N/A"
        )

        messages = [{"role": "system", "content": system_prompt}]

        # Add conversation history if provided
        if conversation_history:
            messages.extend(conversation_history)

        # Add current question
        messages.append({"role": "user", "content": question})

        try:
            response = self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.settings.openai_temperature,
                max_tokens=500  # Concise answers
            )

            answer_text = response.choices[0].message.content
            logger.debug(f"LLM response generated: {len(answer_text)} characters")
            return answer_text

        except Exception as e:
            logger.error(f"LLM API call failed: {e}")
            raise

    def _build_answer_from_response(
        self,
        response_text: str,
        retrieved_points: List[ScoredPoint],
        question: str,
        selected_text: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> Answer:
        """Build Answer model from LLM response.

        Extracts citations from retrieved points and calculates confidence.
        Boosts citations from selected text when provided.

        Args:
            response_text: Generated answer from LLM
            retrieved_points: Retrieved book content points
            question: Original user question
            selected_text: Optional highlighted text for context
            conversation_history: Optional previous messages

        Returns:
            Answer: Structured answer with citations
        """
        if not retrieved_points:
            # No content found - answer is not from book
            return Answer.not_from_book(
                content=response_text,
                related_message_id="n/a"  # Will be set by caller
            )

        # Extract citations from retrieved points
        citations = self._extract_citations_from_points(retrieved_points)

        # Boost citation scores for selected text context
        if selected_text:
            citations = self._boost_selected_text_citations(citations, selected_text)

        # Calculate confidence based on retrieval scores
        confidence = self._calculate_confidence(retrieved_points)

        return Answer.from_book_content(
            content=response_text,
            citations=citations,
            confidence=confidence,
            related_message_id="n/a"  # Will be set by caller
        )

    def _extract_citations_from_points(self, points: List[ScoredPoint]) -> List[Citation]:
        """Extract citations from retrieved points.

        Args:
            points: Retrieved ScoredPoint objects

        Returns:
            List[Citation]: List of citations with chapter/section/chunk_id
        """
        citations = []

        for point in points:
            payload = point.payload
            citation = Citation(
                chapter=payload.get("chapter", "Unknown"),
                section=payload.get("section"),
                chunk_id=payload.get("chunk_id", 0),
                relevance_score=point.score
            )
            citations.append(citation)

        return citations

    def _boost_selected_text_citations(
        self,
        citations: List[Citation],
        selected_text: str
    ) -> List[Citation]:
        """Boost citation scores for selected text context.

        When selected text is provided, citations matching
        that context get a relevance boost.

        Args:
            citations: List of citations from retrieved points
            selected_text: User-selected text

        Returns:
            List[Citation]: Citations with boosted scores
        """
        # Extract metadata from selected text
        from services.retrieval_service import RetrievalService
        retrieval_service = RetrievalService()
        selected_metadata = retrieval_service._extract_metadata_from_text(selected_text)

        # Boost scores for matching metadata
        for citation in citations:
            if selected_metadata.get("chapter") == citation.chapter:
                # Boost score for matching chapter
                citation.relevance_score = min(1.0, citation.relevance_score + 0.1)
            if selected_metadata.get("section") and citation.section:
                # Additional boost for matching section
                if selected_metadata["section"].lower() in citation.section.lower():
                    citation.relevance_score = min(1.0, citation.relevance_score + 0.05)

        # Sort by boosted score
        citations.sort(key=lambda c: c.relevance_score, reverse=True)
        return citations

    def _calculate_confidence(self, points: List[ScoredPoint]) -> float:
        """Calculate confidence score based on retrieval relevance.

        Args:
            points: Retrieved ScoredPoint objects

        Returns:
            float: Average confidence score (0.0-1.0)
        """
        if not points:
            return 0.0

        # Average relevance scores
        avg_score = sum(p.score for p in points) / len(points)

        return round(avg_score, 2)

    def _generate_error_answer(self, question: str, error_message: str) -> Answer:
        """Generate error answer when retrieval/generation fails.

        Args:
            question: Original user question
            error_message: Error description

        Returns:
            Answer: Error answer with is_from_book=False
        """
        content = f"I apologize, but I encountered an issue while processing your question: {error_message}"

        return Answer.not_from_book(
            content=content,
            related_message_id="n/a"  # Will be set by caller
        )


# Singleton instance
_rag_agent: Optional[RAGAgent] = None


def get_rag_agent() -> RAGAgent:
    """Get singleton RAG agent instance.

    Returns:
        RAGAgent: Singleton instance

    Example:
        >>> agent = get_rag_agent()
        >>> answer = agent.generate_answer("How does RAG work?")
    """
    global _rag_agent
    if _rag_agent is None:
        _rag_agent = RAGAgent()
    return _rag_agent
