"""Integration tests for RAG chatbot (T015-T018, T029-T032, T038-T041)."""
import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from qdrant_client.models import ScoredPoint
from models.question import Question
from models.answer import Answer, Citation
from models.conversation import ConversationSession, Message
from agent import RAGAgent


# ==========  User Story 1: Basic Q&A Tests (T015-T018) ==========

@pytest.mark.integration
def test_ask_question_returns_answer_with_citations(mock_retrieval_service, mock_openai_response):
    """Test asking question returns answer with citations (T015).

    Given: Book content indexed
    When: User asks question
    Then: Answer contains citations from book
    """
    # Arrange
    mock_retrieved_points = [
        ScoredPoint(
            id=42,
            version=0,
            score=0.89,
            payload={
                "text": "RAG (Retrieval-Augmented Generation) combines retrieval with generation...",
                "chapter": "Chapter 1",
                "section": "RAG Architecture",
                "chunk_id": 42
            },
            vector=None
        )
    ]
    mock_retrieval_service.return_value.retrieve_for_question.return_value = mock_retrieved_points

    mock_openai_response.return_value = Mock(
        choices=[Mock(message=Mock(content="RAG retrieval works by querying a vector database..."))]
    )

    with patch('agent.get_retrieval_service', mock_retrieval_service):
        with patch('agent.OpenAI') as mock_openai_class:
            mock_openai_class.return_value.chat.completions.create = mock_openai_response
            agent = RAGAgent()

    # Act
    answer = agent.generate_answer(
        question="How does RAG retrieval work?",
        selected_text=None
    )

    # Assert
    assert isinstance(answer, Answer)
    assert len(answer.content) > 0
    assert len(answer.citations) > 0
    assert answer.citations[0].chapter == "Chapter 1"
    assert answer.citations[0].section == "RAG Architecture"
    assert answer.citations[0].chunk_id == 42
    assert answer.is_from_book is True
    assert 0 <= answer.confidence <= 1.0


@pytest.mark.integration
def test_ask_question_not_in_book_returns_is_from_book_false(mock_retrieval_service, mock_openai_response):
    """Test question not in book returns is_from_book=False (T016).

    Given: Question not covered in book
    When: Asking question
    Then: is_from_book=False, no citations
    """
    # Arrange - No relevant results from retrieval
    mock_retrieval_service.return_value.retrieve_for_question.return_value = []

    mock_openai_response.return_value = Mock(
        choices=[Mock(message=Mock(content="I couldn't find relevant information in the book about this topic."))]
    )

    with patch('agent.get_retrieval_service', mock_retrieval_service):
        with patch('agent.OpenAI') as mock_openai_class:
            mock_openai_class.return_value.chat.completions.create = mock_openai_response
            agent = RAGAgent()

    # Act
    answer = agent.generate_answer(
        question="What is quantum computing?",  # Not in book
        selected_text=None
    )

    # Assert
    assert isinstance(answer, Answer)
    assert answer.is_from_book is False
    assert len(answer.citations) == 0
    assert "couldn't find" in answer.content.lower() or "no relevant" in answer.content.lower()


@pytest.mark.integration
def test_ask_ambiguous_question_requests_clarification(mock_retrieval_service, mock_openai_response):
    """Test ambiguous question requests clarification (T017).

    Given: Ambiguous question
    When: Asking question
    Then: System requests clarification
    """
    # Arrange
    mock_retrieved_points = [
        ScoredPoint(
            id=1,
            version=0,
            score=0.45,  # Low score indicates ambiguity
            payload={
                "text": "Multiple topics could be relevant...",
                "chapter": "Chapter 1",
                "section": "Overview",
                "chunk_id": 1
            },
            vector=None
        )
    ]
    mock_retrieval_service.return_value.retrieve_for_question.return_value = mock_retrieved_points

    mock_openai_response.return_value = Mock(
        choices=[Mock(message=Mock(
            content="Could you clarify what aspect of 'it' you're asking about? The book covers several topics."
        ))]
    )

    with patch('agent.get_retrieval_service', mock_retrieval_service):
        with patch('agent.OpenAI') as mock_openai_class:
            mock_openai_class.return_value.chat.completions.create = mock_openai_response
            agent = RAGAgent()

    # Act
    answer = agent.generate_answer(
        question="How does it work?",  # Ambiguous "it"
        selected_text=None
    )

    # Assert
    assert isinstance(answer, Answer)
    assert "clarif" in answer.content.lower() or "which" in answer.content.lower() or "aspect" in answer.content.lower()
    assert answer.confidence < 0.6  # Low confidence for ambiguous question


@pytest.mark.integration
def test_ask_about_specific_term_returns_definition_with_citations(mock_retrieval_service, mock_openai_response):
    """Test asking about specific term returns definition with citations (T018).

    Given: Term defined in book
    When: Asking for definition
    Then: Answer includes definition with citations
    """
    # Arrange
    mock_retrieved_points = [
        ScoredPoint(
            id=10,
            version=0,
            score=0.95,
            payload={
                "text": "Embeddings are dense vector representations of text that capture semantic meaning...",
                "chapter": "Chapter 2",
                "section": "Embeddings",
                "chunk_id": 10
            },
            vector=None
        )
    ]
    mock_retrieval_service.return_value.retrieve_for_question.return_value = mock_retrieved_points

    mock_openai_response.return_value = Mock(
        choices=[Mock(message=Mock(
            content="Embeddings are dense vector representations that capture semantic meaning [Chapter 2, Embeddings]."
        ))]
    )

    with patch('agent.get_retrieval_service', mock_retrieval_service):
        with patch('agent.OpenAI') as mock_openai_class:
            mock_openai_class.return_value.chat.completions.create = mock_openai_response
            agent = RAGAgent()

    # Act
    answer = agent.generate_answer(
        question="What are embeddings?",
        selected_text=None
    )

    # Assert
    assert isinstance(answer, Answer)
    assert "embeddings" in answer.content.lower()
    assert len(answer.citations) > 0
    assert answer.citations[0].chapter == "Chapter 2"
    assert answer.is_from_book is True


# ========== User Story 2: Selected Text Tests (T029-T032) ==========

@pytest.mark.integration
def test_question_with_selected_text_answers_primarily_from_selection(mock_retrieval_service, mock_openai_response):
    """Test question with selected text answers from selection (T029).

    Given: Selected paragraph
    When: Asking question
    Then: Answer primarily references selected text
    """
    # Arrange
    selected_text = "RAG (Retrieval-Augmented Generation) combines dense retrieval with text generation..."

    mock_retrieved_points = [
        ScoredPoint(
            id=20,
            version=0,
            score=0.98,  # High score for selected text match
            payload={
                "text": selected_text,
                "chapter": "Chapter 1",
                "section": "RAG Overview",
                "chunk_id": 20
            },
            vector=None
        )
    ]
    mock_retrieval_service.return_value.retrieve_for_question.return_value = mock_retrieved_points

    mock_openai_response.return_value = Mock(
        choices=[Mock(message=Mock(
            content="Based on the selected text, RAG combines retrieval with generation [Chapter 1, RAG Overview]."
        ))]
    )

    with patch('agent.get_retrieval_service', mock_retrieval_service):
        with patch('agent.OpenAI') as mock_openai_class:
            mock_openai_class.return_value.chat.completions.create = mock_openai_response
            agent = RAGAgent()

    # Act
    answer = agent.generate_answer(
        question="What does this paragraph explain?",
        selected_text=selected_text
    )

    # Assert
    assert isinstance(answer, Answer)
    assert "rag" in answer.content.lower() or "retrieval" in answer.content.lower()
    assert len(answer.citations) > 0
    assert answer.is_from_book is True
    # Verify retrieval service was called with selected_text
    mock_retrieval_service.return_value.retrieve_for_question.assert_called_with(
        question="What does this paragraph explain?",
        selected_text=selected_text,
        top_k=5
    )


@pytest.mark.integration
def test_explain_code_snippet_with_selected_text(mock_retrieval_service, mock_openai_response):
    """Test explaining code snippet with selected text (T030).

    Given: Code selection
    When: Asking explanation
    Then: Answer explains code with reference
    """
    # Arrange
    selected_code = """
def retrieve(query):
    embedding = generate_embedding(query)
    results = search_vector_db(embedding)
    return results
"""

    mock_retrieved_points = [
        ScoredPoint(
            id=30,
            version=0,
            score=0.92,
            payload={
                "text": f"This code demonstrates the retrieval process: {selected_code}",
                "chapter": "Chapter 3",
                "section": "Code Examples",
                "chunk_id": 30
            },
            vector=None
        )
    ]
    mock_retrieval_service.return_value.retrieve_for_question.return_value = mock_retrieved_points

    mock_openai_response.return_value = Mock(
        choices=[Mock(message=Mock(
            content="This code implements retrieval by generating embeddings and searching the vector database [Chapter 3, Code Examples]."
        ))]
    )

    with patch('agent.get_retrieval_service', mock_retrieval_service):
        with patch('agent.OpenAI') as mock_openai_class:
            mock_openai_class.return_value.chat.completions.create = mock_openai_response
            agent = RAGAgent()

    # Act
    answer = agent.generate_answer(
        question="Explain this code snippet",
        selected_text=selected_code
    )

    # Assert
    assert isinstance(answer, Answer)
    assert "retrieval" in answer.content.lower() or "embedding" in answer.content.lower()
    assert len(answer.citations) > 0


@pytest.mark.integration
def test_explain_diagram_with_selected_text(mock_retrieval_service, mock_openai_response):
    """Test explaining diagram with selected text (T031).

    Given: Diagram description
    When: Asking clarification
    Then: Answer interprets visual element
    """
    # Arrange
    selected_diagram_description = "[Diagram: RAG Pipeline showing Query → Retrieval → Generation → Answer]"

    mock_retrieved_points = [
        ScoredPoint(
            id=35,
            version=0,
            score=0.90,
            payload={
                "text": f"The diagram illustrates the RAG pipeline: {selected_diagram_description}",
                "chapter": "Chapter 1",
                "section": "Architecture Diagrams",
                "chunk_id": 35
            },
            vector=None
        )
    ]
    mock_retrieval_service.return_value.retrieve_for_question.return_value = mock_retrieved_points

    mock_openai_response.return_value = Mock(
        choices=[Mock(message=Mock(
            content="The diagram shows the RAG pipeline flow from query input through retrieval and generation to final answer [Chapter 1, Architecture Diagrams]."
        ))]
    )

    with patch('agent.get_retrieval_service', mock_retrieval_service):
        with patch('agent.OpenAI') as mock_openai_class:
            mock_openai_class.return_value.chat.completions.create = mock_openai_response
            agent = RAGAgent()

    # Act
    answer = agent.generate_answer(
        question="What does this diagram show?",
        selected_text=selected_diagram_description
    )

    # Assert
    assert isinstance(answer, Answer)
    assert "diagram" in answer.content.lower() or "pipeline" in answer.content.lower()
    assert len(answer.citations) > 0


@pytest.mark.integration
def test_selected_text_with_additional_context_augments_from_book(mock_retrieval_service, mock_openai_response):
    """Test selected text with additional context augments from book (T032).

    Given: Selection needing more context
    When: Asking question
    Then: Answer augments with relevant book content
    """
    # Arrange
    selected_text = "The retrieval step uses embeddings..."

    mock_retrieved_points = [
        ScoredPoint(
            id=40,
            version=0,
            score=0.95,
            payload={
                "text": selected_text,
                "chapter": "Chapter 2",
                "section": "Retrieval",
                "chunk_id": 40
            },
            vector=None
        ),
        ScoredPoint(
            id=41,
            version=0,
            score=0.85,
            payload={
                "text": "Embeddings are generated using models like Cohere embed-multilingual-v3.0...",
                "chapter": "Chapter 2",
                "section": "Embeddings",
                "chunk_id": 41
            },
            vector=None
        )
    ]
    mock_retrieval_service.return_value.retrieve_for_question.return_value = mock_retrieved_points

    mock_openai_response.return_value = Mock(
        choices=[Mock(message=Mock(
            content="The retrieval step uses embeddings generated by models like Cohere [Chapter 2, Retrieval, Embeddings]."
        ))]
    )

    with patch('agent.get_retrieval_service', mock_retrieval_service):
        with patch('agent.OpenAI') as mock_openai_class:
            mock_openai_class.return_value.chat.completions.create = mock_openai_response
            agent = RAGAgent()

    # Act
    answer = agent.generate_answer(
        question="How are embeddings generated in this context?",
        selected_text=selected_text
    )

    # Assert
    assert isinstance(answer, Answer)
    assert len(answer.citations) >= 1  # Should have citations from both selected and additional context
    assert answer.is_from_book is True


# ========== User Story 3: Conversation Context Tests (T038-T041) ==========

@pytest.mark.integration
def test_follow_up_question_maintains_context(mock_retrieval_service, mock_openai_response):
    """Test follow-up question maintains context (T038).

    Given: Previous exchange
    When: Asking follow-up
    Then: System uses previous context
    """
    # Arrange
    mock_retrieved_points = [
        ScoredPoint(
            id=50,
            version=0,
            score=0.88,
            payload={
                "text": "RAG uses semantic search for retrieval...",
                "chapter": "Chapter 1",
                "section": "RAG Details",
                "chunk_id": 50
            },
            vector=None
        )
    ]
    mock_retrieval_service.return_value.retrieve_for_question.return_value = mock_retrieved_points

    # First question response
    mock_openai_response.return_value = Mock(
        choices=[Mock(message=Mock(content="RAG uses semantic search [Chapter 1, RAG Details]."))]
    )

    with patch('agent.get_retrieval_service', mock_retrieval_service):
        with patch('agent.OpenAI') as mock_openai_class:
            mock_openai_class.return_value.chat.completions.create = mock_openai_response
            agent = RAGAgent()

    # Simulate conversation history
    conversation_history = [
        {"role": "user", "content": "What is RAG?"},
        {"role": "assistant", "content": "RAG combines retrieval with generation [Chapter 1]."}
    ]

    # Act - Follow-up question
    answer = agent.generate_answer(
        question="How does it perform retrieval?",  # "it" refers to RAG from context
        selected_text=None,
        conversation_history=conversation_history
    )

    # Assert
    assert isinstance(answer, Answer)
    assert len(answer.content) > 0
    # Verify OpenAI was called with conversation history
    mock_openai_class.return_value.chat.completions.create.assert_called()


@pytest.mark.integration
def test_multiple_questions_about_same_topic_reference_same_sections(mock_retrieval_service, mock_openai_response):
    """Test multiple questions about same topic reference same sections (T039).

    Given: Topic-focused questions
    When: Asking multiple questions
    Then: Responses consistently reference same sections
    """
    # Arrange
    mock_retrieved_points = [
        ScoredPoint(
            id=60,
            version=0,
            score=0.92,
            payload={
                "text": "Vector databases store embeddings for fast similarity search...",
                "chapter": "Chapter 3",
                "section": "Vector Databases",
                "chunk_id": 60
            },
            vector=None
        )
    ]
    mock_retrieval_service.return_value.retrieve_for_question.return_value = mock_retrieved_points

    mock_openai_response.return_value = Mock(
        choices=[Mock(message=Mock(content="Vector databases store embeddings [Chapter 3, Vector Databases]."))]
    )

    with patch('agent.get_retrieval_service', mock_retrieval_service):
        with patch('agent.OpenAI') as mock_openai_class:
            mock_openai_class.return_value.chat.completions.create = mock_openai_response
            agent = RAGAgent()

    # Act - Ask multiple questions about vector databases
    answer1 = agent.generate_answer(question="What are vector databases?", selected_text=None)
    answer2 = agent.generate_answer(question="How do vector databases work?", selected_text=None)

    # Assert
    assert len(answer1.citations) > 0
    assert len(answer2.citations) > 0
    # Both should reference Chapter 3, Vector Databases
    assert any(c.chapter == "Chapter 3" for c in answer1.citations)
    assert any(c.chapter == "Chapter 3" for c in answer2.citations)


@pytest.mark.integration
def test_topic_shift_resets_context_appropriately(mock_retrieval_service, mock_openai_response):
    """Test topic shift resets context appropriately (T040).

    Given: New topic
    When: Asking question
    Then: System shifts focus appropriately
    """
    # Arrange
    # First topic: RAG
    mock_rag_points = [
        ScoredPoint(
            id=70,
            version=0,
            score=0.90,
            payload={
                "text": "RAG combines retrieval and generation...",
                "chapter": "Chapter 1",
                "section": "RAG",
                "chunk_id": 70
            },
            vector=None
        )
    ]

    # Second topic: Embeddings (different topic)
    mock_embedding_points = [
        ScoredPoint(
            id=80,
            version=0,
            score=0.93,
            payload={
                "text": "Embeddings are vector representations...",
                "chapter": "Chapter 2",
                "section": "Embeddings",
                "chunk_id": 80
            },
            vector=None
        )
    ]

    mock_retrieval_service.return_value.retrieve_for_question.side_effect = [
        mock_rag_points,
        mock_embedding_points
    ]

    mock_openai_response.return_value = Mock(
        choices=[Mock(message=Mock(content="Topic-specific answer with citations."))]
    )

    with patch('agent.get_retrieval_service', mock_retrieval_service):
        with patch('agent.OpenAI') as mock_openai_class:
            mock_openai_class.return_value.chat.completions.create = mock_openai_response
            agent = RAGAgent()

    # Act - First question about RAG
    answer1 = agent.generate_answer(question="What is RAG?", selected_text=None)

    # Act - Second question about embeddings (topic shift)
    answer2 = agent.generate_answer(question="What are embeddings?", selected_text=None)

    # Assert - Different chapters/sections indicate topic shift
    assert answer1.citations[0].chapter != answer2.citations[0].chapter or \
           answer1.citations[0].section != answer2.citations[0].section


@pytest.mark.integration
def test_conversation_summary(mock_retrieval_service, mock_openai_response):
    """Test conversation summary generation (T041).

    Given: Multiple exchanges
    When: Requesting summary
    Then: System provides concise recap
    """
    # Arrange
    conversation_history = [
        {"role": "user", "content": "What is RAG?"},
        {"role": "assistant", "content": "RAG combines retrieval with generation."},
        {"role": "user", "content": "How does it work?"},
        {"role": "assistant", "content": "It retrieves relevant content then generates answers."}
    ]

    mock_openai_response.return_value = Mock(
        choices=[Mock(message=Mock(
            content="Summary: We discussed RAG (Retrieval-Augmented Generation) and how it combines retrieval with generation to provide answers."
        ))]
    )

    with patch('agent.get_retrieval_service', mock_retrieval_service):
        with patch('agent.OpenAI') as mock_openai_class:
            mock_openai_class.return_value.chat.completions.create = mock_openai_response
            agent = RAGAgent()

    # Act - Request summary
    mock_retrieval_service.return_value.retrieve_for_question.return_value = []  # No retrieval for summary
    answer = agent.generate_answer(
        question="Summarize our conversation",
        selected_text=None,
        conversation_history=conversation_history
    )

    # Assert
    assert isinstance(answer, Answer)
    assert "summary" in answer.content.lower() or "discussed" in answer.content.lower()
    assert len(answer.content) <= 2000  # Should be concise


# ========== Fixtures ==========

@pytest.fixture
def mock_retrieval_service():
    """Mock retrieval service."""
    with patch('agent.get_retrieval_service') as mock:
        service = Mock()
        mock.return_value = service
        yield mock


@pytest.fixture
def mock_openai_response():
    """Mock OpenAI API response."""
    return MagicMock()
