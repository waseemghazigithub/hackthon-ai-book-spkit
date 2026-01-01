"""Book content ingestion script.

This module parses book markdown content, generates embeddings,
and indexes content into Qdrant for retrieval.
"""
import os
import argparse
import logging
from typing import List, Dict, Any
from pathlib import Path

from qdrant_client import QdrantClient, models
from cohere import Client

from .services.embedding_service import get_embedding_service
from .services.qdrant_service import get_qdrant_service
from .config import settings, get_settings, setup_logging

logger = logging.getLogger(__name__)


class BookContentIngester:
    """Ingester for book content indexing."""

    def __init__(self, settings_instance=None):
        """Initialize book content ingester.

        Args:
            settings_instance: Optional settings instance (uses global if None)
        """
        self.settings = settings_instance or get_settings()
        self.embedding_service = get_embedding_service()
        self.qdrant_service = get_qdrant_service()

    def parse_book_content(self, input_path: str) -> List[Dict[str, Any]]:
        """Parse markdown book content into chunks.

        Args:
            input_path: Path to book content directory

        Returns:
            List[Dict]: List of chunk dicts with text and metadata

        Example:
            >>> ingester = BookContentIngester()
            >>> chunks = ingester.parse_book_content("../book-content")
            >>> print(len(chunks))
            2485
        """
        input_dir = Path(input_path)
        if not input_dir.exists():
            raise FileNotFoundError(f"Book content path not found: {input_path}")

        # Find all markdown files
        md_files = list(input_dir.glob("**/*.md"))
        md_files.sort()  # Sort by filename

        logger.info(f"Found {len(md_files)} markdown files in {input_path}")

        all_chunks = []

        for md_file in md_files:
            try:
                file_chunks = self._parse_markdown_file(md_file)
                all_chunks.extend(file_chunks)
                logger.debug(f"Parsed {len(file_chunks)} chunks from {md_file.name}")
            except Exception as e:
                logger.error(f"Failed to parse {md_file}: {e}")

        logger.info(f"Total chunks parsed: {len(all_chunks)}")
        return all_chunks

    def _parse_markdown_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """Parse a single markdown file into chunks.

        Args:
            file_path: Path to markdown file

        Returns:
            List[Dict]: List of chunk dicts with text and metadata
        """
        # Extract chapter name from filename
        chapter_name = file_path.stem  # e.g., "chapter-1.md" -> "chapter-1"

        # Try to extract chapter number
        chapter_num_match = re.search(r"\d+", chapter_name)
        if chapter_num_match:
            chapter = f"Chapter {chapter_num_match.group()}"
        else:
            chapter = chapter_name.replace("-", " ").title()

        # Read file content
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Split by paragraphs and chunk
        paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
        chunks = self._create_chunks(paragraphs, chapter)

        return chunks

    def _create_chunks(
        self,
        paragraphs: List[str],
        chapter: str
    ) -> List[Dict[str, Any]]:
        """Create text chunks from paragraphs.

        Args:
            paragraphs: List of paragraphs
            chapter: Chapter name

        Returns:
            List[Dict]: List of chunk dicts
        """
        chunks = []
        current_chunk = ""
        current_section = None
        chunk_id = 0

        for para in paragraphs:
            # Detect section headers
            if para.startswith("#"):
                # Save previous chunk if exists
                if current_chunk:
                    chunks.append(self._create_chunk_dict(
                        current_chunk,
                        chapter,
                        current_section,
                        chunk_id
                    ))
                    chunk_id += 1

                # Extract section name
                header_match = re.match(r"^#+\s+(.+)$", para)
                if header_match:
                    current_section = header_match.group(1).strip()
                    # Clean markdown markers
                    current_section = re.sub(r"^#+\s*", "", current_section).strip()
                else:
                    current_section = None

                # Start new chunk
                current_chunk = ""
                continue

            # Add paragraph to current chunk
            current_chunk += "\n" + para

        # Don't forget the last chunk
        if current_chunk:
            chunks.append(self._create_chunk_dict(
                current_chunk,
                chapter,
                current_section,
                chunk_id
            ))

        return chunks

    def _create_chunk_dict(
        self,
        text: str,
        chapter: str,
        section: str,
        chunk_id: int
    ) -> Dict[str, Any]:
        """Create chunk dict for Qdrant.

        Args:
            text: Chunk text
            chapter: Chapter name
            section: Section name
            chunk_id: Unique chunk identifier

        Returns:
            Dict: Chunk dict with text, metadata, and chunk_id
        """
        return {
            "text": text.strip(),
            "metadata": {
                "chapter": chapter,
                "section": section or "",
                "chunk_id": chunk_id
            },
            "chunk_id": chunk_id
        }

    def generate_embeddings_and_index(
        self,
        chunks: List[Dict[str, Any]],
        collection_name: str
    ) -> None:
        """Generate embeddings and index to Qdrant.

        Args:
            chunks: List of chunk dicts
            collection_name: Qdrant collection name
        """
        # Ensure collection exists
        self.qdrant_service.ensure_collection_exists()
        logger.info(f"Ensured collection: {collection_name}")

        # Extract texts for embedding
        texts = [chunk["text"] for chunk in chunks]
        logger.info(f"Generating embeddings for {len(texts)} chunks...")

        # Generate embeddings in batches
        embeddings = self.embedding_service.generate_embeddings(
            texts=texts,
            batch_size=self.settings.embedding_batch_size
        )

        if len(embeddings) != len(chunks):
            logger.error(
                f"Embedding count mismatch: {len(embeddings)} != {len(chunks)}"
            )
            raise Exception("Embedding generation failed")

        # Prepare Qdrant points
        points = []
        for chunk, embedding in zip(chunks, embeddings):
            point = models.PointStruct(
                id=chunk["chunk_id"],
                vector=embedding,
                payload=chunk["metadata"]
            )
            points.append(point)

        # Index to Qdrant
        logger.info(f"Indexing {len(points)} points to {collection_name}...")
        self.qdrant_service.upsert_points(points)
        logger.info("Indexing complete!")


def main():
    """Main ingestion script entry point."""
    # Setup logging
    setup_logging()

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Ingest book content into Qdrant vector database"
    )

    parser.add_argument(
        "--input-path",
        type=str,
        default=get_settings().ingest_input_path,
        help="Path to book markdown files"
    )

    parser.add_argument(
        "--collection-name",
        type=str,
        default=get_settings().ingest_collection_name,
        help="Qdrant collection name"
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=get_settings().chunk_size_tokens,
        help=f"Tokens per chunk (default: {get_settings().chunk_size_tokens})"
    )

    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=get_settings().chunk_overlap_tokens,
        help=f"Token overlap between chunks (default: {get_settings().chunk_overlap_tokens})"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=get_settings().embedding_batch_size,
        help=f"Embeddings per API call (default: {get_settings().embedding_batch_size})"
    )

    args = parser.parse_args()

    logger.info("Starting book content ingestion...")
    logger.info(f"Input path: {args.input_path}")
    logger.info(f"Collection name: {args.collection_name}")
    logger.info(f"Chunk size: {args.chunk_size} tokens")
    logger.info(f"Chunk overlap: {args.chunk_overlap} tokens")
    logger.info(f"Embedding batch size: {args.batch_size}")

    try:
        # Create ingester
        ingester = BookContentIngester()

        # Parse book content
        logger.info("Parsing book content...")
        chunks = ingester.parse_book_content(args.input_path)

        if not chunks:
            logger.error("No chunks found to ingest!")
            return 1

        # Generate embeddings and index
        logger.info("Generating embeddings and indexing...")
        import time
        start_time = time.time()

        ingester.generate_embeddings_and_index(chunks, args.collection_name)

        elapsed_time = time.time() - start_time

        # Report summary
        logger.info("=" * 50)
        logger.info("INGESTION COMPLETE")
        logger.info("=" * 50)
        logger.info(f"Chunks indexed: {len(chunks)}")
        logger.info(f"Collection: {args.collection_name}")
        logger.info(f"Time taken: {elapsed_time:.2f} seconds ({elapsed_time/60:.1f} minutes)")
        logger.info("=" * 50)

        return 0

    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        logger.exception("Full exception details:")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
