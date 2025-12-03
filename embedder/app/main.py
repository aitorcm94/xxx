import os
import sys
import json
import gc
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import structlog
import click
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from pydantic import BaseModel
from tqdm import tqdm
import psutil

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

class EmbeddingMetadata(BaseModel):
    chunk_id: str
    document_id: str
    model_name: str
    embedding_dimension: int
    processing_timestamp: str
    token_count: int
    chunk_type: str
    section_title: Optional[str] = None
    page_numbers: List[int] = []

class ChunkEmbedding(BaseModel):
    metadata: EmbeddingMetadata
    embedding: List[float]
    original_content: str
    content_hash: str

class EmbeddedDocument(BaseModel):
    document_id: str
    model_name: str
    embedding_dimension: int
    total_embeddings: int
    processing_timestamp: str
    embeddings: List[ChunkEmbedding]
    processing_stats: Dict[str, Any]

class MultiModelEmbedder:
    """
    Multi-model embedder optimized for compliance documents
    Supports multiple embedding models for different use cases
    """

    # Available models optimized for different purposes
    MODELS = {
        "general": "all-MiniLM-L6-v2",  # Fast, general purpose
        "large": "all-mpnet-base-v2",   # Better quality, slower
        "domain": "sentence-transformers/msmarco-bert-base-dot-v5",  # Document retrieval
        "multilingual": "paraphrase-multilingual-MiniLM-L12-v2"  # Multi-language support
    }

    def __init__(self, model_name: str = "general", batch_size: int = 32, use_gpu: bool = None):
        self.model_name = model_name
        self.batch_size = batch_size

        # Auto-detect GPU availability
        if use_gpu is None:
            self.use_gpu = torch.cuda.is_available()
        else:
            self.use_gpu = use_gpu and torch.cuda.is_available()

        # Get actual model name
        self.actual_model_name = self.MODELS.get(model_name, model_name)

        # Set device
        self.device = "cuda" if self.use_gpu else "cpu"

        # Initialize model
        self.model = None
        self.tokenizer = None
        self._load_model()

        logger.info("Multi-model embedder initialized",
                   model_name=self.actual_model_name,
                   device=self.device,
                   batch_size=batch_size,
                   gpu_available=torch.cuda.is_available(),
                   using_gpu=self.use_gpu)

    def _load_model(self):
        """Load the sentence transformer model"""
        try:
            logger.info("Loading embedding model", model=self.actual_model_name)

            # Load model with optimizations
            self.model = SentenceTransformer(
                self.actual_model_name,
                device=self.device
            )

            # Load tokenizer for token counting
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.actual_model_name)
            except:
                # Fallback tokenizer
                self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

            # Get embedding dimension
            self.embedding_dim = self.model.get_sentence_embedding_dimension()

            # Enable evaluation mode for inference
            self.model.eval()

            # Optimize for inference
            if self.use_gpu:
                self.model.half()  # Use FP16 for GPU inference
                torch.backends.cudnn.benchmark = True

            logger.info("Model loaded successfully",
                       embedding_dimension=self.embedding_dim,
                       model_device=str(self.model.device))

        except Exception as e:
            logger.error("Failed to load model", error=str(e), exc_info=True)
            raise

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        try:
            return len(self.tokenizer.encode(text, add_special_tokens=True))
        except:
            # Fallback: rough estimation
            return len(text.split()) * 1.3

    def _create_content_hash(self, content: str) -> str:
        """Create a hash of the content for deduplication"""
        import hashlib
        return hashlib.md5(content.encode('utf-8')).hexdigest()

    def _prepare_texts_for_embedding(self, chunks: List[Dict[str, Any]]) -> List[Tuple[str, Dict[str, Any]]]:
        """Prepare texts for embedding with metadata"""
        prepared_texts = []

        for chunk in chunks:
            content = chunk.get("content", "")
            metadata = chunk.get("metadata", {})

            # Skip empty content
            if not content.strip():
                logger.warning("Skipping empty chunk", chunk_id=metadata.get("chunk_id"))
                continue

            # Enhance content with context if available
            enhanced_content = content

            # Add section context if available
            if metadata.get("section_title"):
                enhanced_content = f"Section: {metadata['section_title']}\n\n{content}"

            # Add table context if chunk contains tables
            if metadata.get("contains_tables", False):
                enhanced_content += "\n[Contains tabular data]"

            prepared_texts.append((enhanced_content, chunk))

        return prepared_texts

    def _batch_embed_texts(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a batch of texts"""
        try:
            with torch.no_grad():
                embeddings = self.model.encode(
                    texts,
                    batch_size=self.batch_size,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=True  # L2 normalize for cosine similarity
                )
            return embeddings

        except Exception as e:
            logger.error("Batch embedding failed", error=str(e), batch_size=len(texts))
            raise

    def embed_chunks(self, chunked_document: Dict[str, Any]) -> EmbeddedDocument:
        """Generate embeddings for all chunks in a document"""
        document_id = chunked_document["document_id"]
        chunks = chunked_document["chunks"]

        logger.info("Starting embedding generation",
                   document_id=document_id,
                   total_chunks=len(chunks),
                   model=self.actual_model_name)

        start_time = datetime.utcnow()

        # Prepare texts for embedding
        prepared_texts = self._prepare_texts_for_embedding(chunks)

        if not prepared_texts:
            raise ValueError("No valid chunks found for embedding")

        # Extract texts and metadata
        texts = [text for text, _ in prepared_texts]
        chunk_metadata_list = [chunk for _, chunk in prepared_texts]

        # Generate embeddings in batches
        all_embeddings = []
        processed_chunks = 0

        logger.info("Processing embeddings in batches",
                   total_texts=len(texts),
                   batch_size=self.batch_size)

        # Process in batches with progress tracking
        for i in tqdm(range(0, len(texts), self.batch_size), desc="Generating embeddings"):
            batch_texts = texts[i:i + self.batch_size]
            batch_metadata = chunk_metadata_list[i:i + self.batch_size]

            # Generate embeddings for batch
            batch_embeddings = self._batch_embed_texts(batch_texts)

            # Create embedding objects
            for j, embedding in enumerate(batch_embeddings):
                chunk_data = batch_metadata[j]
                chunk_metadata = chunk_data.get("metadata", {})

                # Create embedding metadata
                embedding_metadata = EmbeddingMetadata(
                    chunk_id=chunk_metadata.get("chunk_id", f"unknown-{processed_chunks}"),
                    document_id=document_id,
                    model_name=self.actual_model_name,
                    embedding_dimension=len(embedding),
                    processing_timestamp=datetime.utcnow().isoformat(),
                    token_count=self._count_tokens(batch_texts[j]),
                    chunk_type=chunk_metadata.get("chunk_type", "paragraph"),
                    section_title=chunk_metadata.get("section_title"),
                    page_numbers=chunk_metadata.get("page_numbers", [])
                )

                # Create chunk embedding
                chunk_embedding = ChunkEmbedding(
                    metadata=embedding_metadata,
                    embedding=embedding.tolist(),
                    original_content=chunk_data.get("content", ""),
                    content_hash=self._create_content_hash(chunk_data.get("content", ""))
                )

                all_embeddings.append(chunk_embedding)
                processed_chunks += 1

            # Memory management
            if i % (self.batch_size * 4) == 0:  # Every 4 batches
                gc.collect()
                if self.use_gpu:
                    torch.cuda.empty_cache()

        # Calculate processing stats
        end_time = datetime.utcnow()
        processing_time = (end_time - start_time).total_seconds()

        processing_stats = {
            "processing_time_seconds": processing_time,
            "embeddings_per_second": len(all_embeddings) / processing_time if processing_time > 0 else 0,
            "average_tokens_per_chunk": sum(emb.metadata.token_count for emb in all_embeddings) / len(all_embeddings),
            "model_device": self.device,
            "batch_size": self.batch_size,
            "memory_usage_mb": psutil.Process().memory_info().rss / 1024 / 1024
        }

        # Create embedded document
        embedded_doc = EmbeddedDocument(
            document_id=document_id,
            model_name=self.actual_model_name,
            embedding_dimension=self.embedding_dim,
            total_embeddings=len(all_embeddings),
            processing_timestamp=end_time.isoformat(),
            embeddings=all_embeddings,
            processing_stats=processing_stats
        )

        logger.info("Embedding generation completed",
                   document_id=document_id,
                   total_embeddings=len(all_embeddings),
                   processing_time_seconds=processing_time,
                   embeddings_per_second=processing_stats["embeddings_per_second"],
                   memory_usage_mb=processing_stats["memory_usage_mb"])

        return embedded_doc

    def cleanup(self):
        """Clean up resources"""
        if self.model:
            del self.model
        if self.tokenizer:
            del self.tokenizer

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

@click.command()
@click.option('--document-id', required=True, help='Document ID')
@click.option('--chunks', required=True, help='Path to chunks JSON file')
@click.option('--output-dir', default='/tmp/outputs', help='Output directory')
@click.option('--model', default='general', help='Embedding model to use')
@click.option('--batch-size', default=16, help='Batch size for processing')
@click.option('--use-gpu/--no-gpu', default=None, help='Force GPU usage on/off')
def main(document_id: str, chunks: str, output_dir: str, model: str, batch_size: int, use_gpu: Optional[bool]):
    """Embedder - Third step in the ASD compliance pipeline"""

    try:
        logger.info("Starting embedder job",
                   document_id=document_id,
                   chunks_path=chunks,
                   model=model,
                   batch_size=batch_size,
                   gpu_available=torch.cuda.is_available())

        # Load chunks
        if not os.path.exists(chunks):
            raise FileNotFoundError(f"Chunks file not found: {chunks}")

        with open(chunks, 'r', encoding='utf-8') as f:
            chunked_document = json.load(f)

        # Validate document ID
        if chunked_document.get("document_id") != document_id:
            logger.warning("Document ID mismatch",
                          expected=document_id,
                          found=chunked_document.get("document_id"))

        # Initialize embedder
        embedder = MultiModelEmbedder(
            model_name=model,
            batch_size=batch_size,
            use_gpu=use_gpu
        )

        try:
            # Generate embeddings
            embedded_document = embedder.embed_chunks(chunked_document)

            # Ensure output directory exists
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            # Save embeddings
            output_file = output_path / "embeddings.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(embedded_document.model_dump_json(indent=2))

            logger.info("Embedder job completed successfully",
                       document_id=document_id,
                       total_embeddings=embedded_document.total_embeddings,
                       embedding_dimension=embedded_document.embedding_dimension,
                       model=embedded_document.model_name,
                       output_file=str(output_file))

        finally:
            # Cleanup resources
            embedder.cleanup()

    except Exception as e:
        logger.error("Embedder job failed",
                    document_id=document_id,
                    error=str(e),
                    exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()