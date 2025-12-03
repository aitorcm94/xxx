import os
import sys
import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import structlog
import click
from pydantic import BaseModel
import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_text_splitters import MarkdownHeaderTextSplitter

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

class ChunkMetadata(BaseModel):
    chunk_id: str
    document_id: str
    chunk_index: int
    page_numbers: List[int]
    section_title: Optional[str] = None
    section_level: Optional[int] = None
    chunk_type: str  # 'paragraph', 'heading', 'table', 'list'
    start_char: int
    end_char: int
    token_count: int
    word_count: int
    contains_tables: bool = False
    heading_hierarchy: List[str] = []

class DocumentChunk(BaseModel):
    metadata: ChunkMetadata
    content: str
    raw_content: str  # Original content before processing
    context: Optional[str] = None  # Surrounding context for better understanding

class ChunkedDocument(BaseModel):
    document_id: str
    total_chunks: int
    total_tokens: int
    total_words: int
    chunking_strategy: str
    processing_timestamp: str
    chunks: List[DocumentChunk]
    document_metadata: Dict[str, Any]

class SemanticTextChunker:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200, model_name: str = "gpt-4"):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model_name = model_name

        # Initialize tokenizer
        try:
            self.tokenizer = tiktoken.encoding_for_model(model_name)
        except:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")

        # Initialize text splitters
        self.recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=self._count_tokens,
            separators=["\n\n", "\n", ". ", " ", ""],
            keep_separator=True
        )

        # Markdown header splitter for structured documents
        self.header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "Header 1"),
                ("##", "Header 2"),
                ("###", "Header 3"),
                ("####", "Header 4"),
            ]
        )

        logger.info("Semantic text chunker initialized",
                   chunk_size=chunk_size,
                   chunk_overlap=chunk_overlap,
                   model=model_name)

    def _count_tokens(self, text: str) -> int:
        """Count tokens using the model's tokenizer"""
        return len(self.tokenizer.encode(text))

    def _extract_heading_hierarchy(self, structure: Dict[str, Any]) -> Dict[int, List[str]]:
        """Extract heading hierarchy from document structure"""
        page_headings = {}

        for page in structure.get("pages", []):
            page_num = page.get("page_number", 1)
            headings = []

            for element in page.get("elements", []):
                if element.get("type", "").lower() in ["heading", "title", "header"]:
                    heading_text = element.get("text", "").strip()
                    if heading_text:
                        headings.append(heading_text)

            page_headings[page_num] = headings

        return page_headings

    def _determine_chunk_type(self, content: str, element_info: Dict[str, Any] = None) -> str:
        """Determine the type of content chunk"""
        if element_info:
            element_type = element_info.get("type", "").lower()
            if "heading" in element_type or "title" in element_type:
                return "heading"
            elif "table" in element_type:
                return "table"
            elif "list" in element_type:
                return "list"

        # Fallback: analyze content patterns
        content_lower = content.lower().strip()

        if re.match(r'^#+\s', content) or content_lower.startswith(("chapter", "section", "article")):
            return "heading"
        elif "|" in content and content.count("|") > 2:
            return "table"
        elif re.match(r'^\s*[-*•]\s', content) or re.match(r'^\s*\d+\.\s', content):
            return "list"
        else:
            return "paragraph"

    def _find_section_context(self, chunk_start: int, full_text: str, page_headings: Dict[int, List[str]]) -> Tuple[Optional[str], List[str]]:
        """Find the section title and heading hierarchy for a chunk"""
        # Find the last heading before this chunk
        text_before = full_text[:chunk_start]

        # Look for markdown-style headings
        heading_pattern = r'^(#{1,6})\s+(.+)$'
        headings = []
        current_section = None

        for line in text_before.split('\n'):
            match = re.match(heading_pattern, line.strip())
            if match:
                level = len(match.group(1))
                title = match.group(2).strip()

                # Update hierarchy
                if level == 1:
                    headings = [title]
                elif level == 2 and len(headings) >= 1:
                    headings = headings[:1] + [title]
                elif level == 3 and len(headings) >= 2:
                    headings = headings[:2] + [title]
                elif level == 4 and len(headings) >= 3:
                    headings = headings[:3] + [title]

                current_section = title

        return current_section, headings

    def _create_chunks_from_structure(self, parsed_content: Dict[str, Any]) -> List[DocumentChunk]:
        """Create chunks while preserving document structure"""
        chunks = []
        document_id = parsed_content["metadata"]["document_id"]
        full_text = parsed_content["text"]
        structure = parsed_content["structure"]

        # Extract heading hierarchy
        page_headings = self._extract_heading_hierarchy(structure)

        # Process each page
        chunk_index = 0
        current_position = 0

        for page in structure.get("pages", []):
            page_num = page.get("page_number", 1)
            page_text = page.get("text", "")

            if not page_text.strip():
                continue

            # Split page text into manageable chunks
            page_chunks = self.recursive_splitter.split_text(page_text)

            for chunk_text in page_chunks:
                if not chunk_text.strip():
                    continue

                # Find chunk position in full text
                chunk_start = full_text.find(chunk_text, current_position)
                if chunk_start == -1:
                    chunk_start = current_position

                chunk_end = chunk_start + len(chunk_text)
                current_position = chunk_end

                # Determine section context
                section_title, heading_hierarchy = self._find_section_context(
                    chunk_start, full_text, page_headings
                )

                # Determine chunk type
                chunk_type = self._determine_chunk_type(chunk_text)

                # Check for tables
                contains_tables = any(
                    table.get("page_number") == page_num
                    for table in structure.get("tables", [])
                )

                # Create chunk metadata
                metadata = ChunkMetadata(
                    chunk_id=f"{document_id}-chunk-{chunk_index:04d}",
                    document_id=document_id,
                    chunk_index=chunk_index,
                    page_numbers=[page_num],
                    section_title=section_title,
                    section_level=len(heading_hierarchy) if heading_hierarchy else None,
                    chunk_type=chunk_type,
                    start_char=chunk_start,
                    end_char=chunk_end,
                    token_count=self._count_tokens(chunk_text),
                    word_count=len(chunk_text.split()),
                    contains_tables=contains_tables,
                    heading_hierarchy=heading_hierarchy
                )

                # Add context from surrounding text
                context_start = max(0, chunk_start - 200)
                context_end = min(len(full_text), chunk_end + 200)
                context = full_text[context_start:context_end]

                # Create document chunk
                chunk = DocumentChunk(
                    metadata=metadata,
                    content=chunk_text.strip(),
                    raw_content=chunk_text,
                    context=context if context != chunk_text else None
                )

                chunks.append(chunk)
                chunk_index += 1

        return chunks

    def _enhance_chunks_with_tables(self, chunks: List[DocumentChunk], structure: Dict[str, Any]) -> List[DocumentChunk]:
        """Enhance chunks with table information"""
        tables = structure.get("tables", [])

        for chunk in chunks:
            if chunk.metadata.contains_tables:
                # Find relevant tables for this chunk's pages
                relevant_tables = [
                    table for table in tables
                    if table.get("page_number") in chunk.metadata.page_numbers
                ]

                if relevant_tables:
                    # Add table context to chunk content
                    table_summaries = []
                    for table in relevant_tables:
                        caption = table.get("caption", "")
                        data = table.get("data", [])

                        if caption:
                            table_summaries.append(f"Table: {caption}")

                        if data and len(data) > 0:
                            # Add first few rows as context
                            headers = list(data[0].keys()) if data else []
                            if headers:
                                table_summaries.append(f"Columns: {', '.join(headers)}")

                    if table_summaries:
                        chunk.content += "\n\n" + "\n".join(table_summaries)

        return chunks

    def chunk_document(self, parsed_content: Dict[str, Any]) -> ChunkedDocument:
        """Main method to chunk a parsed document"""
        logger.info("Starting document chunking",
                   document_id=parsed_content["metadata"]["document_id"])

        start_time = datetime.utcnow()

        try:
            # Create chunks from document structure
            chunks = self._create_chunks_from_structure(parsed_content)

            # Enhance chunks with table information
            chunks = self._enhance_chunks_with_tables(chunks, parsed_content["structure"])

            # Calculate totals
            total_tokens = sum(chunk.metadata.token_count for chunk in chunks)
            total_words = sum(chunk.metadata.word_count for chunk in chunks)

            # Create chunked document
            chunked_doc = ChunkedDocument(
                document_id=parsed_content["metadata"]["document_id"],
                total_chunks=len(chunks),
                total_tokens=total_tokens,
                total_words=total_words,
                chunking_strategy="semantic_structure_aware",
                processing_timestamp=datetime.utcnow().isoformat(),
                chunks=chunks,
                document_metadata=parsed_content["metadata"]
            )

            processing_time = (datetime.utcnow() - start_time).total_seconds()

            logger.info("Document chunking completed successfully",
                       document_id=parsed_content["metadata"]["document_id"],
                       total_chunks=len(chunks),
                       total_tokens=total_tokens,
                       avg_tokens_per_chunk=total_tokens / len(chunks) if chunks else 0,
                       processing_time_seconds=processing_time)

            return chunked_doc

        except Exception as e:
            logger.error("Document chunking failed",
                        document_id=parsed_content.get("metadata", {}).get("document_id", "unknown"),
                        error=str(e),
                        exc_info=True)
            raise

@click.command()
@click.option('--document-id', required=True, help='Document ID')
@click.option('--parsed-content', required=True, help='Path to parsed content JSON file')
@click.option('--output-dir', default='/tmp/outputs', help='Output directory')
@click.option('--chunk-size', default=1000, help='Maximum chunk size in tokens')
@click.option('--chunk-overlap', default=200, help='Overlap between chunks in tokens')
def main(document_id: str, parsed_content: str, output_dir: str, chunk_size: int, chunk_overlap: int):
    """Text Chunker - Second step in the ASD compliance pipeline"""

    try:
        logger.info("Starting text chunker job",
                   document_id=document_id,
                   parsed_content_path=parsed_content,
                   chunk_size=chunk_size,
                   chunk_overlap=chunk_overlap)

        # Load parsed content
        if not os.path.exists(parsed_content):
            raise FileNotFoundError(f"Parsed content file not found: {parsed_content}")

        with open(parsed_content, 'r', encoding='utf-8') as f:
            parsed_doc = json.load(f)

        # Validate document ID matches
        if parsed_doc.get("metadata", {}).get("document_id") != document_id:
            logger.warning("Document ID mismatch",
                          expected=document_id,
                          found=parsed_doc.get("metadata", {}).get("document_id"))

        # Initialize chunker
        chunker = SemanticTextChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        # Chunk the document
        chunked_document = chunker.chunk_document(parsed_doc)

        # Ensure output directory exists
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save chunked document
        output_file = output_path / "chunks.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(chunked_document.model_dump_json(indent=2))

        logger.info("Text chunker job completed successfully",
                   document_id=document_id,
                   total_chunks=chunked_document.total_chunks,
                   total_tokens=chunked_document.total_tokens,
                   output_file=str(output_file))

    except Exception as e:
        logger.error("Text chunker job failed",
                    document_id=document_id,
                    error=str(e),
                    exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()