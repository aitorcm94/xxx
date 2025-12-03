#!/usr/bin/env python3
"""
Simple test script to verify text-chunker integration with document parser output
"""

import json
import tempfile
from pathlib import Path
from app.main import SemanticTextChunker

def create_sample_parsed_content():
    """Create sample parsed content that mimics document parser output"""
    return {
        "metadata": {
            "document_id": "test-doc-001",
            "filename": "sample_asd.pdf",
            "file_size": 1024000,
            "content_type": "application/pdf",
            "processing_started_at": "2024-01-01T12:00:00Z",
            "processing_completed_at": "2024-01-01T12:05:00Z",
            "page_count": 3,
            "word_count": 500,
            "language": "en"
        },
        "content": {
            "title": "Architecture Specification Document",
            "pages": [
                {"page_number": 1, "text": "Page 1 content"},
                {"page_number": 2, "text": "Page 2 content"},
                {"page_number": 3, "text": "Page 3 content"}
            ]
        },
        "text": "# Architecture Specification Document\n\nThis document describes the system architecture for the compliance checker.\n\n## Section 1: Overview\n\nThe system consists of multiple microservices that work together to analyze documents for compliance.\n\n### 1.1 Components\n\nThe main components include:\n- API Gateway\n- Document Parser\n- Text Chunker\n\n## Section 2: Implementation\n\nThis section covers the technical implementation details.\n\n### 2.1 Technology Stack\n\nWe use the following technologies:\n- Python 3.11\n- FastAPI\n- Azure Services\n\n## Section 3: Compliance Analysis\n\nThe compliance analysis process involves several steps to ensure documents meet Orange Book standards.",
        "structure": {
            "title": "Architecture Specification Document",
            "pages": [
                {
                    "page_number": 1,
                    "text": "# Architecture Specification Document\n\nThis document describes the system architecture.",
                    "elements": [
                        {"type": "heading", "text": "Architecture Specification Document", "bbox": [0, 0, 100, 20]},
                        {"type": "paragraph", "text": "This document describes the system architecture.", "bbox": [0, 30, 100, 50]}
                    ]
                },
                {
                    "page_number": 2,
                    "text": "## Section 1: Overview\n\nThe system consists of multiple microservices.",
                    "elements": [
                        {"type": "heading", "text": "Section 1: Overview", "bbox": [0, 0, 100, 20]},
                        {"type": "paragraph", "text": "The system consists of multiple microservices.", "bbox": [0, 30, 100, 50]}
                    ]
                },
                {
                    "page_number": 3,
                    "text": "## Section 2: Implementation\n\nThis section covers technical details.",
                    "elements": [
                        {"type": "heading", "text": "Section 2: Implementation", "bbox": [0, 0, 100, 20]},
                        {"type": "paragraph", "text": "This section covers technical details.", "bbox": [0, 30, 100, 50]}
                    ]
                }
            ],
            "tables": [
                {
                    "caption": "Technology Stack",
                    "page_number": 2,
                    "data": [
                        {"Component": "API Gateway", "Technology": "FastAPI"},
                        {"Component": "Document Parser", "Technology": "Docling"},
                        {"Component": "Text Chunker", "Technology": "LangChain"}
                    ]
                }
            ],
            "figures": [],
            "sections": [
                {"title": "Overview", "level": 2, "page": 1},
                {"title": "Implementation", "level": 2, "page": 2},
                {"title": "Compliance Analysis", "level": 2, "page": 3}
            ]
        }
    }

def test_chunker():
    """Test the text chunker with sample data"""
    print("🧪 Testing Text Chunker Integration...")
    
    # Create sample data
    parsed_content = create_sample_parsed_content()
    
    # Initialize chunker
    chunker = SemanticTextChunker(
        chunk_size=500,  # Smaller chunks for testing
        chunk_overlap=100
    )
    
    try:
        # Chunk the document
        chunked_doc = chunker.chunk_document(parsed_content)
        
        print(f"✅ Chunking successful!")
        print(f"   Document ID: {chunked_doc.document_id}")
        print(f"   Total chunks: {chunked_doc.total_chunks}")
        print(f"   Total tokens: {chunked_doc.total_tokens}")
        print(f"   Total words: {chunked_doc.total_words}")
        print(f"   Strategy: {chunked_doc.chunking_strategy}")
        
        # Display chunk details
        print(f"\n📄 Chunk Details:")
        for i, chunk in enumerate(chunked_doc.chunks[:3]):  # Show first 3 chunks
            print(f"   Chunk {i+1}:")
            print(f"     ID: {chunk.metadata.chunk_id}")
            print(f"     Type: {chunk.metadata.chunk_type}")
            print(f"     Section: {chunk.metadata.section_title}")
            print(f"     Tokens: {chunk.metadata.token_count}")
            print(f"     Pages: {chunk.metadata.page_numbers}")
            print(f"     Content preview: {chunk.content[:100]}...")
            print()
        
        if chunked_doc.total_chunks > 3:
            print(f"   ... and {chunked_doc.total_chunks - 3} more chunks")
        
        # Test JSON serialization
        json_output = chunked_doc.model_dump_json(indent=2)
        print(f"✅ JSON serialization successful! ({len(json_output)} characters)")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_chunker()
    exit(0 if success else 1)