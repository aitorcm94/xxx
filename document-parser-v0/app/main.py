import os
import sys
import json
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import structlog
import click
from azure.storage.blob import BlobServiceClient
from azure.identity import DefaultAzureCredential
from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from pydantic import BaseModel

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

class DocumentMetadata(BaseModel):
    document_id: str
    filename: str
    file_size: int
    content_type: str
    processing_started_at: str
    processing_completed_at: Optional[str] = None
    page_count: Optional[int] = None
    word_count: Optional[int] = None
    language: Optional[str] = None
    error: Optional[str] = None

class ParsedContent(BaseModel):
    metadata: DocumentMetadata
    content: Dict[str, Any]
    text: str
    structure: Dict[str, Any]

class DocumentParser:
    def __init__(self):
        self.storage_account_name = os.getenv("AZURE_STORAGE_ACCOUNT_NAME")
        self.container_name = os.getenv("AZURE_STORAGE_CONTAINER_NAME")

        if not self.storage_account_name:
            raise ValueError("AZURE_STORAGE_ACCOUNT_NAME environment variable is required")

        if not self.container_name:
            raise ValueError("AZURE_STORAGE_CONTAINER_NAME environment variable is required")

        self.credential = DefaultAzureCredential()
        self.blob_service_client = BlobServiceClient(
            account_url=f"https://{self.storage_account_name}.blob.core.windows.net",
            credential=self.credential
        )

        # Configure Docling with optimized settings for ASD documents
        self.pipeline_options = PdfPipelineOptions(
            do_ocr=True,  # Enable OCR for scanned documents
            do_table_structure=True,  # Extract table structures
            table_structure_options={
                "do_cell_matching": True,
                "mode": "accurate"  # Use accurate mode for compliance documents
            },
            images_scale=2.0,  # Higher resolution for better OCR
            generate_page_images=False,  # Skip page images to save memory
        )

        # Initialize document converter
        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: self.pipeline_options,
            }
        )

        logger.info("Document parser initialized",
                   storage_account=self.storage_account_name,
                   container=self.container_name)

    def download_document(self, blob_name: str, local_path: Path) -> None:
        """Download document from Azure Blob Storage"""
        blob_client = self.blob_service_client.get_blob_client(
            container=self.container_name,
            blob=blob_name
        )

        logger.info("Downloading document", blob_name=blob_name, local_path=str(local_path))

        with open(local_path, "wb") as download_file:
            download_file.write(blob_client.download_blob().readall())

        logger.info("Document downloaded successfully", blob_name=blob_name)

    def parse_document(self, document_path: Path, document_id: str, filename: str) -> ParsedContent:
        """Parse document using Docling"""
        logger.info("Starting document parsing",
                   document_id=document_id,
                   filename=filename,
                   file_size=document_path.stat().st_size)

        processing_started_at = datetime.utcnow().isoformat()

        try:
            # Convert document using Docling
            result = self.converter.convert(document_path)

            # Extract text content
            full_text = result.document.export_to_text()

            # Extract structured content
            structured_content = {
                "title": getattr(result.document, 'title', filename),
                "pages": [],
                "tables": [],
                "figures": [],
                "sections": []
            }

            # Process pages and extract structure
            for page_num, page in enumerate(result.document.pages, 1):
                page_data = {
                    "page_number": page_num,
                    "text": page.export_to_text(),
                    "elements": []
                }

                # Extract page elements (headings, paragraphs, tables, etc.)
                for element in page._page_elements:
                    element_data = {
                        "type": type(element).__name__,
                        "text": getattr(element, 'text', ''),
                        "bbox": getattr(element, 'bbox', None)
                    }
                    page_data["elements"].append(element_data)

                structured_content["pages"].append(page_data)

            # Extract tables
            for table in result.document.tables:
                table_data = {
                    "caption": getattr(table, 'caption', ''),
                    "data": table.export_to_dataframe().to_dict('records') if hasattr(table, 'export_to_dataframe') else [],
                    "page_number": getattr(table, 'page_number', None)
                }
                structured_content["tables"].append(table_data)

            # Calculate statistics
            word_count = len(full_text.split())
            page_count = len(result.document.pages)

            processing_completed_at = datetime.utcnow().isoformat()

            metadata = DocumentMetadata(
                document_id=document_id,
                filename=filename,
                file_size=document_path.stat().st_size,
                content_type="application/pdf",  # Assume PDF for now
                processing_started_at=processing_started_at,
                processing_completed_at=processing_completed_at,
                page_count=page_count,
                word_count=word_count,
                language="en"  # Could be detected using language detection
            )

            parsed_content = ParsedContent(
                metadata=metadata,
                content=result.document.export_to_dict(),
                text=full_text,
                structure=structured_content
            )

            logger.info("Document parsing completed successfully",
                       document_id=document_id,
                       page_count=page_count,
                       word_count=word_count,
                       processing_time_ms=int((datetime.fromisoformat(processing_completed_at.replace('Z', '+00:00')) -
                                             datetime.fromisoformat(processing_started_at.replace('Z', '+00:00'))).total_seconds() * 1000))

            return parsed_content

        except Exception as e:
            error_msg = f"Failed to parse document: {str(e)}"
            logger.error("Document parsing failed",
                        document_id=document_id,
                        error=error_msg,
                        exc_info=True)

            metadata = DocumentMetadata(
                document_id=document_id,
                filename=filename,
                file_size=document_path.stat().st_size,
                content_type="application/pdf",
                processing_started_at=processing_started_at,
                error=error_msg
            )

            # Return empty content on error
            return ParsedContent(
                metadata=metadata,
                content={},
                text="",
                structure={}
            )

    def save_output(self, parsed_content: ParsedContent, output_path: Path) -> None:
        """Save parsed content to output file"""
        logger.info("Saving parsed content", output_path=str(output_path))

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(parsed_content.model_dump_json(indent=2))

        logger.info("Parsed content saved successfully", output_path=str(output_path))

@click.command()
@click.option('--document-id', required=True, help='Document ID')
@click.option('--blob-name', required=True, help='Azure Blob Storage path')
@click.option('--output-dir', default='/tmp/outputs', help='Output directory')
def main(document_id: str, blob_name: str, output_dir: str):
    """Document Parser - First step in the ASD compliance pipeline"""

    try:
        # Extract filename from blob name
        filename = Path(blob_name).name

        logger.info("Starting document parser job",
                   document_id=document_id,
                   blob_name=blob_name,
                   filename=filename)

        # Initialize parser
        parser = DocumentParser()

        # Create temporary directory for processing
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            document_path = temp_path / filename

            # Download document from Azure Blob Storage
            parser.download_document(blob_name, document_path)

            # Parse document using Docling
            parsed_content = parser.parse_document(document_path, document_id, filename)

            # Ensure output directory exists
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            # Save parsed content
            output_file = output_path / "parsed-content.json"
            parser.save_output(parsed_content, output_file)

            if parsed_content.metadata.error:
                logger.error("Document parsing completed with errors",
                           document_id=document_id,
                           error=parsed_content.metadata.error)
                sys.exit(1)
            else:
                logger.info("Document parsing job completed successfully",
                           document_id=document_id,
                           page_count=parsed_content.metadata.page_count,
                           word_count=parsed_content.metadata.word_count)

    except Exception as e:
        logger.error("Document parser job failed",
                    document_id=document_id,
                    error=str(e),
                    exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()