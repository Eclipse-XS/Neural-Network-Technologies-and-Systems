"""Document ingestion and collection management routes."""
from typing import Optional
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from ...domain.exceptions import DocumentNotFoundError, DocumentValidationError, ExtractionError, IndexingError
from ..dependencies import get_ingestion_service
from ..schemas import DocumentListResponse, DocumentResponse

router = APIRouter(prefix="/api/v1/documents", tags=["documents"])


@router.post("", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(...),
    collection: str = "default",
    ingestion_service=Depends(get_ingestion_service),
):
    """Upload and index a technical document (Markdown, Plain text, or PDF)."""
    try:
        content = await file.read()
        doc = ingestion_service.ingest_file(
            filename=file.filename or "unknown.txt",
            content=content,
            collection=collection,
        )
        return DocumentResponse(**doc.to_dict())
    except DocumentValidationError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except ExtractionError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Ingestion failed: {exc}")


@router.get("", response_model=DocumentListResponse)
async def list_documents(
    collection: Optional[str] = None,
    ingestion_service=Depends(get_ingestion_service),
):
    """List all indexed documents and their status."""
    docs = ingestion_service.list_documents(collection=collection)
    return DocumentListResponse(
        documents=[DocumentResponse(**d.to_dict()) for d in docs],
        total=len(docs),
    )


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    document_id: str,
    ingestion_service=Depends(get_ingestion_service),
):
    """Delete document and its active indexed vector chunks."""
    try:
        ingestion_service.delete_document(document_id)
        return None
    except DocumentNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))


@router.post("/{document_id}/reindex", response_model=DocumentResponse)
async def reindex_document(
    document_id: str,
    ingestion_service=Depends(get_ingestion_service),
):
    """Reindex an existing document."""
    try:
        doc = ingestion_service.reindex_document(document_id)
    except DocumentNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except IndexingError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    return DocumentResponse(**doc.to_dict())
