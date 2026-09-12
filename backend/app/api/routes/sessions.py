"""
Precious Edu LLM — Session API Routes

Endpoints for creating, listing, retrieving, and deleting chat sessions.
All business logic is delegated to SessionService.
"""

from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.api.dependencies import get_db
from app.schemas.session import (
    SessionCreate,
    SessionResponse,
    SessionDetailResponse,
    SessionListResponse,
)
from app.services.session_service import SessionService
from app.core.exceptions import SessionNotFoundError

router = APIRouter()


@router.post(
    "/sessions",
    response_model=SessionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create Chat Session",
    description="Creates a new chat conversation session."
)
async def create_session(
    payload: Optional[SessionCreate] = None,
    db=Depends(get_db)
):
    service = SessionService(db)
    title = payload.title if payload else "New Conversation"
    metadata = payload.metadata if payload else {}
    return await service.create_session(title=title, metadata=metadata)


@router.get(
    "/sessions",
    response_model=SessionListResponse,
    summary="List Chat Sessions",
    description="Returns list of all active chat sessions sorted by recent activity."
)
async def list_sessions(
    limit: int = Query(default=50, ge=1, le=100),
    skip: int = Query(default=0, ge=0),
    db=Depends(get_db)
):
    service = SessionService(db)
    return await service.list_sessions(limit=limit, skip=skip)


@router.get(
    "/sessions/{session_id}",
    response_model=SessionDetailResponse,
    summary="Get Chat Session Detail",
    description="Retrieves specific session details and its complete chat history."
)
async def get_session(
    session_id: str,
    db=Depends(get_db)
):
    service = SessionService(db)
    try:
        return await service.get_session_with_messages(session_id)
    except SessionNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=e.message)


@router.delete(
    "/sessions/{session_id}",
    status_code=status.HTTP_200_OK,
    summary="Delete Chat Session",
    description="Deletes a chat session and all associated messages."
)
async def delete_session(
    session_id: str,
    db=Depends(get_db)
):
    service = SessionService(db)
    try:
        await service.delete_session(session_id)
        return {"status": "success", "message": f"Session '{session_id}' and all messages deleted."}
    except SessionNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=e.message)
