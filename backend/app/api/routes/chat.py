"""
Precious Edu LLM — Chat API Routes

Endpoint for sending chat messages and receiving chatbot responses.
Delegates pipeline processing to ChatService.
"""

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.dependencies import get_chat_service
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.chat_service import ChatService
from app.core.exceptions import SessionNotFoundError, InvalidMessageError

router = APIRouter()


@router.post(
    "/chat",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    summary="Send Chat Message",
    description="Processes user message, updates session history, and returns assistant response."
)
async def chat_endpoint(
    payload: ChatRequest,
    service: ChatService = Depends(get_chat_service)
):
    try:
        return await service.process_chat(
            session_id=payload.session_id,
            user_message=payload.message,
            metadata=payload.metadata
        )
    except SessionNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=e.message)
    except InvalidMessageError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=e.message)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Internal processing error: {str(e)}")
