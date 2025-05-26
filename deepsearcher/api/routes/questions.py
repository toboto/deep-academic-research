"""
Question Recommendation Routes

This module contains routes and functions for generating AI recommended questions.
"""

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from deepsearcher.api.models import (
    QuestionRequest,
    QuestionResponse,
    ExceptionResponse,
    DepressCache,
)
from deepsearcher.api.rbase_util import get_response_by_request_hash
from deepsearcher.rbase.ai_models import initialize_ai_request_by_question
from .metadata import build_metadata, build_metadata_by_discuss_thread
from .utils import generate_ai_content

router = APIRouter()

@router.post(
    "/questions",
    summary="AI Question Recommendation API",
    description="""
    Generate AI recommended questions.
    
    - Supports author, topic, and paper related types
    - Optional cache usage
    - Supports streaming response
    """,
)
async def api_generate_questions(request: QuestionRequest):
    """
    Generate AI recommended questions based on the request.

    Args:
        request (QuestionRequest): The question request parameters

    Returns:
        QuestionResponse: The response containing recommended questions

    Raises:
        HTTPException: When request parameters are invalid or processing fails
    """
    try:
        metadata = await build_metadata(request.related_type, request.related_id, request.term_tree_node_ids)
        metadata = await build_metadata_by_discuss_thread(request.thread_uuid, metadata)
        ai_request = initialize_ai_request_by_question(request, metadata)
        if request.depress_cache == DepressCache.DISABLE and 'user_history' not in metadata:
            ai_response = await get_response_by_request_hash(ai_request.request_hash)
        else:
            ai_response = None

        question_response = QuestionResponse(code=0, message="success")
        if not ai_response:
            summary = await generate_ai_content(ai_request, request.related_type, None) 
            question_response.setQuestions(summary)
        else:
            question_response.setQuestions(ai_response.content)
        return question_response
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content=ExceptionResponse(code=500, message=str(e)).model_dump()
        ) 