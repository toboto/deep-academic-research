"""
API路由定义

本模块定义了FastAPI接口的路由和处理函数。
"""

import asyncio
import json
import time
import os
from typing import AsyncGenerator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from deepsearcher import configuration
from deepsearcher.configuration import Configuration, init_config

from deepsearcher.api.models import (
    SummaryRequest,
    SummaryResponse,
    QuestionRequest,
    QuestionResponse,
    AIContentRequest,
)

# 使用匿名函数立即执行配置初始化
(lambda: (
    # 获取配置文件路径
    setattr(configuration, 'config', Configuration(
        os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
            "config.rbase.yaml"
        )
    )),
    # 初始化配置
    init_config(configuration.config)
))()

router = APIRouter()


async def generate_summary_stream(text: str) -> AsyncGenerator[bytes, None]:
    """
    生成流式响应

    Args:
        text: 要发送的文本内容

    Yields:
        bytes: 流式响应数据
    """
    # 发送role消息
    role_chunk = {
        "id": f"chatcmpl-{int(time.time()*1000)}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": "gpt-3.5-turbo",
        "choices": [{
            "index": 0,
            "delta": {"role": "assistant"},
            "finish_reason": None
        }]
    }
    yield f"data: {json.dumps(role_chunk)}\n\n".encode('utf-8')
    
    # 发送内容消息
    words = text.split()
    for i, word in enumerate(words):
        await asyncio.sleep(0.5)  # 模拟延迟
        
        content_chunk = {
            "id": f"chatcmpl-{int(time.time()*1000)}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": "gpt-3.5-turbo",
            "choices": [{
                "index": 0,
                "delta": {"content": word},
                "finish_reason": None if i < len(words) - 1 else "stop"
            }]
        }
        yield f"data: {json.dumps(content_chunk)}\n\n".encode('utf-8')
    
    # 发送结束标记
    yield "data: [DONE]\n\n".encode('utf-8')


@router.post(
    "/generate/summary",
    response_model=SummaryResponse,
    summary="AI概述接口",
    description="""
    生成AI概述内容。
    
    - 支持作者、主题、论文三种关联类型
    - 可选择是否使用缓存
    - 支持流式响应
    """,
)
async def generate_summary(request: SummaryRequest):
    """
    生成AI概述内容

    Args:
        request: 请求参数

    Returns:
        SummaryResponse或StreamingResponse: 响应结果

    Raises:
        HTTPException: 当请求参数无效或处理失败时抛出
    """
    try:
        # TODO: 根据request.related_type和request.related_id获取实际内容
        # 这里使用模拟数据
        content = "这是一个示例概述内容，用于测试流式响应功能。"
        ai_request = AIContentRequest()
        ai_request.parseFromSummaryRequest(request)
        
        if request.stream:
            # 返回流式响应
            return StreamingResponse(
                generate_summary_stream(content),
                media_type="text/event-stream"
            )
        else:
            # 返回非流式响应
            return SummaryResponse(code=0, message=content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/generate/questions",
    response_model=QuestionResponse,
    summary="AI推荐问题接口",
    description="""
    生成AI推荐问题。
    
    - 支持作者、主题、论文三种关联类型
    - 可选择是否使用缓存
    - 支持流式响应
    """,
)
async def generate_questions(request: QuestionRequest) -> QuestionResponse:
    """
    生成AI推荐问题

    Args:
        request: 请求参数

    Returns:
        QuestionResponse: 响应结果

    Raises:
        HTTPException: 当请求参数无效或处理失败时抛出
    """
    try:
        # TODO: 实现具体的处理逻辑
        return QuestionResponse(code=0, message="Success")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 