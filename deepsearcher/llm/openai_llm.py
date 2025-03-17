import os
from typing import Dict, List

from deepsearcher.llm.base import BaseLLM, ChatResponse
from deepsearcher.tools import log


class OpenAI(BaseLLM):
    def __init__(self, model: str = "o1-mini", **kwargs):
        from openai import OpenAI as OpenAI_

        self.model = model
        # 检查是否启用流式调用
        self.stream_mode = kwargs.pop("stream", False)
        # 检查是否启用详细日志
        self.verbose = kwargs.pop("verbose", False)
        
        if "api_key" in kwargs:
            api_key = kwargs.pop("api_key")
        else:
            api_key = os.getenv("OPENAI_API_KEY")
        if "base_url" in kwargs:
            base_url = kwargs.pop("base_url")
        else:
            base_url = os.getenv("OPENAI_BASE_URL")
        self.client = OpenAI_(api_key=api_key, base_url=base_url, **kwargs)

    def chat(self, messages: List[Dict]) -> ChatResponse:
        if self.stream_mode:
            # 流式调用模式
            return self._stream_chat(messages)
        else:
            # 普通调用模式
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
            )
            return ChatResponse(
                content=completion.choices[0].message.content,
                total_tokens=completion.usage.total_tokens,
            )
            
    def _stream_chat(self, messages: List[Dict]) -> ChatResponse:
        """
        使用流式模式调用API
        
        Args:
            messages: 消息列表
            
        Returns:
            聊天响应对象
        """
        # 创建流式请求
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=True,
        )
        
        # 收集完整响应
        collected_content = ""
        reasoning_content = ""  # 收集推理内容
        total_tokens = 0
        is_answering = False  # 标记是否已经从推理过程转为回答过程
        
        # 处理流式响应
        for chunk in stream:
            # 如果chunk.choices为空，可能包含usage信息
            if not chunk.choices:
                if hasattr(chunk, 'usage') and chunk.usage:
                    total_tokens = chunk.usage.total_tokens
                continue
                
            delta = chunk.choices[0].delta
            
            # 处理推理内容（特别是QwQ模型）
            if hasattr(delta, 'reasoning_content') and delta.reasoning_content is not None:
                reasoning_chunk = delta.reasoning_content
                reasoning_content += reasoning_chunk
                if self.verbose:
                    log.debug(f"[推理] {reasoning_chunk}")
            # 处理回答内容
            elif hasattr(delta, 'content') and delta.content is not None:
                # 标记开始回答
                if delta.content != "" and not is_answering:
                    is_answering = True
                    if self.verbose:
                        log.debug("\n--- 开始回答 ---\n")
                
                content_chunk = delta.content
                collected_content += content_chunk
                if self.verbose:
                    log.debug(f"[回答] {content_chunk}")
            
            # 如果有token信息，累加
            if hasattr(chunk, 'usage') and chunk.usage:
                total_tokens = chunk.usage.total_tokens
        
        # 如果没有获取到token信息，可以估算
        if total_tokens == 0:
            total_tokens = (len(collected_content) + len(reasoning_content)) // 2
        
        # 如果有推理内容，可以将其添加到最终响应中（可选）
        final_content = collected_content
        if reasoning_content and self.verbose:
            log.debug(f"\n--- 完整推理过程 ---\n{reasoning_content}")
            
        return ChatResponse(
            content=final_content,
            total_tokens=total_tokens,
        )
