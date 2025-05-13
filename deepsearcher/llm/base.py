import ast
import re
from abc import ABC
from typing import Dict, List, Generator


class ChatResponse(ABC):
    def __init__(self, content: str, total_tokens: int, **kwargs) -> None:
        self.content = content
        self.total_tokens = total_tokens
        self.prompt_tokens = kwargs.get("prompt_tokens", 0)
        self.completion_tokens = kwargs.get("completion_tokens", 0)

    def usage(self, offset: dict = {}) -> dict:
        if offset == {}:
            return {
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens": self.total_tokens,
            }
        else:
            return {
                "prompt_tokens": self.prompt_tokens + offset.get("prompt_tokens", 0),
                "completion_tokens": self.completion_tokens + offset.get("completion_tokens", 0),
                "total_tokens": self.total_tokens + offset.get("total_tokens", 0),
            }

    def __repr__(self) -> str:
        return f"ChatResponse(content={self.content}, total_tokens={self.total_tokens})"


class BaseLLM(ABC):
    def __init__(self):
        pass

    def chat(self, messages: List[Dict]) -> ChatResponse:
        pass

    def stream_generator(self, messages: List[Dict]) -> Generator[object, None, None]:
        pass

    @staticmethod
    def literal_eval(response_content: str):
        response_content = response_content.strip()

        # remove content between <think> and </think>, especial for DeepSeek reasoning model
        if "<think>" in response_content and "</think>" in response_content:
            end_of_think = response_content.find("</think>") + len("</think>")
            response_content = response_content[end_of_think:]

        try:
            if response_content.startswith("```") and response_content.endswith("```"):
                if response_content.startswith("```python"):
                    response_content = response_content[9:-3]
                elif response_content.startswith("```json"):
                    response_content = response_content[7:-3]
                elif response_content.startswith("```str"):
                    response_content = response_content[6:-3]
                elif response_content.startswith("```\n"):
                    response_content = response_content[4:-3]
                else:
                    raise ValueError("Invalid code block format")
            result = ast.literal_eval(response_content.strip())
        except Exception:
            matches = re.findall(r"(\[.*?\]|\{.*?\})", response_content, re.DOTALL)

            if len(matches) != 1:
                raise ValueError(
                    f"Invalid JSON/List format for response content:\n{response_content}"
                )

            json_part = matches[0]
            return ast.literal_eval(json_part)

        return result
