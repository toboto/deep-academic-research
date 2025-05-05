"""
文章总结生成器

本模块实现了基于RAG的文章总结生成功能。
"""

from typing import List, Tuple, Generator
from deepsearcher.agent.base import RAGAgent, describe_class
from deepsearcher.rbase.rbase_article import RbaseArticle
from deepsearcher.llm.base import BaseLLM
from deepsearcher.vector_db import RetrievalResult


@describe_class(
    "This agent is designed to generate comprehensive academic summary on research articles following a structured approach with multiple sections."
)
class SummaryRag(RAGAgent):
    """文章总结生成器"""
    
    def __init__(self, reasoning_llm: BaseLLM, writing_llm: BaseLLM, **kwargs):
        """初始化总结生成器"""
        super().__init__(**kwargs)
        self.verbose = False
        if kwargs.get("top_k_per_section"):
            self.top_k_per_section = kwargs.get("top_k_per_section")
        self.reasoning_llm = reasoning_llm
        self.writing_llm = writing_llm

    def query(
        self,
        query: str,
        articles: List[RbaseArticle],
        min_words: int = 500,
        max_words: int = 1000,
        **kwargs
    ) -> Tuple[str, List[RetrievalResult], int]:
        """
        生成文章总结

        Args:
            query: 生成内容的关键性提示
            articles: 文章列表
            min_words: 最小字数
            max_words: 最大字数
            **kwargs: 其他参数

        Returns:
            str: 生成的总结文本
        """
        collected_content = ""
        total_tokens = 0
        prompt_tokens = 0
        completion_tokens = 0
        for chunk in self.query_generator(query, articles, min_words, max_words, **kwargs):
            if len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                if hasattr(delta, "content") and delta.content is not None:
                    collected_content += delta.content

            # 如果有token信息，累加
            if hasattr(chunk, "usage") and chunk.usage:
                total_tokens += chunk.usage.total_tokens
                prompt_tokens += chunk.usage.prompt_tokens
                completion_tokens += chunk.usage.completion_tokens

        return collected_content, [], total_tokens

    def query_generator(self, query: str, articles: List[RbaseArticle], min_words: int = 500, max_words: int = 1000, **kwargs) -> Generator[str, None, None]:
        """
        生成文章总结
        """
        if kwargs.get("verbose"):
            self.verbose = True
        # 构建文章信息
        articles_info = []
        for article in articles:
            article_info = {
                "article_id": article.article_id,
                "title": article.title,
                "authors": article.authors,
                "journal": article.journal_name,
                "pubdate": article.pubdate,
                "abstract": article.abstract
            }
            articles_info.append(article_info)
        
        # 构建提示词
        prompt = f"""请根据以下文章列表，生成一篇总结性文章，文章的目标是：{query}。要求内容包括：

1. 栏目科研的主题都有哪些
2. 核心文章所阐述的研究内容和科研成果
3. 最新的研究进展
4. 整体研究价值和重要意义
5. 引用文章时，使用格式[X]，X为文章列表中的article_id

字数要求：{min_words}-{max_words}字

文章列表：
{articles_info}

请直接生成总结文本，不要包含任何额外的说明或格式。
"""
        
        # 调用LLM生成总结
        return self.writing_llm.stream_generator([{"role": "user", "content": prompt}])
