from typing import Tuple, List, Generator
from deepsearcher import configuration
from deepsearcher.llm.base import BaseLLM
from deepsearcher.embedding.base import BaseEmbedding
from deepsearcher.agent import AcademicTranslator
from deepsearcher.vector_db.base import BaseVectorDB
from deepsearcher.vector_db import RetrievalResult
from deepsearcher.tools import log
import json


DISCUSS_ACTION_PROMPT = """
你是AI学术助理，有一个用户正在跟你对话，客户当前正在{user_action}，请判断用户的问题的意图以及本次对话是否需要查询文献。

对话背景信息：
{background}

对话历史：
{history}

用户的问题：
{query}

首先，请判断用户的问题实际上表达的是什么意图：
1. 如果用户正在咨询学术方面的问题，或对历史对话内容进行进一步的追问，那么用户的意图是"提问"
2. 如果用户发表了一个学术观点，请对他的观点进行评价，那么用户的意图是"发表观点"
3. 如果用户对之前的历史回答表示质疑，并且提出了他的看法，那么用户的意图是"质疑且需要回复"
4. 如果用户只是表达质疑，那么用户的意图是"质疑"
5. 如果用户只是表达肯定，那么用户的意图是"肯定"
6. 如果用户的表达没有特定意图，或者可能只是发表了感叹，那么意图是"无需回复"

其次，如果用户的意图是1-5（即需要我们进行回复），我们回答用户问题时还要判断是否需要查询更多文献：
1. 如果根据背景信息和对话历史，足以回答用户的问题，那么不需要查询更多文献
2. 如果根据背景信息和对话历史，不足以回答用户的问题，那么需要查询更多文献，请根据用户的问题给出查询文献的请求

请进行上述两项判断，并以JSON格式回复，JSON格式如下：
{{
    "intention": "提问" | "发表观点" | "质疑且需要回复" | "质疑" | "肯定" | "无需回复",
    "need_search": true | false,
    "search_query": "查询文献的请求"
}}

请直接输出JSON数据，不要输出任何解释。
"""

DISCUSS_ANSWER_PROMPT = """
你是AI学术助理，有一个用户正在跟你对话，客户当前正在{user_action}，请根据用户的问题给出回答。

背景信息：
{background}

文献内容检索结果：
{retrieval_results}

对话历史：
{history}

用户的问题：
{query}

用户提问的意图是：
{intention}

回复用户的语言是：{target_lang}

根据用户的不同意图进行回复：
1. "提问"：请用专业、准确、友好的语言回答用户的问题，充分结合背景信息和查询到的文献内容。
2. "发表观点"：请结合背景信息和文献内容，判断用户的观点，并给出你对于用户观点的看法。
3. "质疑且需要回复"：对于用户的质疑，结合用户提出的看法，给出你的回复，你可以坚持自己的观点也可以调整自己的观点。
4. "质疑"：对于用户的质疑，结合背景信息和文献内容，进一步阐述你的观点。
5. "肯定"：请根据用户的问题给出你的回复。
6. "无需回复"：则不再进行更多判断直接回复一个空字符串。

回复客户的原则如下：
1. 请用专业、准确、友好的语言回答用户的问题。
2. 如果检索结果中包含相关信息，请确保引用相应的来源。
3. 不要使用背景信息或者文献内容检索中没有的文献材料作为回答的依据。
4. 回答应当简洁明了，并且针对用户的具体问题提供有用的信息，在实在没有回答思路时，可以回复用户"抱歉，这方面问题我暂时还没有一个清晰的回答思路"，用你的语言表达类似的含义。
"""

class DiscussAgent:
    """
    讨论代理类，用于处理用户与AI之间的学术讨论。
    
    该代理会分析用户问题的意图，决定是否需要查询更多文献，并生成相应的回复。
    """
    def __init__(self, llm: BaseLLM, reasoning_llm: BaseLLM, translator: AcademicTranslator, embedding_model: BaseEmbedding, vector_db: BaseVectorDB, **kwargs):
        """
        初始化讨论代理
        
        Args:
            llm: 语言模型
            reasoning_llm: 推理语言模型
            translator: 学术翻译器
            embedding_model: 向量模型
            vector_db: 向量数据库
        """
        self.llm = llm
        self.reasoning_llm = reasoning_llm
        self.translator = translator
        self.embedding_model = embedding_model
        self.vector_db = vector_db
        
        # 设置默认检索参数
        self.top_k_per_section = kwargs.get("top_k_per_section", 5)
        self.vector_db_collection = kwargs.get("vector_db_collection", self.vector_db.default_collection)
        self.verbose = kwargs.get("verbose", configuration.config.rbase_settings.get("verbose", False))

    def resetUsage(self):
        self.usage = {"total_tokens": 0, "prompt_tokens": 0, "completion_tokens": 0}

    def query(self, query: str, **kwargs) -> Tuple[str, List[RetrievalResult], dict]:
        """
        处理用户查询并生成回复
        
        Args:
            query: 用户查询
            **kwargs: 其他参数，包括user_action, background, history, target_lang, request_params等
            
        Returns:
            Tuple(回复文本, 检索结果列表, 其他元数据)
        """
        collected_content = ""
        for chunk in self.query_generator(query, **kwargs):
            if len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                if hasattr(delta, "content") and delta.content is not None:
                    collected_content += delta.content

            if hasattr(chunk, "usage") and chunk.usage:
                self.usage["total_tokens"] += chunk.usage.total_tokens
                self.usage["prompt_tokens"] += chunk.usage.prompt_tokens
                self.usage["completion_tokens"] += chunk.usage.completion_tokens
        return collected_content, [], self.usage


    def query_generator(self, query: str, **kwargs) -> Generator[str, None, None]:
        """
        处理用户查询并生成回复
        
        Args:
            query: 用户查询
            **kwargs: 其他参数，包括user_action, background, history, target_lang, request_params等
            
        Returns:
            Tuple(回复文本, 检索结果列表, 其他元数据)
        """
        self.resetUsage()
        # 获取参数
        user_action = kwargs.get("user_action", "")
        background = kwargs.get("background", "")
        history = kwargs.get("history", [])
        target_lang = kwargs.get("target_lang", "zh")
        request_params = kwargs.get("request_params", {})
        self.top_k_per_section = kwargs.get("top_k_per_section", self.top_k_per_section)
        self.vector_db_collection = kwargs.get("vector_db_collection", self.vector_db_collection)
        self.verbose = kwargs.get("verbose", self.verbose)
        
        # 格式化对话历史
        formatted_history = ""
        for item in history:
            if item.get("role") == "user":
                formatted_history += f"用户: {item.get('content', '')}\n"
            else:
                formatted_history += f"AI助理: {item.get('content', '')}\n\n"
            
        # 第一步：判断用户意图和是否需要检索
        prompt = DISCUSS_ACTION_PROMPT.format(
            user_action=user_action,
            background=background,
            history=formatted_history,
            query=query
        )
        
        self._verbose(f"<判断意图> 分析用户问题意图... </判断意图>", debug_msg=f"prompt: {prompt}")
        response = self.reasoning_llm.chat([{"role": "user", "content": prompt}])
        self.usage = response.usage()
        try:
            # 解析LLM返回的JSON响应
            content = response.content.strip()
            # 处理可能的markdown代码块格式
            if content.startswith("```json"):
                content = content[7:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
            
            action_result = json.loads(content)
            intention = action_result.get("intention")
            need_search = action_result.get("need_search", False)
            search_query = action_result.get("search_query", "")
            
            # 检查是否需要回复
            if intention == "无需回复":
                return ()
                
            # 第二步：如果需要检索，则从vector_db中检索相关内容
            retrieval_results = []
            if need_search and search_query:
                self._verbose(f"<检索> 正在检索文献，查询语句: '{search_query}' </检索>")
                
                # 准备过滤条件
                filter_str = self._query_filter(request_params)
                
                # 执行检索
                query_vector = self.embedding_model.embed_query(search_query)
                retrieval_results = self.vector_db.search_data(
                    collection=self.vector_db_collection,
                    vector=query_vector,
                    top_k=self.top_k_per_section,
                    filter=filter_str
                )
                
                self._verbose(f"<检索> 检索到 {len(retrieval_results)} 条文献")
            
            # 格式化检索结果
            formatted_results = ""
            for i, result in enumerate(retrieval_results):
                formatted_results += f"[{result.metadata.get('reference_id', i+1)}] \n{result.text}\n\n"
            
            # 生成回复
            answer_prompt = DISCUSS_ANSWER_PROMPT.format(
                user_action=user_action,
                background=background,
                retrieval_results=formatted_results,
                history=formatted_history,
                query=query,
                intention=intention,
                target_lang=target_lang
            )
            
            self._verbose(f"<生成回复> 正在生成回复... </生成回复>", debug_msg=f"answer_prompt: {answer_prompt}")
            return  self.llm.stream_generator([{"role": "user", "content": answer_prompt}])
            
        except json.JSONDecodeError as e:
            log.error(f"解析LLM响应失败: {e}")
            return 
    
    def _query_filter(self, request_params: dict) -> str:
        # 准备过滤条件
        conditions = []
        if request_params:
            # 这里可以根据request_params构建过滤条件
            # 例如时间范围、影响因子等
            if "pubdate" in request_params:
                conditions.append(f"pubdate >= {request_params['pubdate']}")
            if "impact_factor" in request_params:
                conditions.append(f"impact_factor >= {request_params['impact_factor']}")
            if "base_id" in request_params:
                conditions.append(f"ARRAY_CONTAINS(base_ids, {request_params['base_id']})")
        if len(conditions) > 0:
            return " AND ".join(conditions)
        else:
            return ""
    
    def _verbose(self, msg: str, debug_msg: str = ""):
        if self.verbose:
            log.color_print(msg)
            if debug_msg:
                log.debug(debug_msg)
