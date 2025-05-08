"""
文章总结生成器

本模块实现了基于RAG的文章总结生成功能。
"""

from typing import List, Tuple, Generator, Dict
from deepsearcher.agent.base import RAGAgent, describe_class
from deepsearcher.rbase.rbase_article import RbaseArticle
from deepsearcher.llm.base import BaseLLM
from deepsearcher.vector_db import RetrievalResult

class SummaryPromptTemplate:
    id = ""
    lang = "zh"
    target = "summary"
    prompt = ""

    def __init__(self, id: str, target: str, lang: str, prompt: str):
        self.id = id
        self.target = target
        self.lang = lang
        self.prompt = prompt

    def application_description(self) -> str:
        return f"Target Descirption: Used for generating summary articles for {self.target}, suitable for scenarios where the target language is {self.lang}"

    def generate_prompt(self, user_params: dict) -> str:
        """
        根据用户参数生成提示词
        """
        return self.prompt.format(**user_params)


@describe_class(
    "This agent is designed to generate comprehensive academic summary on research articles following a structured approach with multiple sections."
)
class SummaryRag(RAGAgent):
    """文章总结生成器"""
    
    def __init__(self, reasoning_llm: BaseLLM, writing_llm: BaseLLM, **kwargs):
        """初始化总结生成器"""
        super().__init__(**kwargs)
        self.verbose = False
        self.reasoning_llm = reasoning_llm
        self.writing_llm = writing_llm
        if kwargs.get("target_lang"):
            self.target_lang = kwargs.get("target_lang")
        else:
            self.target_lang = "Chinese"
        self.prompt_templates = _prepare_prompt_templates()

    def query(
        self,
        query: str,
        articles: List[RbaseArticle],
        params: dict = {},
        **kwargs
    ) -> Tuple[str, List[RetrievalResult], dict]:
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
        usage = {}
        for chunk in self.query_generator(query, articles, params, **kwargs):
            if len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                if hasattr(delta, "content") and delta.content is not None:
                    collected_content += delta.content

            # 如果有token信息，累加
            if hasattr(chunk, "usage") and chunk.usage:
                usage = chunk.usage
        return collected_content, [], usage

    def query_generator(self, query: str, articles: List[RbaseArticle], params: dict = {}, **kwargs) -> Generator[str, None, None]:
        """
        生成文章总结
        """
        if kwargs.get("verbose"):
            self.verbose = True
        if kwargs.get("target_lang"):
            self.target_lang = kwargs.get("target_lang")

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
        prompt_template = self.select_prompt_template(query, self.target_lang)
        params["query"] = query
        params["articles_info"] = articles_info
        prompt = prompt_template.generate_prompt(user_params=params)
        
        # 调用LLM生成总结
        return self.writing_llm.stream_generator([{"role": "user", "content": prompt}])

    def select_prompt_template(self, query: str, target_lang: str) -> SummaryPromptTemplate:
        """
        根据用户查询和目标语言选择最合适的提示词模板

        Args:
            query: 用户查询内容
            target_lang: 目标语言

        Returns:
            SummaryPromptTemplate: 选中的提示词模板
        """
        # 构建模板选择提示词
        templates_info = []
        for template_id, template in self.prompt_templates.items():
            templates_info.append(f"Template ID: {template_id}\n{template.application_description()}\n")

        prompt = f"""请根据以下信息，选择最合适的提示词模板：

用户查询内容：{query}
目标语言：{target_lang}

可用的模板列表：
{''.join(templates_info)}

请仔细分析用户查询内容和目标语言，选择最匹配的模板ID。只需要返回模板ID，不需要其他解释。
"""
        
        # 使用reasoning_llm选择模板
        response = self.reasoning_llm.chat([{"role": "user", "content": prompt}])
        selected_template_id = response.content.strip()
        
        # 验证选择的模板是否存在
        if selected_template_id not in self.prompt_templates:
            # 如果选择无效，默认使用第一个模板
            selected_template_id = list(self.prompt_templates.keys())[0]
            
        return self.prompt_templates[selected_template_id] 

def _prepare_prompt_templates() -> Dict[str, SummaryPromptTemplate]:
    templates = {}
    id = "channel_summary_01"
    templates[id] = SummaryPromptTemplate(id=id, target="channel summary or column summary", lang="Chinense", prompt="""
请根据以下文章列表，生成一篇总结性文章，文章的目标是：{query}。要求内容包括：

1. 栏目科研的主题都有哪些
2. 核心文章所阐述的研究内容和科研成果
3. 最新的研究进展
4. 整体研究价值和重要意义
5. 引用文章时，使用格式[X]，X为文章列表中的article_id

字数要求：{min_words}-{max_words}字

文章列表：
{articles_info}

请直接生成总结文本，不要包含任何额外的说明或格式。 """)

    id = "channel_summary_02"
    templates[id] = SummaryPromptTemplate(id=id, target="channel summary or column summary", lang="English", prompt="""
Based on the following list of articles, please generate a summary article with the goal of: {query}. The content should include:

1. What are the main themes of scientific research in the column?
2. What are the research contents and scientific achievements of the core articles?
3. What are the latest research developments?
4. What is the overall research value and significance?
5. When citing articles, use the format [X], where X is the article_id in the article list

Word count requirement: {min_words}-{max_words} words

Article list:
{articles_info}

Please generate the summary text directly, without any additional explanations or formats. """)

    id = "channel_question_01"
    templates[id] = SummaryPromptTemplate(id=id, target="user cared questions about the channel", lang="Chinense", prompt="""
本栏目是关于{query}的内容，并且包含了以下文章，请以一个思考并提出用户可能会关心的{question_count}个科研问题。

文章列表：
{articles_info}

请直接生成{question_count}个科研问题，每个问题前面有一个序号，不要包含任何额外的说明或格式。""")

    id = "channel_question_02"
    templates[id] = SummaryPromptTemplate(id=id, target="user cared questions about the channel", lang="English", prompt="""
The column is about {query} content, and includes the following articles, please think of and propose {question_count} scientific research questions that users may care about.

Article list:
{articles_info}

Please generate {question_count} scientific research questions, each with a number in front, without any additional explanations or formats. """)
    return templates
