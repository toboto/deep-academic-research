"""
文章总结生成器

本模块实现了基于RAG的文章总结生成功能。
"""

from typing import List, Tuple, Generator, Dict
from deepsearcher.agent.base import RAGAgent, describe_class
from deepsearcher.rbase.rbase_article import RbaseArticle
from deepsearcher.llm.base import BaseLLM
from deepsearcher.vector_db import RetrievalResult
from deepsearcher.tools.log import debug
from deepsearcher import configuration

PROMPT_MATCHES = {
    "summary_zh":  "channel_summary_01",
    "summary_en":  "channel_summary_02",
    "question_zh":  "channel_question_01",
    "question_en":  "channel_question_02",
    "popular_zh":  "popular_01",
    "ppt_zh":  "ppt_01",
    "footage_zh":  "footage_01",
    "opportunity_zh":  "opportunity_01",
}

class SummaryPromptTemplate:
    id = ""
    lang = "zh"
    target = "summary"
    purpose = ""
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
        self.verbose = configuration.config.rbase_settings.get("verbose", False)
        self.reasoning_llm = reasoning_llm
        self.writing_llm = writing_llm
        if kwargs.get("target_lang"):
            self.target_lang = kwargs.get("target_lang")
        else:
            self.target_lang = "zh"
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
        if kwargs.get("purpose"):
            self.purpose = kwargs.get("purpose")
        else:
            self.purpose = ""
        
        prompt_template = self.select_prompt_template(query, self.target_lang, self.purpose)
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
        params["query"] = query
        params["articles_info"] = articles_info
        params = self._format_user_params(params)
        prompt = prompt_template.generate_prompt(user_params=params)
        if self.verbose:
            debug(f"prompt: {prompt}")
        
        # 调用LLM生成总结
        return self.writing_llm.stream_generator([{"role": "user", "content": prompt}])

    def select_prompt_template(self, query: str, target_lang: str, purpose: str) -> SummaryPromptTemplate:
        """
        根据用户查询和目标语言选择最合适的提示词模板

        Args:
            query: 用户查询内容
            target_lang: 目标语言

        Returns:
            SummaryPromptTemplate: 选中的提示词模板
        """

        selected_template_id = ""
        if purpose != "" and target_lang != "":
            key = f"{purpose}_{target_lang}"
            selected_template_id = PROMPT_MATCHES.get(key, "")

        # 构建模板选择提示词
        templates_info = []
        for template_id, template in self.prompt_templates.items():
            if template_id == selected_template_id:
                return template

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

    def _format_user_params(self, params: dict) -> dict:
        history = ""
        if params.get('user_history'):
            if self.target_lang == "zh":
                history += "\n用户最近的讨论记录：\n" 
            else:
                history += "\nUser's recent discussion record:\n"
            history += "\n\t".join([f"{item['role']}: {item['content']}" for item in params.get('user_history')])
            history += "\n\n"
        params["user_history"] = history
        return params


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

语言要求：中文
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

Language requirement: English
Word count requirement: {min_words}-{max_words} words

Article list:
{articles_info}

Please generate the summary text directly, without any additional explanations or formats. """)

    id = "channel_question_01"
    templates[id] = SummaryPromptTemplate(id=id, target="user cared questions about the channel", lang="Chinense", prompt="""
本栏目是关于{query}的内容，并且包含了以下文章，请以一个思考并提出用户可能会关心的{question_count}个科普性问题。
科普性问题不宜过长，10-20个字为宜。目标是让用户对栏目内容有一个初步的了解，不需要太深入。

文章列表：
{articles_info}

{user_history}

语言要求：中文

请直接生成{question_count}个科研问题，问题内容的前面无需编写序号，不要包含任何额外的说明或格式。""")

    id = "channel_question_02"
    templates[id] = SummaryPromptTemplate(id=id, target="user cared questions about the channel", lang="English", prompt="""
The column is about {query} content, and includes the following articles, please think of and propose {question_count} popular science questions that users might be interested in.
The popular science questions should not be too long, 10-20 words is appropriate. The goal is to give users a preliminary understanding of the column content, not too deep.

Article list:
{articles_info}

{user_history}

Language requirement: English

Please generate {question_count} scientific research questions, without any additional explanations or formats. """)

    id = "popular_01"
    templates[id] = SummaryPromptTemplate(id=id, target="popular science short article", lang="Chinense", prompt="""
你是一位专业的科普作家。请根据用户关注的"核心问题"和提供的"参考文章"，创作一篇"通俗易懂的科普短文"。

核心问题: {query}

文章要求:
1. 深入浅出地解释核心问题的基础概念。
2. 内容力求简洁明了，确保读者轻松理解。
3. 在必要时，巧妙运用生活化类比，帮助读者构建具象认知。
4. 准确引用参考文章，格式为：[X]（X为文章列表中的article_id），遇到参考文章ID请无论在任何位置都使用这种格式，不要直接陈列文章ID。

语言要求：中文
字数要求：{min_words}-{max_words}字

参考文章列表：
{articles_info}

请直接生成科普性短文，不要包含任何额外的说明或格式。 """)

    id = "ppt_01"
    templates[id] = SummaryPromptTemplate(id=id, target="ppt outline", lang="Chinense", prompt="""
你是一位专业的演示文稿设计师，擅长将复杂信息转化为清晰、有条理的PPT结构。

请根据用户关注的"核心主题"和提供的"参考文章"，创作一份"详细的PPT提纲"。

核心主题: {query}

提纲要求:
1. 结构完整：提纲需包含PPT的标题和主要内容点。
2. 逻辑清晰：内容点之间应有明确的逻辑关系，便于演示和理解。
3. 内容提炼：从参考文章中提取关键信息，确保每页内容简洁有力。
4. 准确引用参考文章，格式为：[X]（X为文章列表中的article_id），遇到参考文章ID请无论在任何位置都使用这种格式，不要直接陈列文章ID。

语言要求：中文

文章列表：
{articles_info}

请请直接生成PPT提纲，不要包含任何额外的说明或格式。 """)

    id = "footage_01"
    templates[id] = SummaryPromptTemplate(id=id, target="footage script", lang="Chinense", prompt="""
你是一位经验丰富的短视频内容创作者，擅长将科普知识转化为生动有趣的视频脚本。

请根据用户关注的"核心主题"和提供的"参考文章"，创作一份"详细的短视频脚本"。

核心主题: {query}

脚本要求:
1. 完整性：脚本需包含视频标题、开场白、主体内容（分段落或场景）、关键视觉建议和结尾呼吁/总结。
2. 吸引力：内容应具备短视频的特点，如节奏明快、语言口语化、易于理解和传播。
3. 视觉化：在内容描述中融入对画面的想象和建议，帮助理解和制作。
4. 信息提炼：从参考文章中提取核心观点和关键信息，确保内容的准确性和精炼性。
5. 准确引用参考文章，格式为：[X]（X为文章列表中的article_id），遇到参考文章ID请无论在任何位置都使用这种格式，不要直接陈列文章ID。

语言要求：中文

文章列表：
{articles_info}

请直接生成短视频脚本，不要包含任何额外的说明或格式。 """)

    id = "opportunity_01"
    templates[id] = SummaryPromptTemplate(id=id, target="analyze business opportunity", lang="Chinense", prompt="""
你是一位资深的商业分析师和市场洞察专家，擅长从科研发现中识别潜在的商业价值。

请根据用户关注的"核心科研问题"和提供的"参考文章"，创作一篇"深入分析潜在商业机会的短文"。

核心科研问题: {query}

分析要求:
1. 洞察商机：详细分析文章中提及的科研问题可能催生哪些具体的商业机会（如产品、服务、技术解决方案、市场空白等）。
2. 潜在价值：阐述这些商机的潜在市场规模、商业模式可能性或颠覆性潜力。
3. 关键要素：指出实现这些商机可能需要考虑的技术成熟度、市场需求、竞争格局或合作机会等关键要素。
4. 准确引用参考文章，格式为：[X]（X为文章列表中的article_id），遇到参考文章ID请无论在任何位置都使用这种格式，不要直接陈列文章ID。

语言要求：中文
字数要求：{min_words}-{max_words}字

文章列表：
{articles_info}

请直接生成科普性短文，不要包含任何额外的说明或格式。 """)

    return templates
