"""
Overview RAG Agent Module.

This module provides the OverviewRAG class for generating comprehensive
academic reviews on specified research topics. It follows a structured approach
to create well-organized research overviews by querying knowledge bases for 
relevant information for each section of the review.
"""

import asyncio
from typing import List, Tuple, Dict, Any

from deepsearcher.agent.base import RAGAgent, describe_class
from deepsearcher.agent.academic_translator import AcademicTranslator
from deepsearcher.embedding.base import BaseEmbedding
from deepsearcher.llm.base import BaseLLM
from deepsearcher.tools import log
from deepsearcher.vector_db import RetrievalResult
from deepsearcher.vector_db.base import BaseVectorDB, deduplicate_results
from deepsearcher.agent.collection_router import CollectionRouter

# 结构划分提示词
STRUCTURE_PROMPT = """
You are an academic research assistant tasked with planning a comprehensive literature review on a specific topic.

Generate appropriate search queries for each section of the literature review structure below. The goal is to retrieve relevant academic content from our knowledge base for each section.

Research Topic: <topic>

For each section, please provide:
1. A focused search query that will help retrieve the most relevant content from our academic database
2. Specific search conditions or filters that would help refine the results (e.g., time period, specific subtopics, methodological focus, etc.)

Literature Review Structure:
1. Introduction (Background & Problem Definition)
2. Theoretical Foundations (Core Theory Evolution)
3. Methodological Approaches (Methodology Landscape)
4. Key Findings & Debates (Core Discoveries & Academic Controversies)
5. Emerging Trends (Frontier Analysis)
6. Research Gaps & Future Directions (Prediction of Unexplored Areas)

Format your response as a Python dictionary with the following structure:
{
    "Introduction": {
        "query": "search query for introduction",
        "conditions": ["condition1", "condition2"]
    },
    "Theoretical Foundations": {
        "query": "search query for theoretical foundations",
        "conditions": ["condition1", "condition2"]
    },
    ...
}

Ensure your queries are specific, academic in nature, and designed to retrieve comprehensive information for each section.
"""

# 章节内容生成提示词
SECTION_GENERATION_PROMPT = """
You are an academic writer specializing in creating comprehensive literature reviews. Based on the retrieved academic content, write a detailed section for a literature review.

Section: {section_name}
Topic: {topic}

Retrieved Content:
{retrieved_content}

Guidelines:
1. Write a cohesive, well-structured section that thoroughly covers the topic based on the retrieved content
2. Use appropriate academic language and maintain a scholarly tone
3. Properly cite sources within the text using the format [X], where X corresponds to the chunk number from the retrieved content
4. Synthesize information rather than merely summarizing individual sources
5. Highlight consensus views as well as contrasting perspectives in the field
6. Maintain appropriate length for a section in a comprehensive literature review (approximately 800-1200 words)
7. Ensure logical flow within the section

Your response should be a polished section ready for inclusion in the final literature review.
"""

# 全文润色提示词
POLISH_PROMPT = """
You are a senior academic editor specializing in polishing scholarly literature reviews. Review the complete draft of this literature review and improve it for publication quality.

Topic: {topic}

Draft Literature Review:
{full_text}

Guidelines for Improvement:
1. Ensure logical flow and coherence throughout the entire document
2. Eliminate any redundancies or repetitive content
3. Check for and correct any logical inconsistencies or structural problems
4. Improve transitions between sections
5. Enhance clarity and precision of language
6. Maintain consistent academic tone and style throughout
7. Ensure appropriate depth of analysis in each section
8. Do not change the citations format [X]
9. Keep the overall content and organization, making only improvements to quality rather than substantive changes

Your response should be the complete, polished literature review ready for submission.
"""

# 语言检测提示词
LANGUAGE_DETECT_PROMPT = """
Determine the primary language of the following text. Return only the language code:
- "en" for English
- "zh" for Chinese
- "mixed" if the text contains a significant amount of both languages

Text: {text}

Language code:
"""

# 搜索结果重排提示词
RERANK_PROMPT = """
Based on the query and the retrieved chunk, determine whether the chunk is helpful in addressing the query. Respond with only "YES" or "NO".

Query: {query}
Retrieved Chunk: {retrieved_chunk}

Is the chunk helpful for addressing the query?
"""

@describe_class(
    "This agent is designed to generate comprehensive academic reviews on research topics following a structured approach with multiple sections."
)
class OverviewRAG(RAGAgent):
    """
    OverviewRAG agent for generating comprehensive academic reviews on research topics.
    
    This agent follows a structured approach to create well-organized research overviews
    by querying knowledge bases for each section of the review and synthesizing the
    information into a cohesive document.
    """
    
    def __init__(
        self,
        llm: BaseLLM,
        reasoning_llm: BaseLLM,
        writing_llm: BaseLLM,
        translator: AcademicTranslator,
        embedding_model: BaseEmbedding,
        vector_db: BaseVectorDB,
        route_collection: bool = True,
        text_window_splitter: bool = True,
        **kwargs,
    ):
        """
        Initialize the OverviewRAG agent.
        
        Args:
            llm: Base language model for general tasks
            reasoning_llm: Language model optimized for reasoning tasks
            writing_llm: Language model optimized for writing tasks
            translator: Academic translator for language translation
            embedding_model: Embedding model for vector encoding
            vector_db: Vector database for knowledge retrieval
            text_window_splitter: Whether to use text window splitting
        """
        self.llm = llm
        self.reasoning_llm = reasoning_llm
        self.writing_llm = writing_llm
        self.translator = translator
        self.embedding_model = embedding_model
        self.vector_db = vector_db
        self.text_window_splitter = text_window_splitter
        self.route_collection = route_collection
        if route_collection:
            self.collection_router = CollectionRouter(llm=self.llm, vector_db=self.vector_db)
        else:
            self.collection_router = None

        if kwargs.get("top_k_per_section"):
            self.top_k_per_section = kwargs.get("top_k_per_section")
        else:
            self.top_k_per_section = 10
        
        # Define the standard structure for academic reviews
        self.sections = [
            "Introduction",
            "Theoretical Foundations",
            "Methodological Approaches",
            "Key Findings & Debates",
            "Emerging Trends",
            "Research Gaps & Future Directions"
        ]
    
    def _detect_language(self, text: str) -> str:
        """
        Detect the language of the input text.
        
        Args:
            text: Input text to detect language
            
        Returns:
            Language code: 'en', 'zh', or 'mixed'
        """
        prompt = LANGUAGE_DETECT_PROMPT.format(text=text)
        response = self.llm.chat([{"role": "user", "content": prompt}])
        language = response.content.strip().lower()
        
        # Validate the response
        if language not in ["en", "zh", "mixed"]:
            # Default to mixed if detection is unclear
            return "mixed"
        
        return language
    
    def _translate_to_english(self, text: str) -> str:
        """
        Translate the input text to English.
        
        Args:
            text: Text to translate
            
        Returns:
            Translated text in English
        """
        return self.translator.translate(text, "en")
    
    def _translate_to_chinese(self, text: str) -> str:
        """
        Translate the input text to Chinese.
        
        Args:
            text: Text to translate
            
        Returns:
            Translated text in Chinese
        """
        return self.translator.translate(text, "zh")
    
    def _generate_section_queries(self, topic: str) -> Dict[str, Dict[str, Any]]:
        """
        Generate search queries for each section of the review.
        
        Args:
            topic: The research topic
            
        Returns:
            Dictionary mapping section names to search queries and conditions
        """
        prompt = STRUCTURE_PROMPT.replace("<topic>", topic)
        response = self.reasoning_llm.chat([{"role": "user", "content": prompt}])
        
        # Parse the response to get the dictionary
        try:
            # 尝试多种方式解析LLM返回的字典格式
            import ast
            import re
            import json
            
            # 去除可能存在的多余内容
            content = response.content.strip()
            
            # 方法1：尝试直接用ast.literal_eval解析
            try:
                queries_dict = ast.literal_eval(content)
                return queries_dict
            except (SyntaxError, ValueError):
                pass
            
            # 方法2：用正则表达式提取字典部分
            try:
                dict_pattern = r'\{[\s\S]*\}'
                dict_match = re.search(dict_pattern, content)
                if dict_match:
                    dict_str = dict_match.group(0)
                    queries_dict = ast.literal_eval(dict_str)
                    return queries_dict
            except (SyntaxError, ValueError):
                pass
            
            # 方法3：尝试json解析
            try:
                # 去除可能的markdown代码块标记
                json_content = re.sub(r'```(?:json|python)?|```', '', content).strip()
                queries_dict = json.loads(json_content)
                return queries_dict
            except json.JSONDecodeError:
                pass
            
            # 所有解析方法都失败，使用默认值
            log.warning("无法解析LLM返回的字典格式，使用默认查询")
            return {section: {"query": f"{topic} {section.lower()}", "conditions": []} 
                    for section in self.sections}
        except Exception as e:
            log.error(f"生成查询失败: {e}")
            # 使用基本结构作为后备方案
            return {section: {"query": f"{topic} {section.lower()}", "conditions": []} 
                    for section in self.sections}
    
    async def _search_for_section(self, section: str, query: str, conditions: List[str] = None) -> Tuple[List[RetrievalResult], int]:
        """
        Search for content relevant to a specific section.
        
        Args:
            section: Section name
            query: Search query
            conditions: Optional list of search conditions
            
        Returns:
            Tuple of (retrieved results, tokens used)
        """
        log.color_print(f"<search> Searching for section '{section}' with query: '{query}' </search>\n")
        
        # Incorporate conditions into the query if provided
        if conditions and len(conditions) > 0:
            # enhanced_query = f"{query} " + " ".join(conditions)
            enhanced_query = query
        else:
            enhanced_query = query
            
        query_vector = self.embedding_model.embed_query(enhanced_query)
        consumed_tokens = 0
        
        # 决定要搜索的集合
        if self.route_collection:
            # 使用CollectionRouter选择合适的集合
            selected_collections, n_token_route = self.collection_router.invoke(query=enhanced_query)
            consumed_tokens += n_token_route
            log.color_print(f"<search> Collection router selected: {selected_collections} </search>\n")
        else:
            # 使用默认集合
            selected_collections = ["default"]
            log.color_print(f"<search> Using default collection </search>\n")
        
        accepted_results = []
        
        for collection in selected_collections:
            log.color_print(f"<search> Searching in [{collection}]... </search>\n")
            
            # 从向量数据库检索结果
            retrieved_results = self.vector_db.search_data(
                collection=collection, vector=query_vector, top_k=self.top_k_per_section
            )
            
            if not retrieved_results or len(retrieved_results) == 0:
                log.color_print(f"<search> No relevant document chunks found in '{collection}'! </search>\n")
                continue
                
            # 基于查询的相关性对结果进行重排序
            for retrieved_result in retrieved_results:
                rerank_prompt = RERANK_PROMPT.format(
                    query=enhanced_query,
                    retrieved_chunk=f"<chunk>{retrieved_result.text}</chunk>"
                )
                
                chat_response = self.llm.chat(
                    messages=[{"role": "user", "content": rerank_prompt}]
                )
                consumed_tokens += chat_response.total_tokens
                response_content = chat_response.content.strip()
                
                if "YES" in response_content and "NO" not in response_content:
                    accepted_results.append(retrieved_result)
            
            if len(accepted_results) > 0:
                log.color_print(f"<search> Accepted {len(accepted_results)} document chunks from '{collection}' </search>\n")
        
        # 去重结果
        accepted_results = deduplicate_results(accepted_results)
        return accepted_results, consumed_tokens
    
    def _generate_section_content(self, section: str, topic: str, retrieved_results: List[RetrievalResult]) -> Tuple[str, int]:
        """
        Generate content for a specific section based on retrieved results.
        
        Args:
            section: Section name
            topic: Research topic
            retrieved_results: List of retrieved results
            
        Returns:
            Tuple of (section content, tokens used)
        """
        if not retrieved_results or len(retrieved_results) == 0:
            content = f"No relevant information found for section '{section}'."
            return content, 0
            
        # Format retrieved content for the prompt
        chunk_texts = []
        for i, result in enumerate(retrieved_results):
            if self.text_window_splitter and "wider_text" in result.metadata:
                chunk_texts.append(f"[{i+1}] Source: {result.reference}\n{result.metadata['wider_text']}")
            else:
                chunk_texts.append(f"[{i+1}] Source: {result.reference}\n{result.text}")
        
        retrieved_content = "\n\n".join(chunk_texts)
        
        # Generate section content
        prompt = SECTION_GENERATION_PROMPT.format(
            section_name=section,
            topic=topic,
            retrieved_content=retrieved_content
        )
        
        log.color_print(f"<think> Generating content for section '{section}'... </think>\n")
        response = self.writing_llm.chat([{"role": "user", "content": prompt}])
        
        return response.content, response.total_tokens
    
    def _polish_full_text(self, topic: str, full_text: str) -> Tuple[str, int]:
        """
        Polish the full text of the review.
        
        Args:
            topic: Research topic
            full_text: Full text of the review
            
        Returns:
            Tuple of (polished text, tokens used)
        """
        prompt = POLISH_PROMPT.format(topic=topic, full_text=full_text)
        
        log.color_print("<think> Polishing the full text... </think>\n")
        response = self.writing_llm.chat([{"role": "user", "content": prompt}])
        
        return response.content, response.total_tokens
    
    async def generate_overview(self, topic: str, **kwargs) -> Tuple[Dict[str, str], Dict[str, str], int]:
        """
        Generate a comprehensive overview of the given research topic.
        
        Args:
            topic: Research topic
            
        Returns:
            Tuple of (English sections, Chinese sections, total tokens used)
        """
        # Detect language and translate if needed
        topic_language = self._detect_language(topic)
        if topic_language in ["zh", "mixed"]:
            log.color_print(f"<think> Translating topic from {topic_language} to English... </think>\n")
            english_topic = self._translate_to_english(topic)
        else:
            english_topic = topic
            
        log.color_print(f"<query> Generating overview for: {english_topic} </query>\n")
        
        # Generate section queries
        section_queries = self._generate_section_queries(english_topic)
        
        # Process each section
        english_sections = {}
        total_tokens = 0
        
        for section in self.sections:
            if section not in section_queries:
                log.warning(f"No query found for section: {section}")
                english_sections[section] = f"No content generated for section '{section}'."
                continue
                
            query_info = section_queries[section]
            query = query_info["query"]
            conditions = query_info.get("conditions", [])
            
            # Search for relevant content
            retrieved_results, search_tokens = await self._search_for_section(
                section, query, conditions
            )
            total_tokens += search_tokens
            
            # Generate section content
            section_content, content_tokens = self._generate_section_content(
                section, english_topic, retrieved_results
            )
            total_tokens += content_tokens
            
            english_sections[section] = section_content
            
        # Combine all sections into full text
        full_text = ""
        for section in self.sections:
            full_text += f"## {section}\n\n{english_sections[section]}\n\n"
            
        # Polish the full text
        polished_text, polish_tokens = self._polish_full_text(english_topic, full_text)
        total_tokens += polish_tokens
        
        # Parse sections from polished text
        import re
        polished_sections = {}
        section_pattern = r"## (.*?)\n\n(.*?)(?=\n\n## |$)"
        for match in re.finditer(section_pattern, polished_text, re.DOTALL):
            section_name = match.group(1).strip()
            section_content = match.group(2).strip()
            polished_sections[section_name] = section_content
            
        # Translate each section to Chinese
        chinese_sections = {}
        for section, content in polished_sections.items():
            log.color_print(f"<think> Translating section '{section}' to Chinese... </think>\n")
            chinese_sections[section] = self._translate_to_chinese(content)
            
        return polished_sections, chinese_sections, total_tokens
        
    def query(self, query: str, **kwargs) -> Tuple[str, List[RetrievalResult], int]:
        """
        Process a research topic query and generate a comprehensive overview.
        
        Args:
            query: The research topic query
            
        Returns:
            Tuple of (response text, retrieval results (empty list), tokens used)
        """
        # This method overrides the RAGAgent query method
        english_sections, chinese_sections, total_tokens = asyncio.run(
            self.generate_overview(query, **kwargs)
        )
        
        # Format the response with both English and Chinese content
        response = f"# 综述：{query}\n\n"
        
        for section in self.sections:
            if section in english_sections and section in chinese_sections:
                response += f"## {section}\n\n"
                response += f"### English\n{english_sections[section]}\n\n"
                response += f"### 中文\n{chinese_sections[section]}\n\n"
                
        return response, [], total_tokens
        
    def retrieve(self, query: str, **kwargs) -> Tuple[List[RetrievalResult], int, dict]:
        """
        This method is required by the RAGAgent interface but not used directly.
        Instead, the query method handles the entire process.
        """
        # This is just a placeholder to satisfy the RAGAgent interface
        return [], 0, {}
        
    def _format_chunk_texts(self, chunk_texts: List[str]) -> str:
        """
        Format chunk texts for inclusion in prompts.
        
        Args:
            chunk_texts: List of text chunks
            
        Returns:
            Formatted string of chunk texts
        """
        chunk_str = ""
        for i, chunk in enumerate(chunk_texts):
            chunk_str += f"""<chunk_{i+1}>\n{chunk}\n</chunk_{i+1}>\n"""
        return chunk_str 