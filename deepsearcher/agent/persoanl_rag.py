"""
Personal Research Overview RAG Agent Module.

This module provides the PersonalRAG class for generating comprehensive
academic reviews on specific researchers' work. It follows a structured approach
to create well-organized research overviews by analyzing the researcher's
publications and generating insights about their academic journey and contributions.
"""

import asyncio
from typing import List, Tuple, Dict, Any, Optional
from datetime import datetime, timedelta
import re

from deepsearcher.agent.base import RAGAgent, describe_class
from deepsearcher.agent.academic_translator import AcademicTranslator
from deepsearcher.embedding.base import BaseEmbedding
from deepsearcher.llm.base import BaseLLM
from deepsearcher.tools import log
from deepsearcher.vector_db import RetrievalResult
from deepsearcher.vector_db.base import BaseVectorDB, deduplicate_results
from deepsearcher.agent.collection_router import CollectionRouter
from deepsearcher.rbase_db_loading import get_mysql_connection
from deepsearcher.agent.overview_rag import OverviewRAG
from tqdm import tqdm

# Author name extraction prompt
AUTHOR_EXTRACT_PROMPT = """
Extract the name of the researcher and the language of the name from the following query.
Return a JSON object with the following structure:
{{
    "name": "name of the researcher",
    "language": "en" or "zh"
}}

Query: {query}

Return only the JSON object without any explanations.
"""

# Section content generation prompts
ACADEMIC_GENE_PROMPT = """
You are an expert academic researcher analyzing a researcher's academic background and expertise.
Based on the provided publication data, write a comprehensive section about the researcher's academic gene map.

Researcher: {researcher_name}
Publications:
{publications}

Guidelines:
1. Analyze the researcher's core research areas and expertise
2. Identify their academic lineage and influences
3. Map their research interests and specializations
4. Highlight their unique academic characteristics
5. Use appropriate academic language and maintain a scholarly tone
6. Properly cite sources using [X] format
7. Maintain appropriate length (approximately 800-1200 words)

Your response should be a polished section ready for inclusion in the final review.
"""

RESEARCH_EVOLUTION_PROMPT = """
You are an expert academic researcher analyzing a researcher's academic evolution.
Based on the provided publication data, write a comprehensive section about the researcher's research evolution path.

Researcher: {researcher_name}
Publications:
{publications}

Guidelines:
1. Analyze the chronological development of their research interests
2. Identify key turning points and paradigm shifts
3. Map their research trajectory and progression
4. Highlight major methodological and theoretical developments
5. Use appropriate academic language and maintain a scholarly tone
6. Properly cite sources using [X] format
7. Maintain appropriate length (approximately 800-1200 words)

Your response should be a polished section ready for inclusion in the final review.
"""

CORE_CONTRIBUTIONS_PROMPT = """
You are an expert academic researcher analyzing a researcher's core contributions.
Based on the provided publication data, write a comprehensive section about the researcher's core contribution cube.

Researcher: {researcher_name}
Publications:
{publications}

Guidelines:
1. Identify and analyze their key theoretical contributions
2. Evaluate their methodological innovations
3. Assess their practical applications and impact
4. Highlight their unique research perspectives
5. Use appropriate academic language and maintain a scholarly tone
6. Properly cite sources using [X] format
7. Maintain appropriate length (approximately 800-1200 words)

Your response should be a polished section ready for inclusion in the final review.
"""

ACADEMIC_INFLUENCE_PROMPT = """
You are an expert academic researcher analyzing a researcher's academic influence.
Based on the provided publication data, write a comprehensive section about the researcher's academic influence network.

Researcher: {researcher_name}
Publications:
{publications}

Guidelines:
1. Analyze their citation impact and academic reach
2. Evaluate their influence on specific research fields
3. Assess their collaboration networks
4. Highlight their role in shaping research directions
5. Use appropriate academic language and maintain a scholarly tone
6. Properly cite sources using [X] format
7. Maintain appropriate length (approximately 800-1200 words)

Your response should be a polished section ready for inclusion in the final review.
"""

FUTURE_POTENTIAL_PROMPT = """
You are an expert academic researcher analyzing a researcher's future potential.
Based on the provided publication data, write a comprehensive section about the researcher's future potential prediction.

Researcher: {researcher_name}
Publications:
{publications}

Guidelines:
1. Analyze emerging research trends in their work
2. Identify potential future research directions
3. Evaluate their capacity for innovation
4. Assess their potential impact on the field
5. Use appropriate academic language and maintain a scholarly tone
6. Properly cite sources using [X] format
7. Maintain appropriate length (approximately 800-1200 words)

Your response should be a polished section ready for inclusion in the final review.
"""

# Question generation prompt
QUESTION_GENERATION_PROMPT = """
Based on the following section content, generate 3-5 specific research questions that would help deepen the analysis.
The questions should be focused on the researcher's work and potential future directions.

Section Title: {section_title}
Section Content: {section_content}

Generate questions that:
1. Are specific and focused
2. Address gaps in the current analysis
3. Could lead to new insights
4. Are answerable through academic research
5. Could guide future research directions

Return only the questions, one per line, without numbering or additional text.
"""

# Section optimization prompt
SECTION_OPTIMIZATION_PROMPT = """
You are an expert academic researcher tasked with optimizing a section of a research overview.
Based on the original section and additional research findings, improve the section's content and potentially its title.

Original Section Title: {section_title}
Original Section Content: {section_content}

Additional Research Findings:
{additional_findings}

Guidelines for Improvement:
1. Enhance the depth and breadth of analysis
2. Incorporate new insights from additional research
3. Consider if the section title needs revision
4. Maintain logical flow and coherence
5. Ensure proper citation of sources
6. Keep appropriate academic tone
7. Maintain appropriate length

Your response should include:
1. A suggested section title (if different from original)
2. The optimized section content

Format your response as:
TITLE: [Suggested title]
CONTENT: [Optimized content]
"""

# Structure division prompt
STRUCTURE_PROMPT = """
You are an academic research assistant tasked with planning a comprehensive literature review on a specific topic.

Generate appropriate search queries for each section of the literature review structure below. The goal is to retrieve relevant academic content from our knowledge base for each section.

Research Topic: <topic>

For each section, please provide:
1. A focused search query that will help retrieve the most relevant content from our academic database
2. Analyze the research topic whether has some condition requirements (e.g., time period, specific keywords, impact factor requirements, etc.)
    2.1 if the research topic requires a specific time period, please add 'pubdate' as a condition, which is an integer representing the timestamp(in seconds) of the public date
    2.2 if the research topic requires a specific impact factor, please add 'impact_factor' as a condition, which is a float number greater or equals to 0
    2.3 if the research topic requires one certain keyword or a group of specific keywords, and requires not to exclude any other paper without these keywords, please add 'keywords' as a condition, which is an array of strings
    2.4 conditions are only generated if the topic is required explicitly, otherwise, the conditions should be generated,
        examples:
            if the research topic is "please write a review about the topic of 'planktonic microbial community'", the conditions should be empty 
            if the research topic is "please write a review about the topic of 'planktonic microbial community' and the papers should be published after 2020", the conditions should be 'pubdate >= 1577836800'
            if the research topic is "please write a review about the topic of 'planktonic microbial community' and the papers should be published after 2020 and the impact factor should be greater than 10", the conditions should be 'pubdate >= 1577836800 AND impact_factor >= 10'
            if the research topic is "please write a review about the topic of 'planktonic microbial community' from papers including 'bacteria' or 'virus'", the conditions should be 'ARRAY_CONTAINS_ANY(keywords, ["bacteria", "virus"])'
    2.5 an exception is that for emerging trends or future directions, the condition for pubdate in the latest 5 years should be added automatically

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
        "conditions": "condition_expression"
    },
    "Theoretical Foundations": {
        "query": "search query for theoretical foundations",
        "conditions": "condition_expression"
    },
    ...
}

Condition Syntax:
1. conditions are descripted as plain text which is similar to SQL syntax
2. for integer condition, the format is like 'pubdate >= 1741996800' (which stands for pubdate is later than 2025-03-15)
3. for float condition, the format is like 'impact_factor >= 10' (which stands for impact factor is greater or equals to 10)
4. for a range requirement, the operator can be written in a signle condition, like '1741996800 <= pubdate <= 1742083200' (which stands for pubdate is between 2025-03-15 and 2025-03-16)
5. for an array condition, the format is like 'ARRAY_CONTAINS(keywords, "keyword1")', if the multiple keywords are required, 
   the format is like 'ARRAY_CONTAINS_ANY(keywords, ["keyword1", "keyword2"])' or
   'ARRAY_CONTAINS_ALL(keywords, ["keyword1", "keyword2"]) according to the detail requirement.
6. if several conditions are required, they should be connected by 'AND' or 'OR'

Ensure your queries are specific, academic in nature, and designed to retrieve comprehensive information for each section.
Output the JSON response directly without any comments or explanations.
"""

# Section content generation prompt
SECTION_GENERATION_PROMPT = """
You are an academic writer specializing in creating comprehensive literature reviews. Based on the retrieved academic content, write a detailed section for a literature review.

Section: {section_name}
Topic: {topic}

Retrieved Content:
{retrieved_content}

Guidelines:
1. Write a cohesive, well-structured section that thoroughly covers the topic based on the retrieved content
2. Use appropriate academic language and maintain a scholarly tone
3. Properly cite sources within the text using the format [X], where X corresponds to the chunk Reference ID from the retrieved content
4. Synthesize information rather than merely summarizing individual sources
5. Highlight consensus views as well as contrasting perspectives in the field
6. Maintain appropriate length for a section in a comprehensive literature review (approximately 800-1200 words)
7. Ensure logical flow within the section

Your response should be a polished section ready for inclusion in the final literature review.
"""

# Complete the final paper prompt
COMPILE_REVIEW_PROMPT = """
You are a senior academic researcher and you have deeply researched about the topic from several aspects.
Now you need to complete the final paper with your research drafts that are given in the following Draft Sections.
Meanwhile you are specializing in polishing scholarly literature reviews. 

Guidelines for Improvement:
1. Ensure logical flow and coherence throughout the entire document
2. Eliminate any redundancies or repetitive content
3. Some conclusions in a section are not neccessay, as we will provide a conculusion in the end of the review
4. Check for and correct any logical inconsistencies or structural problems
5. Improve transitions between sections
6. Enhance clarity and precision of language
7. Maintain consistent academic tone and style throughout
8. Ensure appropriate depth of analysis in each section
9. Do not change the citations format [X] and X value, where X corresponds to the chunk Reference ID from the retrieved content
10. Keep the overall content and organization, making only improvements to quality rather than substantive changes

Review the complete draft of this literature review and improve it for publication quality.

Reaserch Topic: {topic}

Draft Sections for the Literature Review:
{draft_text}

Your response should be the complete, polished literature review ready for submission.
"""

# Language detection prompt
LANGUAGE_DETECT_PROMPT = """
Determine the primary language of the following text. Return only the language code:
- "en" for English
- "zh" for Chinese
- "mixed" if the text contains a significant amount of both languages

Text: {text}

Language code:
"""

# Search result reranking prompt
RERANK_PROMPT = """
Based on the query and the retrieved chunk, determine whether the chunk is helpful in addressing the query. Respond with only "YES" or "NO".

Query: {query}
Retrieved Chunk: {retrieved_chunk}

Is the chunk helpful for addressing the query?
"""

# Text cleaning prompt
CLEAN_TEXT_PROMPT = """
Clean and optimize the following academic text by following these specific rules:

1. Remove (do not complete) incomplete sentences at the beginning or end of the text
2. Remove references or citations in the format of: author(s) + title + journal + year + DOI/URL
3. Remove any meaningless text, formatting artifacts, or irrelevant metadata
4. Maintain the academic integrity and completeness of the meaningful content
5. Keep complete paragraphs and well-formed sentences
6. Do not add any new content or complete partial ideas

Return only the cleaned text without any explanations or markup.

Text: {text}
"""

# Abstract and conclusion generation prompt
ABSTRACT_CONCLUSION_PROMPT = """
You are an expert academic researcher who specializes in writing research papers and literature reviews.
Based on the provided literature review content, please generate two distinct sections:

1. Abstract:
- Write a concise summary (200-300 words) of the entire review
- Include the main research topic, key findings, and significant conclusions
- Follow standard academic abstract format
- Focus on the most important aspects of the review

2. Conclusion:
- Write a comprehensive conclusion (300-400 words) that synthesizes the main points
- Highlight the key contributions and implications of the research
- Discuss potential future research directions
- Maintain academic tone and style

Research Topic: {topic}

Literature Review Content:
{review_content}

Please format your response as follows:

ABSTRACT:
[Your abstract text here]

CONCLUSION:
[Your conclusion text here]
"""

@describe_class(
    "This agent is designed to generate comprehensive academic reviews on specific researchers' work, analyzing their publications and generating insights about their academic journey."
)
class PersonalRAG(OverviewRAG):
    """
    PersonalRAG agent for generating comprehensive academic reviews on specific researchers.
    
    This agent follows a structured approach to create well-organized research overviews
    by analyzing the researcher's publications and generating insights about their academic
    journey and contributions.
    
    Unlike the general OverviewRAG which focuses on topic-based research reviews,
    PersonalRAG specifically analyzes an individual researcher's work, including:
    - Academic background and expertise (Academic Gene Map)
    - Chronological development of research interests (Research Evolution Path)
    - Key theoretical and methodological contributions (Core Contribution Cube)
    - Citation impact and academic influence (Academic Influence Network)
    - Future research directions and potential (Future Potential Prediction)
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
        rbase_settings: dict = {},
        **kwargs,
    ):
        """
        Initialize the PersonalRAG agent.
        
        Args:
            llm: Base language model for general tasks
            reasoning_llm: Language model optimized for reasoning tasks
            writing_llm: Language model optimized for writing tasks
            translator: Academic translator for language translation
            embedding_model: Embedding model for vector encoding
            vector_db: Vector database for knowledge retrieval
            route_collection: Whether to use collection routing
            rbase_settings: Settings for database connection
        """
        super().__init__(
            llm=llm,
            reasoning_llm=reasoning_llm,
            writing_llm=writing_llm,
            translator=translator,
            embedding_model=embedding_model,
            vector_db=vector_db,
            route_collection=route_collection,
            rbase_settings=rbase_settings,
            **kwargs
        )
        
        # Define the standard structure for personal research reviews
        self.sections = [
            "Academic Gene Map",
            "Research Evolution Path",
            "Core Contribution Cube",
            "Academic Influence Network",
            "Future Potential Prediction"
        ]
        
        # Section prompts mapping
        self.section_prompts = {
            "Academic Gene Map": ACADEMIC_GENE_PROMPT,
            "Research Evolution Path": RESEARCH_EVOLUTION_PROMPT,
            "Core Contribution Cube": CORE_CONTRIBUTIONS_PROMPT,
            "Academic Influence Network": ACADEMIC_INFLUENCE_PROMPT,
            "Future Potential Prediction": FUTURE_POTENTIAL_PROMPT
        }

    def _extract_author_info(self, query: str) -> Dict[str, str]:
        """
        Extract author name and language from the query.
        
        Args:
            query: Input query containing author name
            
        Returns:
            Dictionary containing author name and language
        """
        prompt = AUTHOR_EXTRACT_PROMPT.format(query=query)
        
        try:
            # 尝试通过LLM提取作者信息
            response = self.llm.chat([{"role": "user", "content": prompt}])
            
            # 检查响应是否为空
            if not response or not response.content or response.content.strip() == "":
                log.warning("Empty response from LLM when extracting author info")
                return self._extract_fallback(query)
                
            # 解析JSON
            import json
            content = response.content.strip()
            
            # 处理可能的前缀和后缀
            if "```json" in content:
                content = content.split("```json", 1)[1]
            if "```" in content:
                content = content.split("```", 1)[0]
                
            # 移除开头可能的非JSON内容
            start_idx = content.find('{')
            end_idx = content.rfind('}')
            if start_idx != -1 and end_idx != -1:
                content = content[start_idx:end_idx+1]
            else:
                log.warning("No valid JSON format found in LLM response")
                return self._extract_fallback(query)
            
            # 尝试解析JSON
            try:
                author_info = json.loads(content)
                
                # 验证结果格式
                if "name" not in author_info or "language" not in author_info:
                    log.warning("JSON parsed but missing required fields")
                    return self._extract_fallback(query)
                    
                # 验证值不为空
                if not author_info["name"] or author_info["name"].strip() == "":
                    log.warning("Author name is empty in JSON response")
                    return self._extract_fallback(query)
                    
                return author_info
            except json.JSONDecodeError as je:
                log.error(f"Failed to parse author info JSON: {je}")
                return self._extract_fallback(query)
                
        except Exception as e:
            log.error(f"Error in author extraction: {e}")
            return self._extract_fallback(query)
            
    def _extract_fallback(self, query: str) -> Dict[str, str]:
        """
        回退方法，当JSON解析失败时使用规则提取作者
        
        Args:
            query: 原始查询
            
        Returns:
            包含作者信息的字典
        """
        log.warning("Using fallback author extraction")
        
        # 从查询中提取最后一个名词短语作为作者名
        name = ""
        language = "en"
        
        # 检查中文作者名模式（"关于XXX的"或"XXX教授的"等）
        if "关于" in query and "的" in query:
            parts = query.split("关于", 1)[1].split("的", 1)[0].strip()
            if parts:
                name = parts
                language = "zh"
        elif "教授" in query:
            parts = query.split("教授", 1)[0].strip().split()
            if parts:
                name = parts[-1]
                language = "zh"
        else:
            # 默认提取查询中的词组
            # 针对"请为我写一份关于于君教授的科研综述"这种模式特别处理
            if "写" in query and "关于" in query and "教授" in query:
                start = query.find("关于") + 2
                end = query.find("教授")
                if start < end:
                    name = query[start:end].strip()
                    language = "zh"
            else:
                # 提取最后一个名词
                parts = query.split()
                if len(parts) > 0:
                    name = parts[-1]
                    # 检测是否为中文名（简单判断，不完美）
                    if any('\u4e00' <= char <= '\u9fff' for char in name):
                        language = "zh"
                    else:
                        language = "en"
                        
        log.info(f"Extracted author name: {name}, language: {language}")
        return {"name": name, "language": language}

    def _get_author_id(self, author_info: Dict[str, str]) -> Optional[int]:
        """
        Get author ID from database based on name and language.
        
        Args:
            author_info: Dictionary containing author name and language
            
        Returns:
            Author ID if found, None otherwise
        """
        conn = get_mysql_connection(self.rbase_settings.get("database", {}))
        try:
            with conn.cursor() as cursor:
                if author_info["language"] == "en":
                    query = "SELECT id FROM author WHERE name = %s"
                    cursor.execute(query, (author_info["name"],))
                else:
                    query = "SELECT id FROM author WHERE cname = %s"
                    cursor.execute(query, (author_info["name"],))
                
                result = cursor.fetchone()
                if result:
                    return result["id"]
                return None
        except Exception as e:
            log.error(f"Failed to get author ID: {e}")
            return None

    def _get_author_articles(self, author_id: int) -> List[Dict[str, Any]]:
        """
        Get author's articles from database.
        
        Args:
            author_id: Author ID
            
        Returns:
            List of article dictionaries
        """
        conn = get_mysql_connection(self.rbase_settings.get("database", {}))
        try:
            with conn.cursor() as cursor:
                # Get articles ordered by impact factor and pubdate
                query = """
                    SELECT a.id, a.title, a.journal_name, a.pubdate, a.doi, a.summary, a.impact_factor
                    FROM article a
                    JOIN author_article aa ON a.id = aa.article_id
                    WHERE aa.author_id = %s AND a.raw_article_id = 0
                    ORDER BY a.impact_factor DESC, a.pubdate DESC
                    LIMIT %s
                """
                cursor.execute(query, (author_id, self.max_articles))
                articles = cursor.fetchall()
                
                # Get recent articles
                recent_date = datetime.now() - timedelta(days=int(self.recent_months * 30.5))
                recent_query = """
                    SELECT a.id, a.title, a.journal_name, a.pubdate, a.doi, a.summary, a.impact_factor
                    FROM article a
                    JOIN author_article aa ON a.id = aa.article_id
                    WHERE aa.author_id = %s AND a.raw_article_id = 0 AND a.pubdate >= %s
                    ORDER BY a.pubdate DESC
                    LIMIT %s
                """
                cursor.execute(recent_query, (author_id, recent_date, self.max_articles))
                recent_articles = cursor.fetchall()
                
                # Merge and deduplicate articles
                all_articles = []
                seen_ids = set()
                
                # First add recent articles
                for article in recent_articles:
                    if article["id"] not in seen_ids:
                        all_articles.append(article)
                        seen_ids.add(article["id"])
                
                # Then add other articles
                for article in articles:
                    if len(all_articles) >= self.max_articles:
                        break
                    if article["id"] not in seen_ids:
                        all_articles.append(article)
                        seen_ids.add(article["id"])
                
                return all_articles
        except Exception as e:
            log.error(f"Failed to get author articles: {e}")
            return []

    def _format_publications_for_prompt(self, articles: List[Dict[str, Any]]) -> str:
        """
        Format articles for inclusion in prompts.
        
        Args:
            articles: List of article dictionaries
            
        Returns:
            Formatted string of articles
        """
        formatted_articles = []
        for article in articles:
            formatted_article = f"""
Title: {article['title']}
Journal: {article['journal_name']}
Impact Factor: {article['impact_factor']}
Publication Date: {article['pubdate']}
Summary: {article['summary']}
"""
            formatted_articles.append(formatted_article)
        return "\n".join(formatted_articles)

    def _generate_section_content(self, section: str, researcher_name: str, articles: List[Dict[str, Any]]) -> Tuple[str, int]:
        """
        Generate content for a specific section.
        
        Args:
            section: Section name
            researcher_name: Name of the researcher
            articles: List of article dictionaries
            
        Returns:
            Tuple of (section content, tokens used)
        """
        if not articles:
            content = f"No relevant information found for section '{section}'."
            return content, 0
            
        # Format publications for the prompt
        publications = self._format_publications_for_prompt(articles)
        
        # Get the appropriate prompt for this section
        prompt_template = self.section_prompts.get(section, "")
        if not prompt_template:
            return f"No prompt template found for section '{section}'.", 0
        
        # Generate section content
        prompt = prompt_template.format(
            researcher_name=researcher_name,
            publications=publications
        )
        
        log.color_print(f"<writing> Generating content for section '{section}'... </writing>\n")
        response = self.writing_llm.chat([{"role": "user", "content": prompt}])
        
        return response.content, response.total_tokens
    
    def _generate_references(self, articles: List[Dict[str, Any]]) -> str:
        """
        Generate a formatted reference list from author's articles.
        
        Args:
            articles: List of article dictionaries
            
        Returns:
            Formatted reference list as a string
        """
        log.color_print("<optimizing> Generating references from articles... </optimizing>\n")
        references = []
        
        for i, article in enumerate(articles):
            # Process publication year
            try:
                year = article['pubdate'].year
            except:
                year = "n.d." # no date
                
            # Generate reference entry
            try:
                authors = article.get('authors', '').split(',')
                if len(authors) > 5:
                    authors = authors[:5] + ['et al']
                authors_str = ', '.join([a.strip() for a in authors if a.strip()])
                if not authors_str:
                    authors_str = "Author(s)"
                    
                reference = f"[{i + 1}] {authors_str}. ({year}). {article['title']}. {article['journal_name']}. DOI: {article['doi']}"
                references.append(reference)
            except Exception as e:
                log.error(f"Error formatting reference for article {article.get('id', 'unknown')}: {e}")
                references.append(f"[{i + 1}] Reference information unavailable. Article ID: {article.get('id', 'unknown')}")
                
        return "\n\n".join(references)
        
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

    async def generate_overview(self, query: str, **kwargs) -> Tuple[Dict[str, str], Dict[str, str], int]:
        """
        Generate a comprehensive overview of the researcher's work.
        
        Args:
            query: Query containing researcher's name
            
        Returns:
            Tuple of (compiled English sections, Chinese translated sections, total tokens used)
        """
        # Extract author information
        author_info = self._extract_author_info(query)
        if not author_info["name"]:
            raise ValueError("Could not extract author name from query")
            
        # Get author ID
        author_id = self._get_author_id(author_info)
        if not author_id:
            raise ValueError(f"No author found in database for name: {author_info['name']}")
            
        # Get author's articles
        articles = self._get_author_articles(author_id)
        if not articles:
            raise ValueError(f"No articles found for author: {author_info['name']}")
        
        # Process each section
        english_sections = {}
        total_tokens = 0
        
        for section in self.sections:
            # Generate initial section content
            section_content, content_tokens = self._generate_section_content(
                section, author_info["name"], articles
            )
            total_tokens += content_tokens
            
            # Generate questions for the section
            questions = self._generate_questions(section, section_content)
            
            # Search for additional content based on questions
            additional_results = await self._search_for_questions(questions, author_id)
            
            # Optimize section with additional findings
            optimized_title, optimized_content, optimize_tokens = self._optimize_section(
                section, section_content, additional_results
            )
            total_tokens += optimize_tokens
            
            english_sections[optimized_title] = optimized_content
        
        # Generate abstract and conclusion
        abstract, conclusion, abstract_tokens = self._generate_abstract_and_conclusion(
            author_info["name"], "\n\n".join(english_sections.values())
        )
        total_tokens += abstract_tokens
        
        # Reorganize references
        reorganized_text, references_text, ref_tokens = self._reorganize_references(
            "\n\n".join(english_sections.values())
        )
        
        # If no references found in text, generate from articles
        if not references_text:
            references_text = self._generate_references(articles)
            
        total_tokens += ref_tokens
        
        # Add abstract, conclusion and references
        english_sections["Abstract"] = abstract
        english_sections["Conclusion"] = conclusion
        english_sections["References"] = references_text
            
        # Translate each section to Chinese
        chinese_sections = {}
        for section, content in english_sections.items():
            if section != "References":  # Don't translate references
                log.color_print(f"<translating> Translating section '{section}' to Chinese... </translating>\n")
                chinese_sections[section] = self._translate_to_chinese(content)
            else:
                chinese_sections[section] = content
            
        return english_sections, chinese_sections, total_tokens
        
    def query(self, query: str, **kwargs) -> Tuple[str, List[RetrievalResult], int]:
        """
        Process a researcher query and generate a comprehensive overview.
        
        Args:
            query: The researcher query
            
        Returns:
            Tuple of (response text, retrieval results (empty list), tokens used)
        """
        if kwargs.get("verbose"):
            self.verbose = True
        if kwargs.get("max_articles"):
            self.max_articles = kwargs.get("max_articles")
        if kwargs.get("recent_months"):
            self.recent_months = kwargs.get("recent_months")
            
        try:
            # Generate overview
            english_sections, chinese_sections, total_tokens = asyncio.run(
                self.generate_overview(query, **kwargs)
            )
        
            # Format the response with both English and Chinese content
            self.english_response = f"# Research Overview: {query}\n\n"
            self.chinese_response = f"# 研究综述：{query}\n\n"
        
            # Add sections in order
            overview_sections = ['Abstract']
            overview_sections.extend(self.sections)
            overview_sections.append('Conclusion')
            overview_sections.append('References')
                
            for section in overview_sections:
                if section in english_sections:
                    self.english_response += f"## {section}\n\n"
                    self.english_response += f"{english_sections[section]}\n\n"
                        
                # Add translated section to Chinese response
                if section in chinese_sections:
                    self.chinese_response += self.translator.translate(f"## {section}", "zh") + "\n\n"
                    self.chinese_response += f"{chinese_sections[section]}\n\n"
                    
            return self.english_response, [], total_tokens
            
        except Exception as e:
            error_message = f"Error generating research overview: {str(e)}"
            log.error(error_message)
            return error_message, [], 0
        
    def retrieve(self, query: str, **kwargs) -> Tuple[List[RetrievalResult], int, dict]:
        """
        This method is required by the RAGAgent interface but not used directly.
        Instead, the query method handles the entire process.
        """
        return [], 0, {}