"""
Overview RAG Agent Module.

This module provides the OverviewRAG class for generating comprehensive
academic reviews on specified research topics. It follows a structured approach
to create well-organized research overviews by querying knowledge bases for
relevant information for each section of the review.
"""

import asyncio
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

from deepsearcher.agent.academic_translator import AcademicTranslator
from deepsearcher.agent.base import RAGAgent, describe_class
from deepsearcher.agent.collection_router import CollectionRouter
from deepsearcher.db.mysql_connection import get_mysql_connection
from deepsearcher.embedding.base import BaseEmbedding
from deepsearcher.llm.base import BaseLLM
from deepsearcher.tools import log
from deepsearcher.vector_db import RetrievalResult
from deepsearcher.vector_db.base import BaseVectorDB, deduplicate_results

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

REWRITE_SEARCH_QUERY_PROMPT = """
You are an academic research assistant tasked with planning a comprehensive literature review on a specific topic.

For the given topic, you have already generated a search query for a specific section.
But with this query, we can search few relevant content from vector database. 
It may be due to the query is not specific enough, or the query is not related enough to the topic.
Please rewrite the query to make it more specific and related to the topic.

Topic: {topic}
Section: {section}
Original Query: {query}

Please output the rewritten search query directly.
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
        rbase_settings: dict = {},
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
        """
        self.llm = llm
        self.reasoning_llm = reasoning_llm
        self.writing_llm = writing_llm
        self.translator = translator
        self.embedding_model = embedding_model
        self.vector_db = vector_db
        self.route_collection = route_collection
        self.rbase_settings = rbase_settings
        if route_collection:
            self.collection_router = CollectionRouter(llm=self.llm, vector_db=self.vector_db)
        else:
            self.collection_router = None

        if kwargs.get("top_k_per_section"):
            self.top_k_per_section = kwargs.get("top_k_per_section")
        else:
            self.top_k_per_section = 20

        if kwargs.get("top_k_accepted_results"):
            self.top_k_accepted_results = kwargs.get("top_k_accepted_results")
        else:
            self.top_k_accepted_results = 20

        if kwargs.get("vector_db_collection"):
            self.vector_db_collection = kwargs.get("vector_db_collection")
        else:
            self.vector_db_collection = "default"

        # Define the standard structure for academic reviews
        self.sections = [
            "Introduction",
            "Theoretical Foundations",
            "Methodological Approaches",
            "Key Findings & Debates",
            "Emerging Trends",
            "Research Gaps & Future Directions",
        ]
        # Final English and Chinese reviews
        self.english_response = ""
        self.chinese_response = ""
        # Whether to print detailed logs
        self.verbose = False

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

    def _clean_chunk_text(self, text: str) -> Tuple[str, int]:
        """
        Clean and optimize a text chunk by removing incomplete sentences and meaningless text.

        Args:
            text: The text to clean

        Returns:
            Tuple of (cleaned text, tokens used)
        """
        prompt = CLEAN_TEXT_PROMPT.format(text=text)

        response = self.llm.chat([{"role": "user", "content": prompt}])
        cleaned_text = response.content.strip()

        return cleaned_text, response.total_tokens

    def _translate_to_english(self, text: str, user_dict: List[dict] = None) -> str:
        """
        Translate the input text to English.

        Args:
            text: Text to translate

        Returns:
            Translated text in English
        """
        return self.translator.translate(text, "en", user_dict)

    def _translate_to_chinese(self, text: str, user_dict: List[dict] = None) -> str:
        """
        Translate the input text to Chinese.

        Args:
            text: Text to translate

        Returns:
            Translated text in Chinese
        """
        return self.translator.translate(text, "zh", user_dict)

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
            # Try multiple methods to parse the dictionary format returned by LLM
            import ast
            import json
            import re

            # Remove any extra content that might exist
            content = response.content.strip()

            # Method 1: Try direct parsing with ast.literal_eval
            try:
                queries_dict = ast.literal_eval(content)
                return queries_dict
            except (SyntaxError, ValueError):
                pass

            # Method 2: Use regex to extract the dictionary part
            try:
                dict_pattern = r"\{[\s\S]*\}"
                dict_match = re.search(dict_pattern, content)
                if dict_match:
                    dict_str = dict_match.group(0)
                    queries_dict = ast.literal_eval(dict_str)
                    return queries_dict
            except (SyntaxError, ValueError):
                pass

            # Method 3: Try json parsing
            try:
                # Remove possible markdown code block markers
                json_content = re.sub(r"```(?:json|python)?|```", "", content).strip()
                queries_dict = json.loads(json_content)
                return queries_dict
            except json.JSONDecodeError:
                pass

            # All parsing methods failed, use default values
            log.warning("Could not parse LLM dictionary format, using default queries")
            return {
                section: {"query": f"{topic} {section.lower()}", "conditions": []}
                for section in self.sections
            }
        except Exception as e:
            log.critical(f"Failed to generate queries: {e}")
            # Use basic structure as fallback
            return {
                section: {"query": f"{topic} {section.lower()}", "conditions": []}
                for section in self.sections
            }

    async def _search_for_section(
        self, section: str, query: str, filter: Optional[str] = ""
    ) -> Tuple[List[RetrievalResult], int]:
        """
        Search for content relevant to a specific section.

        Args:
            section: Section name
            query: Search query
            conditions: Optional list of search conditions

        Returns:
            Tuple of (retrieved results, tokens used)
        """
        log.color_print(
            f"<search> Searching for section '{section}' with query: '{query}' </search>\n"
        )

        query_vector = self.embedding_model.embed_query(query)
        consumed_tokens = 0

        # Determine which collections to search
        if self.route_collection:
            # Use CollectionRouter to select appropriate collections
            selected_collections, n_token_route = self.collection_router.invoke(query=query)
            consumed_tokens += n_token_route
            log.color_print(
                f"<search> Collection router selected: {selected_collections} </search>\n"
            )
        else:
            # Use default collection
            selected_collections = [self.vector_db_collection]

        accepted_results = []

        for collection in selected_collections:
            # Retrieve results from vector database
            retrieved_results = self.vector_db.search_data(
                collection=collection,
                vector=query_vector,
                top_k=self.top_k_per_section,
                filter=filter,
            )

            if self.verbose:
                log.debug(
                    f"{len(retrieved_results)} chunks retrived in '{collection}' for query: '{query}'"
                )

            if not retrieved_results or len(retrieved_results) == 0:
                log.color_print(
                    f"<search> No relevant document chunks found in '{collection}'! </search>\n"
                )
                continue

            # Rerank results based on query relevance
            for retrieved_result in tqdm(retrieved_results, desc="Reranking results"):
                rerank_prompt = RERANK_PROMPT.format(
                    query=query, retrieved_chunk=f"<chunk>{retrieved_result.text}</chunk>"
                )

                chat_response = self.llm.chat(messages=[{"role": "user", "content": rerank_prompt}])
                consumed_tokens += chat_response.total_tokens
                response_content = chat_response.content.strip()

                if "YES" in response_content and "NO" not in response_content:
                    # Clean text, remove incomplete or meaningless content
                    cleaned_text, clean_tokens = self._clean_chunk_text(retrieved_result.text)
                    consumed_tokens += clean_tokens

                    # Update the retrieved result text
                    retrieved_result.text = cleaned_text
                    accepted_results.append(retrieved_result)

            if self.verbose:
                log.debug(
                    f"{len(accepted_results)} chunks accepted from '{collection}' for query: '{query}'"
                )

            if len(accepted_results) > 0:
                log.color_print(
                    f"<search> Accepted {len(accepted_results)} document chunks from '{collection}' </search>\n"
                )

        # Deduplicate results
        accepted_results = deduplicate_results(accepted_results)

        # If results exceed limit, sort by score and truncate
        if len(accepted_results) > self.top_k_accepted_results:
            # Sort by score (higher scores first)
            accepted_results.sort(key=lambda x: x.score, reverse=True)
            # Take top_k_accepted_results items
            accepted_results = accepted_results[: self.top_k_accepted_results]
            if self.verbose:
                log.debug(f"truncated to {self.top_k_accepted_results} top scoring results")

        return accepted_results, consumed_tokens

    def _generate_section_content(
        self, section: str, topic: str, retrieved_results: List[RetrievalResult]
    ) -> Tuple[str, int]:
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
            chunk_texts.append(f"[{result.metadata['reference_id']}] \n{result.text}")

        retrieved_content = "\n\n".join(chunk_texts)

        # Generate section content
        prompt = SECTION_GENERATION_PROMPT.format(
            section_name=section, topic=topic, retrieved_content=retrieved_content
        )

        response = self.writing_llm.chat([{"role": "user", "content": prompt}])

        return response.content, response.total_tokens

    def _compile_final_review(self, topic: str, draft_text: str) -> Tuple[str, int]:
        """
        Compile and refine the final review from individual section drafts.

        This function takes the draft sections of the review and compiles them into
        a cohesive, well-structured final document. It improves the logical flow,
        eliminates redundancies, enhances transitions between sections, and ensures
        a consistent academic tone throughout the entire review.

        Args:
            topic: Research topic
            draft_text: Combined text of all section drafts

        Returns:
            Tuple of (compiled final review, tokens used)
        """
        prompt = COMPILE_REVIEW_PROMPT.format(topic=topic, draft_text=draft_text)

        response = self.writing_llm.chat([{"role": "user", "content": prompt}])

        return response.content, response.total_tokens

    def _generate_abstract_and_conclusion(
        self, topic: str, review_content: str
    ) -> Tuple[str, str, int]:
        """
        Generate abstract and conclusion sections for the literature review.

        Args:
            topic: Research topic
            review_content: The complete literature review content

        Returns:
            Tuple of (abstract text, conclusion text, tokens used)
        """
        prompt = ABSTRACT_CONCLUSION_PROMPT.format(topic=topic, review_content=review_content)

        response = self.reasoning_llm.chat([{"role": "user", "content": prompt}])

        # Parse the response to extract abstract and conclusion
        content = response.content.strip()
        import re

        abstract_match = re.search(r"ABSTRACT:\s*(.*?)(?=CONCLUSION:|$)", content, re.DOTALL)
        conclusion_match = re.search(r"CONCLUSION:\s*(.*?)$", content, re.DOTALL)

        abstract = abstract_match.group(1).strip() if abstract_match else ""
        conclusion = conclusion_match.group(1).strip() if conclusion_match else ""

        return abstract, conclusion, response.total_tokens

    def _reorganize_references(self, text: str) -> Tuple[str, str, int]:
        """
        Reorganize citations and generate a reference list.

        Extract citation IDs from the text, replace them with sequential numbers [1][2][3]...,
        and generate a corresponding reference list. Handles both single citations [123] and
        multiple citations [123, 456] or [123,456].

        Args:
            text: Text containing citations

        Returns:
            Tuple of (reorganized text, reference list, tokens used)
        """
        import re

        # First, convert multiple citations in a single bracket to separate citations
        # e.g., [123, 456] -> [123][456]
        text = re.sub(
            r"\[(\d+)\s*,\s*(\d+)\s*(?:,\s*\d+\s*)*\]",
            lambda m: "".join(f"[{id}]" for id in m.group(1).split(",") + m.group(2).split(",")),
            text,
        )

        # Extract all citation IDs
        reference_pattern = r"\[(\d+)\]"
        reference_ids = re.findall(reference_pattern, text)

        if not reference_ids:
            return text, "", 0

        # Deduplicate while maintaining order
        unique_ids = []
        seen = set()
        for ref_id in reference_ids:
            if ref_id not in seen:
                unique_ids.append(ref_id)
                seen.add(ref_id)

        conn = get_mysql_connection(self.rbase_settings.get("database", {}))
        # Get reference information from database
        references = []
        try:
            with conn.cursor() as cursor:
                for ref_id in unique_ids:
                    # Query article information from database
                    query = f"SELECT title, journal_name, authors, doi, pubdate FROM article WHERE id = {ref_id}"
                    cursor.execute(query)
                    article = cursor.fetchone()

                    if not article:
                        references.append(
                            f"some authors, some title, some journal, some year, some doi for some article {ref_id}"
                        )
                        continue

                    # Process author list
                    authors = article["authors"].split(",")
                    if len(authors) > 5:
                        authors = authors[:5] + ["et al"]
                    authors_str = ", ".join(authors)

                    # Process publication year
                    year = article["pubdate"].year

                    # Generate citation description
                    reference = f"[{len(references) + 1}] {authors_str}. {article['title']}. {article['journal_name']}. {year};{article['doi']}"
                    references.append(reference)
        except Exception as e:
            log.critical(f"Failed to get reference information from database: {e}")
            return text, "", 0

        # Replace citations in text
        new_text = text
        for i, ref_id in enumerate(unique_ids):
            new_text = new_text.replace(f"[{ref_id}]", f"[{i + 1}]")

        # Generate references list
        references_text = "\n\n".join(references)

        if self.verbose:
            log.debug(f"References list: {references_text}")
        return new_text, references_text, 0

    def _rewrite_search_query(
        self, topic: str, section: str, query: str
    ) -> Tuple[str, int]:
        """
        Generate abstract and conclusion sections for the literature review.

        Args:
            topic: Research topic
            review_content: The complete literature review content

        Returns:
            Tuple of (abstract text, conclusion text, tokens used)
        """
        prompt = REWRITE_SEARCH_QUERY_PROMPT.format(topic=topic, section=section, query=query)

        response = self.reasoning_llm.chat([{"role": "user", "content": prompt}])

        # Parse the response to extract abstract and conclusion
        content = response.content.strip()
        return content, response.total_tokens

    async def generate_overview(
        self, topic: str, **kwargs
    ) -> Tuple[Dict[str, str], Dict[str, str], int]:
        """
        Generate a comprehensive overview of the given research topic.

        Args:
            topic: Research topic

        Returns:
            Tuple of (compiled English sections, Chinese translated sections, total tokens used)
        """
        step = 1
        # Detect language and translate if needed
        topic_language = self._detect_language(topic)
        if topic_language in ["zh", "mixed"]:
            log.color_print(
                f"<Step {step}> Translating topic from {topic_language} to English... </Step {step}>\n"
            )
            step += 1
            english_topic = self._translate_to_english(topic)
        else:
            english_topic = topic

        log.color_print(f"<Step {step}> Generating overview article structure for: {english_topic} </Step {step}>\n")
        step += 1

        # Generate section queries
        section_queries = self._generate_section_queries(english_topic)

        # Process each section
        english_sections = {}
        total_tokens = 0

        log.color_print(f"<Step {step}> Search related contents for each section. </Step {step}>\n")
        step += 1
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

            if len(retrieved_results) <= 3:
                log.debug(f"Regenerate search query for section '{section}', original query: {query}")
                query, query_tokens = self._rewrite_search_query(english_topic, section, query)
                log.debug(f"Regenerated query: {query}")
                total_tokens += query_tokens
                secondary_retrieved_results, search_tokens = await self._search_for_section(
                    section, query, conditions
                )
                total_tokens += search_tokens
                retrieved_results.extend(secondary_retrieved_results)

            if len(retrieved_results) > 0:
                # Generate section content
                log.color_print(f"<writting> Generating content for section '{section}'... </writting>\n")
                section_content, content_tokens = self._generate_section_content(
                    section, english_topic, retrieved_results
                )
                total_tokens += content_tokens
            else:
                log.debug(f"No relevant content found for section '{section}', skip it.")
                section_content = ""

            english_sections[section] = section_content

        # Combine all sections into full text
        full_text = ""
        for section in self.sections:
            if english_sections[section] != "":
                full_text += f"## {section}\n\n{english_sections[section]}\n\n"

        # Compile and refine the final review
        log.color_print(f"<Step {step}> Compiling and refining the final review... </Step {step}>\n")
        step += 1
        compiled_text, compile_tokens = self._compile_final_review(english_topic, full_text)
        total_tokens += compile_tokens

        # Generate abstract and conclusion
        log.color_print(f"<Step {step}> Generating abstract and conclusion... </Step {step}>\n")
        step += 1
        abstract, conclusion, abstract_tokens = self._generate_abstract_and_conclusion(
            english_topic, compiled_text
        )
        total_tokens += abstract_tokens

        # Reorganize references and generate reference list
        log.color_print(f"<Step {step}> Reorganizing references... </Step {step}>\n")
        step += 1
        reorganized_text, references_text, ref_tokens = self._reorganize_references(compiled_text)
        total_tokens += ref_tokens

        # Parse sections from reorganized text
        import re

        compiled_sections = {}
        compiled_sections["Abstract"] = abstract
        section_pattern = r"## (.*?)\n\n(.*?)(?=\n\n## |$)"
        for match in re.finditer(section_pattern, reorganized_text, re.DOTALL):
            section_name = match.group(1).strip()
            section_content = match.group(2).strip()
            compiled_sections[section_name] = section_content

        # Add abstract, conclusion and references to the sections
        compiled_sections["Conclusion"] = conclusion
        compiled_sections["References"] = references_text

        # Translate each section to Chinese
        log.color_print(f"<Step {step}> Translating sections to Chinese... </Step {step}>\n")
        chinese_sections = {}
        for section, content in compiled_sections.items():
            if section != "References":  # Don't translate references
                log.debug(f"Translating section '{section}' to Chinese...")
                chinese_sections[section] = self._translate_to_chinese(content)
            else:
                chinese_sections[section] = content

        return compiled_sections, chinese_sections, total_tokens

    def query(self, query: str, **kwargs) -> Tuple[str, List[RetrievalResult], int]:
        """
        Process a research topic query and generate a comprehensive overview.

        Args:
            query: The research topic query

        Returns:
            Tuple of (response text, retrieval results (empty list), tokens used)
        """
        if kwargs.get("verbose"):
            self.verbose = True
        if kwargs.get("top_k_per_section"):
            self.top_k_per_section = kwargs.get("top_k_per_section")
        if kwargs.get("top_k_accepted_results"):
            self.top_k_accepted_results = kwargs.get("top_k_accepted_results")
        if kwargs.get("vector_db_collection"):
            self.vector_db_collection = kwargs.get("vector_db_collection")
            self.route_collection = False
        # This method overrides the RAGAgent query method
        english_sections, chinese_sections, total_tokens = asyncio.run(
            self.generate_overview(query, **kwargs)
        )

        # Format the response with both English and Chinese content
        self.english_response = f"# Overview: {query}\n\n"
        self.chinese_response = f"# 综述：{query}\n\n"

        for section in english_sections:
            log.debug(f"Merge English section '{section}', text length: {len(english_sections[section])}")
            self.english_response += f"## {section}\n\n"
            self.english_response += f"{english_sections[section]}\n\n"

        for section in chinese_sections:
            log.debug(f"Merge Chinese section '{section}', text length: {len(chinese_sections[section])}")
            self.chinese_response += self.translator.translate(f"## {section}", "zh") + "\n\n"
            self.chinese_response += f"{chinese_sections[section]}\n\n"

        return self.english_response, [], total_tokens

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
            chunk_str += f"""<chunk_{i + 1}>\n{chunk}\n</chunk_{i + 1}>\n"""
        return chunk_str
