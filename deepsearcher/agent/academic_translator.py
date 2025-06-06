"""
Academic Translation Agent Module.

This module provides the AcademicTranslator class for translating academic texts
into specified languages.  Since academic texts often contain specialized terminology,
this class uses jieba word segmentation and the concept table in MySQL database
to improve translation quality.
"""

import os
import re
from typing import Dict, List, Optional, Tuple

import jieba
import jieba.posseg as pseg

from deepsearcher.agent.base import BaseAgent, describe_class
from deepsearcher.llm.base import BaseLLM
from deepsearcher.tools.log import debug, error, warning
from deepsearcher.db.mysql_connection import get_mysql_connection

@describe_class(
    "This Agent is used to translate academic texts into specified languages, with special focus on accurate translation of professional terminology."
)
class AcademicTranslator(BaseAgent):
    """
    Academic Translation Agent for translating academic texts into specified languages.

    This class uses jieba word segmentation and the concept table in MySQL database to
    improve the translation quality of professional terminology.
    Currently only supports translation between Chinese and English.
    """

    def __init__(self, llm: BaseLLM, rbase_settings: dict, **kwargs):
        """
        Initialize the AcademicTranslator class.

        Load configuration, initialize LLM, connect to database, and load jieba user dictionary.
        """
        super().__init__(**kwargs)

        # Get LLM
        self.llm = llm

        # Get database configuration and connect to database
        self.db_config = rbase_settings["database"]
        self.dict_config = rbase_settings["dict_path"]
        self.db_connection = None

        # Load jieba user dictionary
        self._load_jieba_dict()

        # Cache previously queried term translations to avoid duplicate database queries
        self.term_cache = {}

    def _load_jieba_dict(self) -> None:
        """
        Load jieba user dictionary.

        Find and load rbase_dict_cn.txt for jieba segmentation.
        For rbase_dict_en.txt, store the words in a sorted list for English to Chinese translation.
        """
        # Possible dictionary paths
        possible_cn_paths = [
            self.dict_config.get("cn", "rbase_dict_cn.txt"),
            os.path.join(os.getcwd(), "rbase_dict_cn.txt"),
        ]
        possible_en_paths = [
            self.dict_config.get("en", "rbase_dict_en.txt"),
            os.path.join(os.getcwd(), "rbase_dict_en.txt"),
        ]

        # Initialize English terms dictionary
        self.en_terms = {}
        # Initialize sorted English terms list (multi-word terms first, sorted by length)
        self.sorted_en_terms = []

        # Find Chinese dictionary
        cn_dict_path = None
        for path in possible_cn_paths:
            if os.path.exists(path):
                cn_dict_path = path
                break

        # Find English dictionary
        en_dict_path = None
        for path in possible_en_paths:
            if os.path.exists(path):
                en_dict_path = path
                break

        # Load Chinese dictionary into jieba
        if cn_dict_path:
            jieba.load_userdict(cn_dict_path)
            debug(f"Loaded Chinese user dictionary: {cn_dict_path}")
        else:
            warning("Chinese user dictionary file rbase_dict_cn.txt not found")

        # Load English dictionary into a sorted list
        if en_dict_path:
            try:
                with open(en_dict_path, "r", encoding="utf-8") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 3:
                            # Format: [phrase] [freq] [part-of-speech]
                            # The phrase might contain spaces, so we need to handle it carefully
                            pos = parts[-1]
                            phrase = " ".join(parts[:-2])

                            # Store the English term with its part of speech
                            self.en_terms[phrase] = pos

                # Pre-sort English terms: multi-word terms first (sorted by length), then single words
                multi_word_terms = [term for term in self.en_terms.keys() if " " in term]
                single_word_terms = [term for term in self.en_terms.keys() if " " not in term]

                # Sort multi-word terms by length (longest first)
                multi_word_terms.sort(key=len, reverse=True)
                # Sort single word terms alphabetically
                single_word_terms.sort()

                # Combine the sorted lists
                self.sorted_en_terms = multi_word_terms + single_word_terms

                debug(f"Loaded {len(self.en_terms)} English terms from: {en_dict_path}")
            except Exception as e:
                error(f"Failed to load English dictionary: {e}")
        else:
            warning("English user dictionary file rbase_dict_en.txt not found")

    def _detect_language(self, text: str, target_lang: str) -> str:
        """
        Detect the main language of the text.

        Args:
            text: The text to detect language
            target_lang: Target language, 'zh' or 'en'

        Returns:
            Language code: 'zh' for Chinese, 'en' for English, 'mixed' for mixed language
        """
        # Calculate the number of Chinese and English characters
        chinese_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
        english_chars = len(re.findall(r"[a-zA-Z]", text))

        # Determine the main language based on character count
        if chinese_chars > 0 and english_chars == 0:
            return "zh"
        elif english_chars > 0 and chinese_chars == 0:
            return "en"
        elif chinese_chars > 0 and english_chars > 0:
            return "en" if target_lang == "zh" else "zh"
        else:
            return "unknown"

    def _segment_chinese_text(self, text: str) -> List[Tuple[str, str]]:
        """
        Use jieba to segment Chinese text.

        Args:
            text: The Chinese text to segment

        Returns:
            List of tuples, each containing a word and its POS tag
        """
        # Use jieba for POS tagging segmentation
        words = pseg.cut(text)
        return [(word, flag) for word, flag in words]

    def _query_term_translation(
        self, term: str, source_lang: str, target_lang: str
    ) -> Optional[str]:
        """
        Query the translation of a term from the database.

        Args:
            term: The term to translate
            source_lang: Source language, 'zh' or 'en'
            target_lang: Target language, 'zh' or 'en'

        Returns:
            The translation of the term, or None if not found
        """
        # Check cache
        cache_key = f"{term}_{source_lang}_{target_lang}"
        if cache_key in self.term_cache:
            return self.term_cache[cache_key]

        try:
            self.db_connection = get_mysql_connection(self.db_config)
            with self.db_connection.cursor() as cursor:
                if source_lang == "zh" and target_lang == "en":
                    # Chinese -> English
                    sql = "SELECT name FROM concept WHERE cname = %s AND name IS NOT NULL AND name != '' AND intro is NOT NULL AND intro != '' LIMIT 1"
                    cursor.execute(sql, (term,))
                elif source_lang == "en" and target_lang == "zh":
                    # English -> Chinese
                    sql = "SELECT cname FROM concept WHERE name = %s AND cname IS NOT NULL AND cname != '' AND intro is NOT NULL AND intro != '' LIMIT 1"
                    cursor.execute(sql, (term,))
                else:
                    return None

                result = cursor.fetchone()
                if result:
                    translation = result["name"] if source_lang == "zh" else result["cname"]
                    # Cache the translation
                    self.term_cache[cache_key] = translation
                    return translation
                else:
                    self.term_cache[cache_key] = None
                    return None
        except Exception as e:
            error(f"Failed to query term translation: {e}")
            return None

    def _build_translation_glossary(
        self, text: str, source_lang: str, target_lang: str
    ) -> Dict[str, str]:
        """
        Build a translation glossary.

        Args:
            text: The text to translate
            source_lang: Source language, 'zh' or 'en'
            target_lang: Target language, 'zh' or 'en'

        Returns:
            A translation glossary, with source language terms as keys
            and target language translations as values
        """
        glossary = {}

        if source_lang == "zh":
            # Segment Chinese text
            words = self._segment_chinese_text(text)

            # Filter possible terms (nouns, proper nouns, etc.)
            term_flags = {"n", "nr", "ns", "nt", "nz", "nw", "l", "PER", "LOC", "ORG"}
            for word, flag in words:
                if flag in term_flags and len(word) >= 2:  # Only consider words with length >= 2
                    translation = self._query_term_translation(word, "zh", "en")
                    if translation:
                        glossary[word] = translation

        elif source_lang == "en":
            # Use the pre-sorted English terms list
            for term in self.sorted_en_terms:
                # For multi-word terms, check if they exist in the text
                if " " in term and term.lower() in text.lower():
                    translation = self._query_term_translation(term, "en", "zh")
                    if translation:
                        glossary[term] = translation
                # For single words, use regex to match whole words only
                elif " " not in term:
                    # Create a regex pattern to match the word as a whole word
                    pattern = r"\b" + re.escape(term) + r"\b"
                    if re.search(pattern, text, re.IGNORECASE):
                        # Skip common English stop words and short words
                        if len(term) <= 2 or term.lower() in {
                            "the",
                            "and",
                            "of",
                            "to",
                            "in",
                            "is",
                            "it",
                            "for",
                            "as",
                            "on",
                            "at",
                            "by",
                            "with",
                        }:
                            continue

                        translation = self._query_term_translation(term, "en", "zh")
                        if translation:
                            glossary[term] = translation

        return glossary

    def translate(self, text: str, target_lang: str, user_dict: List[dict] = None) -> str:
        """
        Translate text to the target language.

        Args:
            text: The text to translate
            target_lang: Target language, 'zh' or 'en'
            user_dict: List of dictionaries, each containing 'source' and 'translation' keys

        Returns:
            The translated text

        Raises:
            ValueError: If the target language is not 'zh' or 'en'
        """
        # Check target language
        if target_lang not in ["zh", "en"]:
            raise ValueError("target_lang must be 'zh' or 'en'")

        # Detect source text language
        source_lang = self._detect_language(text, target_lang)

        # If source language and target language are the same, no translation is needed
        if source_lang == target_lang:
            return text

        # If the source language is mixed, determine the source language based on the target language
        if source_lang == "mixed":
            source_lang = "en" if target_lang == "zh" else "zh"

        # Build a translation glossary
        glossary = self._build_translation_glossary(text, source_lang, target_lang)

        # Prepare translation prompt
        prompt = f"""
        Please translate the following academic text from {self._get_language_name(source_lang)} to {self._get_language_name(target_lang)}.
        This is an academic text, please maintain the accuracy and academic style of professional terminology.
        
        Output translation directly, do not include any other text or reasoning process.

        Here are some translation references for professional terms, please use these translations as a reference:
        """

        if user_dict is not None:
            for term in user_dict:
                prompt += f"\n- {term['source']}: {term['translation']}"

        # Add term glossary
        if glossary:
            for term, translation in glossary.items():
                prompt += f"\n- {term}: {translation}"
        else:
            prompt += "\n(No translation references found for professional terms)"

        prompt += f"""
        
        Original text ({self._get_language_name(source_lang)}):
        {text}
        
        Translation ({self._get_language_name(target_lang)}):
        """

        # Invoke LLM for translation
        response = self.llm.chat([{"role": "user", "content": prompt}])

        return response.content.strip()

    def _get_language_name(self, lang_code: str) -> str:
        """
        Get the name of the language corresponding to the language code.

        Args:
            lang_code: Language code

        Returns:
            The name of the language
        """
        if lang_code == "zh":
            return "Chinese"
        elif lang_code == "en":
            return "English"
        else:
            return "Unknown language"

    def invoke(self, query: str, **kwargs) -> str:
        """
        Invoke the translation function.

        Args:
            query: The text to translate
            target_lang: Target language, 'zh' or 'en', default is 'zh'

        Returns:
            The translated text
        """
        target_lang = kwargs.get("target_lang", "zh")
        return self.translate(query, target_lang)
