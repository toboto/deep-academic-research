#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script for creating user dictionaries for jieba library.
Reads data from the concept table in rbase's MySQL database, extracts Chinese names (cname) and English names (name),
and outputs them to rbase_dict_cn.txt and rbase_dict_en.txt files respectively.
"""

import os
import sys
import pymysql
import logging
from typing import List, Dict, Tuple, Optional
from deepsearcher.tools.log import color_print, error, warning, debug, info

# Suppress unnecessary log output
logging.getLogger("httpx").setLevel(logging.WARNING)

# Add project root directory to system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deepsearcher import configuration

class ConceptDictCreator:
    """
    Reads data from the concept table in MySQL database 
    and generates user dictionary files required by jieba library.
    """
    
    def __init__(self, config_path: str = "../config.rbase.yaml"):
        """
        Initialize the ConceptDictCreator class.
        
        Args:
            config_path: Configuration file path
        """
        self.config = configuration.Configuration(config_path)
        configuration.init_config(self.config)
        self.db_config = self.config.rbase_settings["database"]["config"]
        # Get dictionary paths from configuration, use default values if not specified
        self.cn_dict_path = self.config.rbase_settings.get("dict_path", {}).get("cn", "rbase_dict_cn.txt")
        self.en_dict_path = self.config.rbase_settings.get("dict_path", {}).get("en", "rbase_dict_en.txt")
        
    def connect_to_db(self) -> pymysql.connections.Connection:
        """
        Connect to MySQL database.
        
        Returns:
            Database connection object
        """
        try:
            connection = pymysql.connect(
                host=self.db_config["host"],
                user=self.db_config["username"],
                password=self.db_config["password"],
                database=self.db_config["database"],
                port=self.db_config["port"],
                charset='utf8mb4',
                cursorclass=pymysql.cursors.DictCursor
            )
            info("Successfully connected to MySQL database")
            return connection
        except Exception as e:
            error(f"Failed to connect to database: {e}")
            raise
    
    def get_concepts(self, connection: pymysql.connections.Connection) -> List[Dict]:
        """
        Get data from the concept table in the database.
        
        Args:
            connection: Database connection object
            
        Returns:
            List containing concept table data
        """
        try:
            with connection.cursor() as cursor:
                # Query cname and name fields from the concept table
                sql = "SELECT id, cname, name FROM concept WHERE cname IS NOT NULL AND cname != '' AND intro IS NOT NULL AND intro != ''"
                cursor.execute(sql)
                result = cursor.fetchall()
                color_print(f"Retrieved {len(result)} records from concept table")
                return result
        except Exception as e:
            error(f"Failed to query database: {e}")
            raise
    
    def get_word_pos(self, word: str) -> str:
        """
        使用LLM获取词的词性。
        
        Args:
            word: 需要判断词性的词
            
        Returns:
            Part-of-speech tag
        """
        prompt = f"""
        请判断以下中文词语的词性，只需要回答词性的缩写代码，不要有任何解释。
        对于提供了两个词的情况，只分析其中更具特异性的词，并返回一个结果。
        对于其他任何场景，如果判断可能具有多种词性，都只返回最常见的一个词性结果。
        
        请参考以下jieba分词的词性标注体系：
        
        名词(n): 
        - n 名词
        - nr 人名
        - ns 地名
        - nt 机构团体名
        - nz 其它专名
        
        动词(v):
        - v 动词
        - vd 副动词
        - vn 名动词
        
        形容词(a):
        - a 形容词
        - ad 副形词
        - an 名形词
        
        其他常见词性:
        - r 代词
        - m 数词
        - q 量词
        - d 副词
        - p 介词
        - c 连词
        - j 简称略语
        - nw 新词

        专有名词类型:
        - PER 人名
        - LOC 地名
        - ORG 机构名
        - TIME 时间
        
        词语: {word}
        词性代码:
        """
        
        try:
            response = configuration.llm.chat([{"role": "user", "content": prompt}])
            pos = response.content.strip()
            # If the returned part of speech is not in jieba's part-of-speech tagging system, 
            # default to noun(n)
            # Define list of valid jieba part-of-speech tags
            valid_pos_tags = ['n', 'nr', 'ns', 'nt', 'nz', 
                              'v', 'vd', 'vn', 'a', 'ad', 'an', 
                              'r', 'm', 'q', 'd', 'p', 'c', 'j', 'nw', 'eng', 
                              'PER', 'LOC', 'ORG', 'TIME']
            
            if not pos or pos not in valid_pos_tags:
                warning(f"LLM returned invalid part of speech '{pos}' for '{word}', using default part of speech 'n'")
                pos = "n"
            return pos
        except Exception as e:
            error(f"Failed to call LLM: {e}")
            return "n"  # Default to noun
    
    def create_dict_files(self, concepts: List[Dict]) -> Tuple[int, int]:
        """
        Create user dictionary files.
        
        Args:
            concepts: List containing concept table data
            
        Returns:
            Tuple containing the number of entries in Chinese and English dictionaries
        """
        cn_count = 0
        en_count = 0
        
        try:
            with open(self.cn_dict_path, 'w', encoding='utf-8') as cn_file, \
                 open(self.en_dict_path, 'w', encoding='utf-8') as en_file:
                
                # Use tqdm to create progress bar
                from tqdm import tqdm
                
                total = len(concepts)
                for concept in tqdm(concepts, desc="Creating dictionary", total=total, unit="entries"):
                    # Process Chinese name
                    if concept['cname'] and len(concept['cname'].strip()) > 0:
                        cname = concept['cname'].strip()
                        pos = self.get_word_pos(cname)
                        cn_file.write(f"{cname} 1000 {pos}\n")
                        cn_count += 1
                    
                    # Process English name
                    if concept['name'] and len(concept['name'].strip()) > 0:
                        name = concept['name'].strip()
                        # English words default to foreign word part of speech (eng)
                        en_file.write(f"{name} 1000 eng\n")
                        en_count += 1
            info(f"Successfully created Chinese dictionary file with {cn_count} entries")
            info(f"Successfully created English dictionary file with {en_count} entries")
            return cn_count, en_count
        except Exception as e:
            error(f"Failed to create dictionary files: {e}")
            raise
    
    def run(self) -> None:
        """
        Run the dictionary creation process.
        """
        try:
            connection = self.connect_to_db()
            concepts = self.get_concepts(connection)
            self.create_dict_files(concepts)
            connection.close()
            info("Dictionary creation completed")
        except Exception as e:
            error(f"Dictionary creation failed: {e}")
            sys.exit(1)

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    yaml_file = os.path.join(current_dir, "..", "config.rbase.yaml")
    creator = ConceptDictCreator(yaml_file)
    creator.run()
