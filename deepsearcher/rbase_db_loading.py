import os
from typing import List, Tuple
import tempfile
import pymysql
from tqdm import tqdm
import re

from deepsearcher import configuration
from deepsearcher.loader.splitter import split_docs_to_chunks
from deepsearcher.rbase.rbase_article import RbaseArticle, RbaseAuthor
from deepsearcher.tools.log import error, warning
from deepsearcher.db.mysql_connection import get_mysql_connection, close_mysql_connection

# Global variable to store active database connection
_active_connection = None

def get_mysql_connection(rbase_db_config: dict) -> pymysql.connections.Connection:
    """
    Get MySQL database connection, prioritizing reuse of existing active connection
    
    Args:
        rbase_db_config: Database configuration dictionary
        
    Returns:
        MySQL database connection object
    
    Raises:
        ValueError: If the database provider is not MySQL
        ConnectionError: If connection to database fails
    """
    global _active_connection
    
    # Check database provider
    if rbase_db_config.get('provider', '').lower() != 'mysql':
        raise ValueError("Currently only MySQL database is supported")
    
    # If there is an active connection, try to reuse it
    if _active_connection is not None:
        try:
            # Test if the connection is valid
            _active_connection.ping(reconnect=True)
            return _active_connection
        except Exception:
            # Connection is invalid, close and create a new one
            try:
                _active_connection.close()
            except Exception:
                pass
            _active_connection = None
    
    # Create a new connection
    try:
        conn = pymysql.connect(
            host=rbase_db_config.get('config', {}).get('host', 'localhost'), 
            port=int(rbase_db_config.get('config', {}).get('port', 3306)),
            user=rbase_db_config.get('config', {}).get('username', ''), 
            password=rbase_db_config.get('config', {}).get('password', ''),
            database=rbase_db_config.get('config', {}).get('database', ''), 
            charset='utf8mb4',
            cursorclass=pymysql.cursors.DictCursor
        )
        _active_connection = conn
        return conn
    except Exception as e:
        raise ConnectionError(f"Failed to connect to MySQL database: {e}")

def close_mysql_connection():
    """
    Close the current active MySQL connection
    """
    global _active_connection
    if _active_connection is not None:
        try:
            _active_connection.close()
        except Exception:
            pass
        finally:
            _active_connection = None

def init_vector_db(collection_name: str, collection_description: str, force_new_collection: bool = False):
    vector_db = configuration.vector_db

    collection_name = collection_name.replace(" ", "_").replace("-", "_")
    embedding_model = configuration.embedding_model
    vector_db.init_collection(
        dim=embedding_model.dimension,
        collection=collection_name,
        description=collection_description,
        force_new_collection=force_new_collection,
    )


def _process_authors(cursor, article: RbaseArticle, bypass_rbase_db: bool = False) -> Tuple[List[str], List[int], List[str], List[int]]:
    """
    Process author information, get author IDs and set authors for RbaseArticle object
    
    Args:
        cursor: Database cursor
        article: RbaseArticle object
        
    Returns:
        Tuple of author lists and author ID lists
    """
    # Ensure parameters are not None
    
    if not bypass_rbase_db:
        authors = article.authors or ""
        corresponding_authors = article.corresponding_authors or ""
        
        author_ids = []
        corresponding_author_ids = []
        # Process regular authors
        author_list = [author.strip() for author in authors.split(',') if author.strip()]
        
        # Process corresponding authors
        corresponding_author_list = [author.strip() for author in corresponding_authors.split(',') if author.strip()]
        
        # Merge author lists and remove duplicates
        all_authors_set = set(author_list + corresponding_author_list)
        all_authors = list(all_authors_set)
    
        # Query each author's ID and create RbaseAuthor objects
        for author_name in all_authors:
            # Determine if it's an English name or Chinese name
            is_english = all(ord(c) < 128 for c in author_name)
            
            # Create author object
            if is_english:
                author_obj = RbaseAuthor(name=author_name, ename=author_name)
                # Query by English name
                author_sql = """
                SELECT id FROM author WHERE ename = %s ORDER BY modified DESC
                """
                cursor.execute(author_sql, (author_name,))
            else:
                author_obj = RbaseAuthor(name=author_name, cname=author_name)
                # Query by Chinese name
                author_sql = """
                SELECT id FROM author WHERE cname = %s ORDER BY modified DESC
                """
                cursor.execute(author_sql, (author_name,))
            
            # Get all matching author IDs
            author_results = cursor.fetchall()
            if author_results:
                # Extract all author IDs
                ids = [result['id'] for result in author_results]
                # Set author IDs
                author_obj.set_author_ids(ids)
                # Add to ID list
                author_ids.extend(ids)
                
                # If it's a corresponding author, also add to corresponding author ID list
                is_corresponding = author_name in corresponding_author_list
                author_obj.is_corresponding = is_corresponding
                if is_corresponding:
                    corresponding_author_ids.extend(ids)
            
            # Add to article
            article.set_author(author_obj)
    
        return all_authors, author_ids, corresponding_author_list, corresponding_author_ids


def _process_keywords(article: RbaseArticle, bypass_rbase_db: bool = False) -> List[str]:
    """
    Process article keywords information, either retrieving from database or using keywords directly from article object.
    
    Args:
        article: RbaseArticle object containing article information
        bypass_rbase_db: Whether to bypass Rbase database and use keywords directly from article object, defaults to False
        
    Returns:
        List[str]: Deduplicated list of keywords
    """

    if not bypass_rbase_db:
        # Ensure parameters are not None
        source_keywords = article.source_keywords or ""
        mesh_keywords = article.mesh_keywords or ""
        
        # Split keywords by semicolon and remove whitespace
        source_keywords_list = [kw.strip() for kw in source_keywords.split(';') if kw.strip()]
        mesh_keywords_list = [kw.strip() for kw in mesh_keywords.split(';') if kw.strip()]
        
        # Merge keyword lists and remove duplicates
        keywords_set = set(source_keywords_list + mesh_keywords_list)
        keywords_list = list(keywords_set)
    else:
        keywords_list = article.keywords
    
    return keywords_list


def _download_file_content(url: str) -> str:
    """
    Download file content from URL
    
    Args:
        url: File URL
        
    Returns:
        File content
    
    Raises:
        Exception: If download fails
    """
    import requests
    import urllib3
    
    # Disable SSL verification to resolve SSL connection issues
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    response = requests.get(url, verify=False)
    response.raise_for_status()  # Ensure request was successful
    
    return response.text


def insert_to_vector_db(rbase_config: dict, 
                      articles: list[RbaseArticle],
                      collection_name: str = None,
                      collection_description: str = None,
                      force_new_collection: bool = False,
                      chunk_size: int = 1500,
                      chunk_overlap: int = 100,
                      batch_size: int = 256,
                      bypass_rbase_db: bool = False,
                      save_downloaded_file: bool = False):
    """
    Load article data into vector database
    
    Args:
        rbase_config: Configuration dictionary containing OSS and database configurations
        articles: List of RbaseArticle objects containing article data to process
        collection_name: Vector database collection name
        collection_description: Vector database collection description
        force_new_collection: Whether to force create a new collection
        chunk_size: Text chunk size
        chunk_overlap: Text chunk overlap size
        batch_size: Batch processing size
    """
    # Check OSS configuration
    rbase_oss_config = rbase_config.get('oss', {})
    
    # Ensure OSS host address is not empty
    if not rbase_oss_config.get('host'):
        raise ValueError("OSS host address cannot be empty")
    
    # Check database configuration
    rbase_db_config = rbase_config.get('database', {})
    
    # Initialize vector database
    vector_db = configuration.vector_db
    embedding_model = configuration.embedding_model
    file_loader = configuration.file_loader
    
    if collection_name is None:
        collection_name = vector_db.default_collection
    collection_name = collection_name.replace(" ", "_").replace("-", "_")
    
    # Initialize vector database collection
    init_vector_db(collection_name, collection_description, force_new_collection)
    
    # Get MySQL connection
    conn = get_mysql_connection(rbase_db_config)
    
    all_docs = []
    
    try:
        with conn.cursor() as cursor:
            # Process each article
            for article in tqdm(articles, desc="Processing article files"):
                txt_file_path = article.txt_file
                
                # Additional check if txt_file_path is empty or not ending with .md
                if not txt_file_path or not txt_file_path.endswith('.md'):
                    warning(f"Skipping invalid file path: {txt_file_path}")
                    continue
                
                # Process author information
                _process_authors(cursor, article, bypass_rbase_db)
                
                # Process keywords
                keywords_list = _process_keywords(article, bypass_rbase_db) 
                
                # Create temporary file to save markdown content
                with tempfile.NamedTemporaryFile(suffix='.md', delete=False) as temp_file:
                    temp_path = temp_file.name
                    
                    # Download file content from OSS server to temporary file
                    # Ensure both host and txt_file_path are not empty
                    host = rbase_oss_config.get('host', '')
                    if not host or not txt_file_path:
                        warning(f"Skipping download: OSS host address or file path is empty - host: {host}, path: {txt_file_path}")
                        continue
                        
                    full_url = host + txt_file_path
                    
                    try:
                        content = _download_file_content(full_url)
                        
                        # Remove content after "# REFERENCES" (case-insensitive)
                        references_pattern = re.compile(r'#\s*references.*$', re.IGNORECASE | re.DOTALL)
                        content = re.sub(references_pattern, '', content)
                        
                        temp_file.write(content.encode('utf-8'))
                        
                        # 在database/markdown/目录下存储备份文件
                        if save_downloaded_file:
                            current_dir = os.path.dirname(os.path.abspath(__file__))
                            backup_dir = os.path.join(current_dir, '..', 'database', 'markdown')
                            os.makedirs(backup_dir, exist_ok=True)
                            backup_filename = os.path.basename(txt_file_path)
                            backup_path = os.path.join(backup_dir, backup_filename)
                            with open(backup_path, 'w', encoding='utf-8') as backup_file:
                                backup_file.write(content)
                    except Exception as e:
                        error(f"Failed to download file: {e}, URL: {full_url}")
                        continue
                
                # Load temporary file
                docs = file_loader.load_file(temp_path)
                
                # Add metadata to each document
                for doc in docs:
                    # Get author information
                    if not bypass_rbase_db:
                        author_names = [author.name for author in article.author_objects]
                        author_ids = []
                        corresponding_author_names = []
                        corresponding_author_ids = []
                        
                        for author in article.author_objects:
                            if hasattr(author, 'author_ids'):
                                author_ids.extend(author.author_ids)
                                if author.is_corresponding:
                                    corresponding_author_names.append(author.name)
                                    corresponding_author_ids.extend(author.author_ids)
                    else:
                        author_names = list(set(article.authors + article.corresponding_authors))
                        author_ids = list(set(article.author_ids + article.corresponding_author_ids))
                        corresponding_author_names = article.corresponding_authors
                        corresponding_author_ids = article.corresponding_author_ids
                    
                    author_names = author_names[:200] if len(author_names) > 200 else author_names
                    corresponding_author_names = corresponding_author_names[:200] if len(corresponding_author_names) > 200 else corresponding_author_names
                    author_ids = author_ids[:200] if len(author_ids) > 200 else author_ids
                    corresponding_author_ids = corresponding_author_ids[:200] if len(corresponding_author_ids) > 200 else corresponding_author_ids

                    doc.metadata.update({
                        'title': article.title,
                        'authors': author_names,
                        'author_ids': author_ids,
                        'corresponding_authors': corresponding_author_names,
                        'corresponding_author_ids': corresponding_author_ids,
                        'keywords': keywords_list,
                        'pubdate': article.pubdate,
                        'article_id': article.article_id,
                        'impact_factor': article.impact_factor,
                        'reference': f"Article ID: {article.article_id}"
                    })
                
                all_docs.extend(docs)
                
                # Delete temporary file
                os.unlink(temp_path)
            
            # Split documents into chunks
            chunks = split_docs_to_chunks(
                all_docs,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            
            # Embed vectors
            chunks = embedding_model.embed_chunks(chunks, batch_size=batch_size)
            
            # Insert into vector database
            return vector_db.insert_data(collection=collection_name, chunks=chunks)
            
    except Exception as e:
        # Close connection when exception occurs
        close_mysql_connection()
        raise Exception(f"Failed to process article data: {e}")


def load_from_rbase_db(rbase_config: dict, offset: int = 0, limit: int = 10) -> list[RbaseArticle]:
    """
    Load article data from Rbase database to vector database
    
    Args:
        rbase_config: Database configuration dictionary
        offset: Query start position
        limit: Query limit count
        
    Returns:
        List of RbaseArticle objects
    """
    # Get MySQL connection
    rbase_db_config = rbase_config.get('database', {})
    conn = get_mysql_connection(rbase_db_config)
    
    pdf_files = []
    try:
        with conn.cursor() as cursor:
            # Query data from article_pdf_file table that meets the conditions, and ensure raw_article_id exists in article table
            # Also ensure txt_file is not empty and ends with .md
            # sql = """
            # SELECT apf.id, apf.raw_article_id, apf.txt_file, 
            #        a.title, a.authors, a.corresponding_authors,
            #        a.impact_factor, a.source_keywords, a.mesh_keywords, a.pubdate
            # FROM article_pdf_file apf
            # INNER JOIN article a ON apf.raw_article_id = a.id
            # WHERE apf.user_id = 0 AND apf.status = 1 
            # AND apf.txt_file IS NOT NULL 
            # AND apf.txt_file LIKE '%%.md'
            # ORDER BY apf.modified DESC LIMIT %s OFFSET %s
            # """
            sql = """
            SELECT a.id, a.raw_article_id, a.txt_file, 
                   a.title, a.authors, a.corresponding_authors,
                   a.impact_factor, a.source_keywords, a.mesh_keywords, a.pubdate
            FROM article a
            WHERE a.txt_file IS NOT NULL AND a.txt_file LIKE '%%.md'
            ORDER BY a.modified DESC LIMIT %s OFFSET %s
            """
            # Use parameterized query to avoid string formatting issues
            cursor.execute(sql, (limit, offset))
            pdf_files = cursor.fetchall()
    except Exception as e:
        # Close connection when exception occurs
        close_mysql_connection()
        raise Exception(f"Failed to process database data: {e}")
    
    return [RbaseArticle(pdf) for pdf in pdf_files]
            