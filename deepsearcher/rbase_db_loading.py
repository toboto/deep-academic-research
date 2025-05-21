import re
import tempfile
import os
from typing import List, Tuple
from tqdm import tqdm

from deepsearcher import configuration
from deepsearcher.rbase.rbase_article import RbaseArticle, RbaseAuthor
from deepsearcher.db.mysql_connection import get_mysql_connection, close_mysql_connection
from deepsearcher.db.async_mysql_connection import get_mysql_pool
from deepsearcher.loader.splitter import split_docs_to_chunks
from deepsearcher.tools.log import warning, error, debug

def init_vector_db(
    collection_name: str, collection_description: str, force_new_collection: bool = False
):
    vector_db = configuration.vector_db

    collection_name = collection_name.replace(" ", "_").replace("-", "_")
    embedding_model = configuration.embedding_model
    vector_db.init_collection(
        dim=embedding_model.dimension,
        collection=collection_name,
        description=collection_description,
        force_new_collection=force_new_collection,
    )


def _process_authors(
    cursor, article: RbaseArticle, bypass_rbase_db: bool = False
) -> Tuple[List[str], List[int], List[str], List[int]]:
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
        author_list = [author.strip() for author in authors.split(",") if author.strip()]

        # Process corresponding authors
        corresponding_author_list = [
            author.strip() for author in corresponding_authors.split(",") if author.strip()
        ]

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
                ids = [result["id"] for result in author_results]
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
                    author_ids = author_ids[:500] if len(author_ids) > 500 else author_ids
                    corresponding_author_names = corresponding_author_names[:40] if len(corresponding_author_names) > 40 else corresponding_author_names
                    corresponding_author_ids = corresponding_author_ids[:100] if len(corresponding_author_ids) > 100 else corresponding_author_ids

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
            sql = """
            SELECT a.id, a.raw_article_id, a.txt_file, 
                   a.title, a.authors, a.corresponding_authors,
                   a.impact_factor, a.source_keywords, a.mesh_keywords, a.pubdate,
                   a.summary as abstract, a.journal_name
            FROM article a
            WHERE a.txt_file IS NOT NULL AND a.txt_file LIKE '%%.md' AND a.base_id=1
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


async def load_articles_by_channel(channel_id: int, term_tree_node_ids: list[int], offset: int = 0, limit: int = 10) -> list[RbaseArticle]:
    """
    Load article data from Rbase database by channel
    
    Args:
        channel_id: Channel ID
        term_tree_node_ids: List of term tree node IDs
        offset: Query start position
        limit: Query limit count
        
    Returns:
        List of RbaseArticle objects
    """
    concept_id_lists = []
    if term_tree_node_ids:
        for term_tree_node_id in term_tree_node_ids:
            concepts = await get_sub_node_concept_ids(channel_id, term_tree_node_id)
            if concepts:
                concept_id_lists.append(concepts)
    
    term_id_groups = []
    if len(concept_id_lists) > 0:
        for concept_ids in concept_id_lists:
            term_id_groups.append(await get_concept_term_ids(concept_ids))
    
    return await load_articles_by_term_ids(term_id_groups, channel_id, offset, limit)


async def get_sub_node_concept_ids(base_id: int, term_tree_node_id: int) -> list[int]:
    """
    Get sub node concept id list from term_tree_node table
    
    Args:
        base_id: Base ID
        term_tree_node_id: Term tree node ID

    Returns:
        List of concept ids
    """
    node_ids = []
    concept_ids = []
    if base_id == 0 or term_tree_node_id == 0:
        return concept_ids

    node_ids.append(term_tree_node_id)
    try:
        pool = await get_mysql_pool(configuration.config.rbase_settings.get("database"))
        async with pool.acquire() as conn:
            async with conn.cursor() as cursor:
                for i in range(0, 1000000):
                    if i >= len(node_ids):
                        break
                    term_tree_node_id = node_ids[i]
                    # Get concept ids
                    sql = """
                    SELECT tn.node_concept_id FROM term_tree_node tn LEFT JOIN term_tree tr ON tn.tree_id = tr.id 
                    WHERE tr.related_base_id=%s AND tn.id=%s
                    """
                    await cursor.execute(sql, (base_id, term_tree_node_id))
                    result = await cursor.fetchall()
                    if result:
                        for row in result:
                            concept_ids.append(row["node_concept_id"])

                    # Get sub node ids
                    sql = """
                    SELECT tn.id FROM term_tree_node tn LEFT JOIN term_tree tr ON tn.tree_id = tr.id 
                    WHERE tr.related_base_id=%s AND tn.parent_node_id=%s
                    """
                    await cursor.execute(sql, (base_id, term_tree_node_id))
                    result = await cursor.fetchall()
                    if result:
                        for row in result:
                            node_ids.append(row["id"])
    except Exception as e:
        raise Exception(f"Failed to get sub node ids: {e}")
    
    return concept_ids


async def get_concept_term_ids(term_tree_node_concept_ids: list[int]) -> list[int]:
    """
    Get concept term id list from concept table
    
    Args:
        term_tree_node_concept_ids: List of term tree node concept ids
        
    Returns:
        List of concept term ids
    """
    if len(term_tree_node_concept_ids) == 0:
        return []
    try:
        rts = []
        pool = await get_mysql_pool(configuration.config.rbase_settings.get("database"))
        async with pool.acquire() as conn:
            async with conn.cursor() as cursor:
                # 使用参数化查询，为每个 ID 创建占位符
                placeholders = ", ".join(["%s"] * len(term_tree_node_concept_ids))
                sql = f"""
                SELECT concept_term_id, concept_term_id2 FROM concept WHERE id IN ({placeholders})
                """
                await cursor.execute(sql, term_tree_node_concept_ids)
                result = await cursor.fetchall()
                for row in result:
                    if row["concept_term_id2"]:
                        rts.append(row["concept_term_id2"])
                    elif row["concept_term_id"]:
                        rts.append(row["concept_term_id"])

                return rts
    except Exception as e:
        raise Exception(f"Failed to get concept term ids: {e}")


async def load_articles_by_term_ids(term_id_groups: list[list[int]], base_id: int, offset: int = 0, limit: int = 10) -> list[RbaseArticle]:
    """
    Load article data from Rbase database
    
    Args:
        term_id_groups: List of term IDs
        base_id: Base ID
        offset: Query start position
        limit: Query limit count
        
    Returns:
        List of RbaseArticle objects
    """
    # Get MySQL connection
    rbase_db_config = configuration.config.rbase_settings.get("database")
    
    try:
        pool = await get_mysql_pool(rbase_db_config)
        async with pool.acquire() as conn:
            async with conn.cursor() as cursor:
                params = [base_id]
                sql = """
                SELECT a.id, a.raw_article_id, a.txt_file, 
                    a.title, a.authors, a.corresponding_authors,
                    a.impact_factor, a.source_keywords, a.mesh_keywords, a.pubdate,
                    a.summary as abstract, a.journal_name
                FROM article a
                WHERE a.status=1 AND a.base_id=%s 
                """
                if len(term_id_groups) > 0:
                    for term_ids in term_id_groups:
                        placeholders = ", ".join(["%s"] * len(term_ids))
                        sql += f"\n AND EXISTS (SELECT 1 FROM article_label al WHERE al.article_id=a.id AND al.term_id IN ({placeholders}))"
                        params.extend(term_ids)
                
                sql += "\n ORDER BY a.pubdate DESC LIMIT %s OFFSET %s"
                params.append(limit)
                params.append(offset)
                # Use parameterized query to avoid string formatting issues
                await cursor.execute(sql, tuple(params))
                pdf_files = await cursor.fetchall()
                debug(f"已读取{len(pdf_files)}条文章数据")
    except Exception as e:
        raise Exception(f"Failed to process database data: {e}")
    
    return [RbaseArticle(pdf) for pdf in pdf_files]

async def load_articles_by_article_ids(article_ids: list[int], offset: int = 0, limit: int = 10) -> list[RbaseArticle]:
    """
    Load article data from Rbase database 

    Args:
        article_ids: List of article IDs
        offset: Query start position
        limit: Query limit count
        
    Returns:
        List of RbaseArticle objects
    """
    # Get MySQL connection
    rbase_db_config = configuration.config.rbase_settings.get("database")
    
    try:
        pool = await get_mysql_pool(rbase_db_config)
        async with pool.acquire() as conn:
            async with conn.cursor() as cursor:
                placeholders = ", ".join(["%s"] * len(article_ids))
                sql = f"""
                SELECT a.id, a.raw_article_id, a.txt_file, 
                    a.title, a.authors, a.corresponding_authors,
                    a.impact_factor, a.source_keywords, a.mesh_keywords, a.pubdate,
                    a.summary as abstract, a.journal_name
                FROM article a
                WHERE a.id IN ({placeholders})
                """
                await cursor.execute(sql, tuple(article_ids))
                pdf_files = await cursor.fetchall()
    except Exception as e:
        raise Exception(f"Failed to process database data: {e}")
    
    return [RbaseArticle(pdf) for pdf in pdf_files]