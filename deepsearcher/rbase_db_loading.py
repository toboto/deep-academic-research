import os
from typing import List, Union, Tuple, Dict, Any, Optional
import tempfile
import pymysql
from langchain_core.documents import Document
from tqdm import tqdm

# from deepsearcher.configuration import embedding_model, vector_db, file_loader
from deepsearcher import configuration
from deepsearcher.loader.splitter import split_docs_to_chunks
from deepsearcher.rbase.rbase_article import RbaseArticle, RbaseAuthor

# 全局变量，用于存储活跃的数据库连接
_active_connection = None

def get_mysql_connection(rbase_db_config: dict) -> pymysql.connections.Connection:
    """
    获取MySQL数据库连接，优先复用已有的活跃连接
    
    参数:
        rbase_db_config: 数据库配置字典
        
    返回:
        MySQL数据库连接对象
    
    异常:
        ValueError: 如果数据库提供商不是MySQL
        ConnectionError: 如果连接数据库失败
    """
    global _active_connection
    
    # 检查数据库提供商
    if rbase_db_config.get('provider', '').lower() != 'mysql':
        raise ValueError("当前仅支持MySQL数据库")
    
    # 如果已有活跃连接，尝试复用
    if _active_connection is not None:
        try:
            # 测试连接是否有效
            _active_connection.ping(reconnect=True)
            return _active_connection
        except Exception:
            # 连接已失效，关闭并创建新连接
            try:
                _active_connection.close()
            except Exception:
                pass
            _active_connection = None
    
    # 创建新连接
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
        raise ConnectionError(f"连接MySQL数据库失败: {e}")

def close_mysql_connection():
    """
    关闭当前活跃的MySQL连接
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


def _process_authors(cursor, article: RbaseArticle) -> Tuple[List[str], List[int], List[str], List[int]]:
    """
    处理作者信息，获取作者ID并为RbaseArticle对象设置作者
    
    参数:
        cursor: 数据库游标
        article: RbaseArticle对象
        
    返回:
        作者列表和作者ID列表的元组
    """
    # 确保参数不为None
    authors = article.authors or ""
    corresponding_authors = article.corresponding_authors or ""
    
    # 处理普通作者
    author_list = [author.strip() for author in authors.split(',') if author.strip()]
    
    # 处理通讯作者
    corresponding_author_list = [author.strip() for author in corresponding_authors.split(',') if author.strip()]
    
    # 合并作者列表并去重
    all_authors_set = set(author_list + corresponding_author_list)
    all_authors = list(all_authors_set)
    
    author_ids = []
    corresponding_author_ids = []
    
    # 查询每个作者的ID并创建RbaseAuthor对象
    for author_name in all_authors:
        # 判断是英文名还是中文名
        is_english = all(ord(c) < 128 for c in author_name)
        
        # 创建作者对象
        if is_english:
            author_obj = RbaseAuthor(name=author_name, ename=author_name)
            # 查询英文名
            author_sql = """
            SELECT id FROM author WHERE ename = %s ORDER BY modified DESC
            """
            cursor.execute(author_sql, (author_name,))
        else:
            author_obj = RbaseAuthor(name=author_name, cname=author_name)
            # 查询中文名
            author_sql = """
            SELECT id FROM author WHERE cname = %s ORDER BY modified DESC
            """
            cursor.execute(author_sql, (author_name,))
        
        # 获取所有匹配的作者ID
        author_results = cursor.fetchall()
        if author_results:
            # 提取所有作者ID
            ids = [result['id'] for result in author_results]
            # 设置作者ID
            author_obj.set_author_ids(ids)
            # 添加到ID列表
            author_ids.extend(ids)
            
            # 如果是通讯作者，也添加到通讯作者ID列表
            is_corresponding = author_name in corresponding_author_list
            author_obj.is_corresponding = is_corresponding
            if is_corresponding:
                corresponding_author_ids.extend(ids)
        
        # 添加到文章中
        article.set_author(author_obj)
    
    return all_authors, author_ids, corresponding_author_list, corresponding_author_ids


def _process_keywords(source_keywords: str, mesh_keywords: str) -> List[str]:
    """
    处理关键词信息
    
    参数:
        source_keywords: 来源关键词字符串，分号分隔
        mesh_keywords: MeSH关键词字符串，分号分隔
        
    返回:
        去重后的合并关键词列表
    """
    # 确保参数不为None
    source_keywords = source_keywords or ""
    mesh_keywords = mesh_keywords or ""
    
    # 按分号分割关键词并去除空白
    source_keywords_list = [kw.strip() for kw in source_keywords.split(';') if kw.strip()]
    mesh_keywords_list = [kw.strip() for kw in mesh_keywords.split(';') if kw.strip()]
    
    # 合并关键词列表并去重
    keywords_set = set(source_keywords_list + mesh_keywords_list)
    keywords_list = list(keywords_set)
    
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
    
    # 禁用SSL验证，解决SSL连接问题
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    response = requests.get(url, verify=False)
    response.raise_for_status()  # 确保请求成功
    
    return response.text


def insert_to_vector_db(rbase_config: dict, 
                      articles: list[RbaseArticle],
                      collection_name: str = None,
                      collection_description: str = None,
                      force_new_collection: bool = False,
                      chunk_size: int = 1500,
                      chunk_overlap: int = 100,
                      batch_size: int = 256):
    """
    将文章数据加载到向量数据库
    
    参数:
        rbase_config: 配置字典，包含OSS配置和数据库配置
        articles: RbaseArticle对象列表，包含要处理的文章数据
        collection_name: 向量数据库集合名称
        collection_description: 向量数据库集合描述
        force_new_collection: 是否强制创建新集合
        chunk_size: 文本分块大小
        chunk_overlap: 文本分块重叠大小
        batch_size: 批处理大小
    """
    # 检查OSS配置
    rbase_oss_config = rbase_config.get('oss', {})
    
    # 确保OSS主机地址不为空
    if not rbase_oss_config.get('host'):
        raise ValueError("OSS主机地址不能为空")
    
    # 检查数据库配置
    rbase_db_config = rbase_config.get('database', {})
    
    # 初始化向量数据库
    vector_db = configuration.vector_db
    embedding_model = configuration.embedding_model
    file_loader = configuration.file_loader
    
    if collection_name is None:
        collection_name = vector_db.default_collection
    collection_name = collection_name.replace(" ", "_").replace("-", "_")
    
    # 初始化向量数据库集合
    init_vector_db(collection_name, collection_description, force_new_collection)
    
    # 获取MySQL连接
    conn = get_mysql_connection(rbase_db_config)
    
    all_docs = []
    
    try:
        with conn.cursor() as cursor:
            # 处理每篇文章
            for article in tqdm(articles, desc="处理文章文件"):
                txt_file_path = article.txt_file
                
                # 额外检查txt_file_path是否为空或不是以.md结尾
                if not txt_file_path or not txt_file_path.endswith('.md'):
                    print(f"跳过无效的文件路径: {txt_file_path}")
                    continue
                
                # 处理作者信息
                _process_authors(cursor, article)
                
                # 处理关键词
                keywords_list = _process_keywords(
                    article.source_keywords, 
                    article.mesh_keywords
                )
                
                # 创建临时文件保存markdown内容
                with tempfile.NamedTemporaryFile(suffix='.md', delete=False) as temp_file:
                    temp_path = temp_file.name
                    
                    # 从OSS服务器下载文件内容到临时文件
                    # 确保host和txt_file_path都不为空
                    host = rbase_oss_config.get('host', '')
                    if not host or not txt_file_path:
                        print(f"跳过下载：OSS主机地址或文件路径为空 - host: {host}, path: {txt_file_path}")
                        continue
                        
                    full_url = host + txt_file_path
                    
                    try:
                        content = _download_file_content(full_url)
                        temp_file.write(content.encode('utf-8'))
                    except Exception as e:
                        print(f"下载文件失败: {e}, URL: {full_url}")
                        continue
                
                # 加载临时文件
                docs = file_loader.load_file(temp_path)
                
                # 为每个文档添加元数据
                for doc in docs:
                    # 获取作者信息
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
                    
                    doc.metadata.update({
                        'title': article.title,
                        'authors': author_names,
                        'author_ids': author_ids,
                        'corresponding_authors': corresponding_author_names,
                        'corresponding_author_ids': corresponding_author_ids,
                        'keywords': keywords_list,
                        'pubdate': article.pubdate,
                        'article_id': article.article_id,
                        'reference': f"Article ID: {article.article_id}"
                    })
                
                all_docs.extend(docs)
                
                # 删除临时文件
                os.unlink(temp_path)
            
            # 分割文档为块
            chunks = split_docs_to_chunks(
                all_docs,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            
            # 嵌入向量
            chunks = embedding_model.embed_chunks(chunks, batch_size=batch_size)
            
            # 插入向量数据库
            return vector_db.insert_data(collection=collection_name, chunks=chunks)
            
    except Exception as e:
        # 发生异常时关闭连接
        close_mysql_connection()
        raise Exception(f"处理文章数据失败: {e}")


def load_from_rbase_db(rbase_config: dict, offset: int = 0, limit: int = 10) -> list[RbaseArticle]:
    """
    从Rbase数据库加载文章数据到向量数据库
    
    参数:
        rbase_config: 数据库配置字典
        offset: 查询起始位置
        limit: 查询数量限制
        
    返回:
        RbaseArticle对象列表
    """
    # 获取MySQL连接
    rbase_db_config = rbase_config.get('database', {})
    conn = get_mysql_connection(rbase_db_config)
    
    pdf_files = []
    try:
        with conn.cursor() as cursor:
            # 查询article_pdf_file表中符合条件的数据，并确保raw_article_id在article表中存在
            # 同时确保txt_file不为空且以.md结尾
            sql = """
            SELECT apf.id, apf.raw_article_id, apf.txt_file, 
                   a.title, a.authors, a.corresponding_authors, 
                   a.source_keywords, a.mesh_keywords, a.pubdate
            FROM article_pdf_file apf
            INNER JOIN article a ON apf.raw_article_id = a.id
            WHERE apf.user_id = 0 AND apf.status = 1 
            AND apf.txt_file IS NOT NULL 
            AND apf.txt_file LIKE '%%.md'
            ORDER BY apf.modified DESC LIMIT %s OFFSET %s
            """
            # 使用参数化查询，避免字符串格式化问题
            cursor.execute(sql, (limit, offset))
            pdf_files = cursor.fetchall()
    except Exception as e:
        # 发生异常时关闭连接
        close_mysql_connection()
        raise Exception(f"处理数据库数据失败: {e}")
    
    return [RbaseArticle(pdf) for pdf in pdf_files]
            