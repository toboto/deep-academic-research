"""
从Rbase数据库加载文章数据到向量数据库的示例脚本

本脚本演示如何从Rbase数据库中提取文章数据，并将其加载到向量数据库中，
以便后续进行语义搜索和相似度查询。
"""

import logging
import argparse
import os
from tqdm import tqdm

from deepsearcher import configuration
from deepsearcher.configuration import Configuration, init_config
from deepsearcher.rbase_db_loading import insert_to_vector_db, load_markdown_articles, save_vector_db_log, log_raw_article_deleted
from deepsearcher.tools.log import info, set_dev_mode, set_level

# 抑制不必要的日志输出
logging.getLogger("httpx").setLevel(logging.WARNING)


def main(ver: int, env: str, collection_name: str, offset: int = 0, limit: int = 10, **kwargs):
    """
    主函数：从Rbase数据库加载文章并存入向量数据库

    流程:
    1. 初始化配置
    2. 从Rbase数据库加载文章数据
    3. 将文章数据插入到向量数据库
    """
    base_id = kwargs.get("base_id", 0)
    doc_rebuild = kwargs.get("doc_rebuild", False)
    force_new_collection = kwargs.get("force_new_collection", False)
    collection_description = kwargs.get("collection_description", None)

    # 步骤1：初始化配置
    # 获取当前脚本所在目录，并构建配置文件的路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    yaml_file = os.path.join(current_dir, "..", "config.rbase.yaml")

    # 从YAML文件加载配置
    config = Configuration(yaml_file)

    # 应用配置，使其在全局生效
    init_config(config)

    # 步骤2：从Rbase数据库加载文章数据到向量数据库
    # 设置向量数据库集合名称和描述
    embedding_model = config.provide_settings["embedding"]["config"]["model"]
    # 处理embedding_model字符串，提取模型名称并规范化格式
    if '/' in embedding_model:
        embedding_model = embedding_model.split('/')[-1]
    embedding_model = embedding_model.replace('-', '_').replace('.', '_')

    if not collection_name:
        collection_name = f"{env}_rbase_{embedding_model}_{ver}"
    
    if not collection_description:
        collection_description = "Academic Research Literature Dataset"
    
    # 从Rbase数据库加载文章数据
    # offset和limit参数用于分页加载数据
    articles = load_markdown_articles(config.rbase_settings, offset=offset, limit=limit, base_id=base_id, doc_rebuild=doc_rebuild)

    insert_count = 0
    insert_min_id = 0
    insert_max_id = 0
    # 步骤3：将文章数据插入到向量数据库
    for article in tqdm(articles, desc="Processing articles"):
        insert_result = insert_to_vector_db(
            rbase_config=config.rbase_settings,  # Rbase配置，包含数据库和OSS配置
            articles=[article],  # 要插入的文章列表
            collection_name=collection_name,  # 向量数据库集合名称
            collection_description=collection_description,  # 集合描述
            force_new_collection=force_new_collection,  # 是否强制创建新集合（首次运行时设置为True，之后可设为False）
        )

        # 打印插入结果统计
        if insert_result:
            log_raw_article_deleted(config.rbase_settings, article.raw_article_id, collection_name)
            ids = insert_result.get("ids", [])
            min_id = min(ids) if ids else 0
            max_id = max(ids) if ids else 0
            save_vector_db_log(config.rbase_settings, 
                               article.raw_article_id, 
                               collection_name, 
                               operation='insert',
                               id_from=min_id,
                               id_to=max_id)
            insert_count += insert_result.get('insert_count', 0)
            insert_min_id = min(insert_min_id, min_id) if insert_min_id else min_id
            insert_max_id = max(insert_max_id, max_id) if insert_max_id else max_id

    # 打印成功信息
    configuration.vector_db.flush(collection_name)
    info(f"成功将Rbase数据库中的文章加载到向量数据库集合 '{collection_name}'")
    info(f"成功插入数据 {insert_count} 条，ID范围: {insert_min_id} - {insert_max_id}")

def parse_args():
    """
    解析命令行参数
    
    Returns:
        argparse.Namespace: 包含解析后的参数
    """
    parser = argparse.ArgumentParser(description='创建Rbase向量数据库')
    parser.add_argument('--ver', type=int, default=1, help='版本号，默认为1')
    parser.add_argument('--env', '-e', type=str, default='dev', help='环境，默认为dev')
    parser.add_argument('--offset', '-o', type=int, default=0, help='偏移量，默认为0')
    parser.add_argument('--limit', '-l', type=int, default=10, help='限制数量，默认为10')
    parser.add_argument('--base_id', '-b', type=int, default=0, help='基础ID，默认为0')
    parser.add_argument('--doc_rebuild', '-r', action='store_true', help='是否重建文档，默认为False')
    parser.add_argument('--collection_name', '-n', type=str, help='集合名称，默认为None')
    parser.add_argument('--collection_description', '-d', type=str, default='Academic Research Literature Dataset', 
                        help='集合描述')
    parser.add_argument('-f', '--force_new_collection', action='store_true', help='是否强制创建新集合，默认为False')
    parser.add_argument('-v', '--verbose', action='store_true', help='是否打印详细信息，默认为False')
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if args.verbose:
        set_dev_mode(True)
        set_level(logging.DEBUG)
    main(args.ver, args.env, args.collection_name, args.offset, args.limit, 
        collection_description=args.collection_description,
        force_new_collection=args.force_new_collection, 
        doc_rebuild=args.doc_rebuild, 
        base_id=args.base_id)
