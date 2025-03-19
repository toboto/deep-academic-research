"""
从Rbase数据库加载文章数据到向量数据库的示例脚本

本脚本演示如何从Rbase数据库中提取文章数据，并将其加载到向量数据库中，
以便后续进行语义搜索和相似度查询。
"""

import logging
import os
from deepsearcher.rbase_db_loading import insert_to_vector_db, load_from_rbase_db
from deepsearcher.configuration import Configuration, init_config

# 抑制不必要的日志输出
logging.getLogger("httpx").setLevel(logging.WARNING)

def main():
    """
    主函数：从Rbase数据库加载文章并存入向量数据库
    
    流程:
    1. 初始化配置
    2. 从Rbase数据库加载文章数据
    3. 将文章数据插入到向量数据库
    """
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
    collection_name = "rbase_articles"
    collection_description = "Biomedical Research Literature Dataset"

    # 从Rbase数据库加载文章数据
    # offset和limit参数用于分页加载数据
    articles = load_from_rbase_db(config.rbase_settings, offset=0, limit=10)
    
    # 步骤3：将文章数据插入到向量数据库
    insert_result = insert_to_vector_db(
        rbase_config=config.rbase_settings,  # Rbase配置，包含数据库和OSS配置
        articles=articles,                   # 要插入的文章列表
        collection_name=collection_name,     # 向量数据库集合名称
        collection_description=collection_description,  # 集合描述
        force_new_collection=True            # 是否强制创建新集合（首次运行时设置为True，之后可设为False）
    )
    
    # 打印插入结果统计
    if insert_result:
        # 打印成功插入的数据条数
        print(f"成功插入数据 {insert_result.get('insert_count', 0)} 条")
        
        # 打印插入数据的ID范围
        ids = insert_result.get('ids', [])
        if ids:
            min_id = min(ids)
            max_id = max(ids)
            print(f"ID范围: {min_id} - {max_id}")
        else:
            print("未获取到插入ID")

    # 打印成功信息
    print(f"成功将Rbase数据库中的文章加载到向量数据库集合 '{collection_name}'")


if __name__ == "__main__":
    main() 