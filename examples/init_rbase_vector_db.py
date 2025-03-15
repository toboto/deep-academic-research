"""
初始化向量数据库示例脚本

本脚本演示如何初始化和配置向量数据库(Milvus)，为后续的文档检索和查询做准备。
向量数据库是存储文档嵌入向量的关键组件，支持高效的相似性搜索。
"""

import logging
from deepsearcher.offline_loading import load_from_local_files  # 用于从本地文件加载数据
from deepsearcher.online_query import query  # 用于执行在线查询
from deepsearcher.configuration import Configuration, init_config  # 配置管理工具
from deepsearcher import configuration  # 全局配置访问

# 抑制第三方库的不必要日志输出
logging.getLogger("httpx").setLevel(logging.WARNING)

def main():
    """
    主函数：初始化并配置向量数据库
    
    流程:
    1. 创建并配置系统设置
    2. 初始化向量数据库连接
    3. 创建或连接到指定的集合(collection)
    """
    # 步骤1: 初始化配置对象
    config = Configuration()

    # 步骤2: 配置向量数据库提供者为Milvus
    # 这里使用空字典{}表示使用默认配置参数
    config.set_provider_config("vector_db", "Milvus", {})

    # 步骤3: 应用配置，使其在全局生效
    init_config(config)

    try:
        # 获取已配置的向量数据库实例
        vector_db = configuration.vector_db

        # 设置集合参数
        collection_name = "rbase_basics"  # 集合名称
        collection_description = "Rbase Basics"  # 集合描述
        force_new_collection = False  # 是否强制创建新集合(若为True则删除同名集合)

        # 处理集合名称，替换空格和连字符为下划线，确保名称符合Milvus要求
        collection_name = collection_name.replace(" ", "_").replace("-", "_")
        
        # 获取嵌入模型，用于确定向量维度
        embedding_model = configuration.embedding_model
        
        # 初始化集合，设置向量维度和其他参数
        vector_db.init_collection(
            dim=embedding_model.dimension,  # 向量维度，由嵌入模型决定
            collection=collection_name,  # 集合名称
            description=collection_description,  # 集合描述
            force_new_collection=force_new_collection,  # 是否强制创建新集合
        )
    except Exception as e:
        # 捕获并记录初始化过程中的任何错误
        logging.error(f"Error initializing vector database: {e}")


if __name__ == "__main__":
    main()
