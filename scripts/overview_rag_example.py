#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
示例程序：使用OverviewRAG生成科研综述

本示例展示如何使用OverviewRAG agent生成关于特定研究主题的综述文章。
"""

import argparse
import logging
import os
import time

from deepsearcher import configuration
from deepsearcher.configuration import Configuration, init_config
from deepsearcher.tools import log

# Get current directory path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)


def main(
    query: str,
    output_file: str,
    verbose: bool = False,
    vector_db_collection: str = "dev_rag_articles",
    top_k_per_section: int = 80,
    top_k_accepted_results: int = 80,
):
    """
    使用OverviewRAG生成科研综述的主函数

    Args:
        query: 查询主题
        output_file: 输出文件路径
    """
    yaml_file = os.path.join(current_dir, "..", "config.rbase.yaml")

    # 定义中文输出文件名
    file_name, file_ext = os.path.splitext(output_file)
    output_chs_file = f"{file_name}.chs{file_ext}"

    # 从YAML文件加载配置
    config = Configuration(yaml_file)

    # 应用配置，使其在全局生效
    configuration.config = config
    init_config(config)

    log.color_print(f"开始生成主题为'{query}'的科研文章...\n")
    start_time = time.time()

    # 调用OverviewRAG生成综述
    from deepsearcher.configuration import overview_rag

    response, _, tokens_used = overview_rag.query(
        query,
        verbose=verbose,
        top_k_per_section=top_k_per_section,
        top_k_accepted_results=top_k_accepted_results,
        route_collection=False,
        vector_db_collection=vector_db_collection,
    )

    # 计算处理时间
    end_time = time.time()
    time_spent = end_time - start_time

    # 显示统计信息
    log.color_print("\n科研文章生成完成！")
    log.color_print(f"用时: {time_spent:.2f}秒")
    log.color_print(f"消耗tokens: {tokens_used}")

    # 将结果保存到文件
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(response)

    log.color_print(f"英文综述已保存至: {output_file}")

    with open(output_chs_file, "w", encoding="utf-8") as f:
        f.write(overview_rag.chinese_response)

    log.color_print(f"中文综述已保存至: {output_chs_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate overview article using OverviewRAG")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    parser.add_argument("--query", "-q", help="Query for overview article")
    parser.add_argument("--output", "-o", help="Output file path")
    parser.add_argument(
        "--vector_db_collection",
        "-vdc",
        default="dev_rag_articles",
        help="Vector database collection name",
    )
    parser.add_argument("--top_k_per_section", "-tkps", default=80, help="Top k per section")
    parser.add_argument(
        "--top_k_accepted_results", "-tkar", default=80, help="Top k accepted results"
    )
    args = parser.parse_args()

    if args.verbose:
        log.set_dev_mode(True)
        log.set_level(logging.DEBUG)
    else:
        log.set_dev_mode(False)
        log.set_level(logging.INFO)

    query = args.query if args.query else "请写一篇有关阿克曼氏菌方面的综述"
    output_file = (
        args.output
        if args.output
        else os.path.join(current_dir, "..", "outputs", "academic_overview.md")
    )
    main(query, output_file, verbose=args.verbose)
