#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
示例程序：使用PersonalRAG生成研究者学术综述

本示例展示如何使用PersonalRAG agent生成关于特定研究者的学术工作综述。
"""

import argparse
import logging
import os
import time
from datetime import datetime

from deepsearcher import configuration
from deepsearcher.agent.persoanl_rag import PersonalRAG
from deepsearcher.configuration import Configuration, init_config
from deepsearcher.tools import log

# Get current directory path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)


def main(
    query: str,
    output_file: str,
    verbose: bool = False,
    max_articles: int = 50,
    recent_months: int = 24,
    use_debug_cache: bool = False,
):
    """
    使用PersonalRAG生成研究者学术综述的主函数

    Args:
        query: 包含研究者姓名的查询
        output_file: 输出文件路径
        verbose: 是否输出详细日志
        max_articles: 最大处理文章数量
        recent_months: 最近几个月的文章优先级较高
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

    # 初始化PersonalRAG实例
    personal_rag = PersonalRAG(
        llm=configuration.llm,
        reasoning_llm=configuration.reasoning_llm,
        writing_llm=configuration.writing_llm,
        translator=configuration.academic_translator,
        embedding_model=configuration.embedding_model,
        vector_db=configuration.vector_db,
        route_collection=True,
        rbase_settings=config.rbase_settings,
    )

    log.color_print(f"开始生成研究者'{query}'的学术综述...")
    start_time = time.time()

    # 调用PersonalRAG生成综述
    response, _, tokens_used = personal_rag.query(
        query,
        verbose=verbose,
        max_articles=max_articles,
        recent_months=recent_months,
        top_k_per_section=20,
        top_k_accepted_results=20,
        vector_db_collection=config.provide_settings["vector_db"]["config"]["default_collection"],
        use_debug_cache=use_debug_cache,
    )

    # 计算处理时间
    end_time = time.time()
    time_spent = end_time - start_time

    # 显示统计信息
    log.color_print("研究者学术综述生成完成！")
    log.color_print(f"用时: {time_spent:.2f}秒")
    log.color_print(f"消耗tokens: {tokens_used}")

    # 将结果保存到文件
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(response)

    log.color_print(f"英文综述已保存至: {output_file}")

    with open(output_chs_file, "w", encoding="utf-8") as f:
        f.write(personal_rag.chinese_response)

    log.color_print(f"中文综述已保存至: {output_chs_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="生成研究者学术综述")
    parser.add_argument("--verbose", "-v", action="store_true", help="输出详细日志")
    parser.add_argument("--query", "-q", help="研究者姓名")
    parser.add_argument("--output", "-o", help="输出文件路径")
    parser.add_argument("--max-articles", "-m", type=int, default=50, help="最大处理文章数量")
    parser.add_argument(
        "--recent-months", "-r", type=int, default=24, help="最近几个月的文章优先级较高"
    )
    parser.add_argument("--use-debug-cache", "-d", action="store_true", help="使用调试缓存")
    args = parser.parse_args()

    if args.verbose:
        log.set_dev_mode(True)
        log.set_level(logging.DEBUG)
    else:
        log.set_dev_mode(False)
        log.set_level(logging.INFO)

    # 默认查询为Nature杂志创始主编
    query = args.query if args.query else "于君"

    # 生成输出文件名，包含时间戳
    timestamp = datetime.now().strftime("%m%d")

    output_file = (
        args.output
        if args.output
        else os.path.join(parent_dir, "outputs", f"{query}-personal-rag-{timestamp}.md")
    )
    query = f"请为我写一份关于{query}教授的科研综述"

    main(
        query,
        output_file,
        verbose=args.verbose,
        max_articles=int(args.max_articles),
        recent_months=int(args.recent_months),
        use_debug_cache=args.use_debug_cache,
    )
