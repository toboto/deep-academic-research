#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
示例程序：使用SummaryRag生成科研摘要

本示例展示如何使用SummaryRag agent生成关于特定文章集合的摘要。
"""

import argparse
import logging
import os
import time
from typing import List

from deepsearcher import configuration
from deepsearcher.configuration import Configuration, init_config
from deepsearcher.tools import log
from deepsearcher.agent.summary_rag import SummaryRag
from deepsearcher.rbase.rbase_article import RbaseArticle
from deepsearcher.rbase_db_loading import load_from_rbase_db

def get_sample_articles(config: Configuration, limit: int = 10) -> List[RbaseArticle]:
    """
    获取示例文章列表
    
    Args:
        limit: 获取文章数量
        
    Returns:
        文章列表
    """
    rbase_config = config.rbase_settings
    articles = load_from_rbase_db(rbase_config, limit=limit)
    
    return articles


def main(
    query: str,
    output_file: str,
    articles_count: int = 10,
    min_words: int = 500,
    max_words: int = 2000,
    verbose: bool = False,
):
    """
    使用SummaryRag生成科研摘要的主函数

    Args:
        query: 查询主题
        output_file: 输出文件路径
        articles_count: 获取文章数量
        min_words: 最小字数限制
        max_words: 最大字数限制
        verbose: 是否启用详细输出
    """
    (lambda: (
        # 获取配置文件路径
        setattr(configuration, 'config', Configuration(
            os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                "config.rbase.yaml"
            )
        )),
        # 初始化配置
        init_config(configuration.config)
    ))()
    config = configuration.config

    # 获取示例文章
    log.color_print(f"正在获取{articles_count}篇示例文章...")
    articles = get_sample_articles(config, limit=articles_count)
    log.color_print(f"成功获取{len(articles)}篇文章")

    # 创建SummaryRag实例
    summary_rag = SummaryRag(
        reasoning_llm=configuration.reasoning_llm,
        writing_llm=configuration.writing_llm,
    )

    log.color_print(f"开始生成主题为'{query}'的科研总结...\n")
    start_time = time.time()

    # 调用SummaryRag生成摘要
    summary, _, tokens = summary_rag.query(
        query=query,
        articles=articles,
        min_words=min_words,
        max_words=max_words,
        verbose=verbose,
    )

    # 计算处理时间
    end_time = time.time()
    time_spent = end_time - start_time

    # 显示统计信息
    log.color_print("\n科研总结生成完成！")
    log.color_print(f"用时: {time_spent:.2f}秒")
    log.color_print(f"消耗tokens: {tokens}")

    # 将结果保存到文件
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(summary)

    log.color_print(f"总结已保存至: {output_file}")
    
    # 输出摘要内容预览
    preview_length = min(500, len(summary))
    log.color_print(f"\n总结预览:\n{summary[:preview_length]}...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate summary using SummaryRag")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    parser.add_argument("--query", "-q", help="Query for summary")
    parser.add_argument("--output", "-o", help="Output file path")
    parser.add_argument("--articles", "-a", type=int, default=10, help="Number of articles to use")
    parser.add_argument("--min_words", "-min", type=int, default=500, help="Minimum word count")
    parser.add_argument("--max_words", "-max", type=int, default=800, help="Maximum word count")
    args = parser.parse_args()

    if args.verbose:
        log.set_dev_mode(True)
        log.set_level(logging.DEBUG)
    else:
        log.set_dev_mode(False)
        log.set_level(logging.INFO)

    query = args.query if args.query else "请分析这个栏目收录的这些文章的研究主题和科研成果，给首次来到这个栏目的读者一个阅读指引"
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_file = (
        args.output
        if args.output
        else os.path.join(current_dir, "..", "outputs", "summary_result.md")
    )
    
    main(
        query=query, 
        output_file=output_file, 
        articles_count=args.articles,
        min_words=args.min_words,
        max_words=args.max_words,
        verbose=args.verbose
    ) 