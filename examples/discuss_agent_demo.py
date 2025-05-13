#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
示例程序：使用DiscussAgent进行学术对话

本示例展示如何使用DiscussAgent进行学术讨论，包括意图判断和知识检索功能。
"""

import argparse
import logging
import os
import time
import json
from typing import List, Dict

from deepsearcher import configuration
from deepsearcher.configuration import Configuration, init_config
from deepsearcher.tools import log
from deepsearcher.agent.discuss_agent import DiscussAgent

def main(
    query: str,
    output_file: str,
    user_action: str = "浏览学术论文",
    background: str = "",
    verbose: bool = False,
):
    """
    使用DiscussAgent进行学术对话的主函数

    Args:
        query: 用户问题
        output_file: 输出文件路径
        user_action: 用户当前正在进行的操作
        background: 对话背景信息
        verbose: 是否启用详细输出
    """
    # 初始化配置
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
    
    # 创建DiscussAgent实例
    discuss_agent = DiscussAgent(
        llm=configuration.writing_llm,
        reasoning_llm=configuration.reasoning_llm,
        translator=configuration.academic_translator,
        embedding_model=configuration.embedding_model,
        vector_db=configuration.vector_db,
        verbose=verbose
    )

    # 模拟对话历史
    history = [
        {
            "content": "我想了解人工智能在医疗领域的应用",
            "role": "user",
        },
        {
            "content": "人工智能在医疗领域有多种应用，包括疾病诊断、医学影像分析、药物研发和个性化治疗方案等。具体来说，AI可以分析大量医学数据以识别疾病模式，提高诊断准确性；通过计算机视觉技术分析X光、CT和MRI等医学影像；加速药物筛选和设计过程；根据患者的基因组数据和病史制定个性化治疗方案。",
            "role": "assistant"
        },
    ]

    # 设置请求参数（用于过滤检索结果）
    request_params = {
        "pubdate": int(time.time()) - 5 * 365 * 24 * 3600,  # 5年内的文献
        "impact_factor": 3.0  # 影响因子大于3的文献
    }

    log.color_print(f"开始处理用户问题: '{query}'...\n")
    start_time = time.time()

    # 调用DiscussAgent处理用户问题
    answer, retrieval_results, usage = discuss_agent.query(
        query=query,
        user_action=user_action,
        background=background,
        history=history,
        request_params=request_params,
        verbose=verbose
    )

    # 计算处理时间
    end_time = time.time()
    time_spent = end_time - start_time

    # 显示统计信息
    log.color_print("\n用户问题处理完成！")
    log.color_print(f"用时: {time_spent:.2f}秒")
    if isinstance(usage, dict) and "total_tokens" in usage:
        log.color_print(f"消耗tokens: {usage['total_tokens']}")

    # 显示检索结果统计
    log.color_print(f"检索到的文献数量: {len(retrieval_results)}")
    
    # 保存结果
    result = {
        "query": query,
        "answer": answer,
        "retrieval_count": len(retrieval_results),
        "retrieval_ids": [r.metadata.get("reference_id", "") for r in retrieval_results],
        "time_spent": time_spent,
        "usage": usage
    }
    
    # 将结果保存到文件
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    log.color_print(f"结果已保存至: {output_file}")
    
    # 输出回答内容
    log.color_print(f"\n回答:\n{answer}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="学术对话代理演示")
    parser.add_argument("--verbose", "-v", action="store_true", help="启用详细输出")
    parser.add_argument("--query", "-q", help="用户问题")
    parser.add_argument("--output", "-o", help="输出文件路径")
    parser.add_argument("--user_action", "-a", default="浏览学术论文", help="用户当前行为")
    parser.add_argument("--background", "-b", default="", help="对话背景信息")
    args = parser.parse_args()

    if args.verbose:
        log.set_dev_mode(True)
        log.set_level(logging.DEBUG)
    else:
        log.set_dev_mode(False)
        log.set_level(logging.INFO)

    query = args.query if args.query else "最新的深度学习技术在医疗诊断中有哪些突破性应用？能否给出一些具体例子？"
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_file = (
        args.output
        if args.output
        else os.path.join(current_dir, "..", "outputs", "discuss_result.json")
    )
    
    main(
        query=query, 
        output_file=output_file, 
        user_action=args.user_action,
        background=args.background,
        verbose=args.verbose
    ) 