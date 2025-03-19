#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
示例程序：使用OverviewRAG生成科研综述

本示例展示如何使用OverviewRAG agent生成关于特定研究主题的综述文章。
"""

import os
import sys
import time

# 将项目根目录添加到路径中，以便导入deepsearcher模块
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from deepsearcher.tools import log
from deepsearcher.configuration import Configuration, init_config
from deepsearcher import configuration

def main():
    """
    使用OverviewRAG生成科研综述的主函数
    """
    yaml_file = os.path.join(current_dir, "..", "config.rbase.yaml")
    
    # 从YAML文件加载配置
    config = Configuration(yaml_file)

    # 应用配置，使其在全局生效
    configuration.config = config
    init_config(config)
    
    # 设置查询主题
    query = "请写一篇有关动脉硬化方面的综述"
    
    log.color_print(f"开始生成主题为'{query}'的综述...\n")
    start_time = time.time()
    
    # 调用OverviewRAG生成综述
    from deepsearcher.configuration import overview_rag
    response, _, tokens_used = overview_rag.query(query)
    
    # 计算处理时间
    end_time = time.time()
    time_spent = end_time - start_time
    
    # 显示统计信息
    log.color_print(f"\n综述生成完成！")
    log.color_print(f"用时: {time_spent:.2f}秒")
    log.color_print(f"消耗tokens: {tokens_used}")
    
    # 将结果保存到文件
    output_file = os.path.join(current_dir, "..", "outputs", "arterial_sclerosis_review.md")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(response)
    
    log.color_print(f"综述已保存至: {output_file}")


if __name__ == "__main__":
    main() 