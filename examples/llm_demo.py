#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LLM模型演示脚本

该脚本用于验证三种不同LLM模型的配置和功能：
1. 标准LLM - 用于一般对话和查询
2. 推理LLM - 用于复杂推理任务
3. 写作LLM - 用于生成高质量文本内容
"""

import os
import sys
import logging
import platform

# 设置环境变量以抑制gRPC警告
os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "false"

# 根据操作系统设置不同的GRPC_POLL_STRATEGY
if platform.system() == "Linux":
    os.environ["GRPC_POLL_STRATEGY"] = "epoll1"
elif platform.system() == "Darwin":  # macOS
    os.environ["GRPC_POLL_STRATEGY"] = "poll"
else:  # Windows或其他系统
    os.environ["GRPC_POLL_STRATEGY"] = "poll"

# Suppress unnecessary log output
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("absl").setLevel(logging.ERROR)  # 抑制absl库的警告

# 抑制其他常见的警告源
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("grpc").setLevel(logging.ERROR)

# 设置警告过滤器
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

from deepsearcher.configuration import Configuration, init_config
from deepsearcher.tools import log

def get_model_name(model):
    """获取模型名称，如果模型没有model_name属性，则返回模型的model属性"""
    if model is None:
        return "未初始化"
    if hasattr(model, "model_name"):
        return model.model_name
    if hasattr(model, "model"):
        return model.model
    return str(model)

def test_standard_llm(llm):
    """
    测试标准LLM的简单问答功能
    
    Args:
        llm: 标准LLM模型实例
    
    Returns:
        bool: 测试是否成功
    """
    print("\n1. 测试标准LLM (简单问答):")
    if llm is None:
        print("错误: 标准LLM未初始化!")
        return False
    
    print(f"使用模型: {get_model_name(llm)}")
    try:
        messages = [{"role": "user", "content": "什么是人工智能？请简短回答。"}]
        response = llm.chat(messages)
        print(f"回答: {response.content}")
        return True
    except Exception as e:
        print(f"生成回答时出错: {str(e)}")
        return False

def test_reasoning_llm(reasoning_llm):
    """
    测试推理LLM的逻辑推理能力
    
    Args:
        reasoning_llm: 推理LLM模型实例
    
    Returns:
        bool: 测试是否成功
    """
    print("\n2. 测试推理LLM (逻辑推理):")
    if reasoning_llm is None:
        print("错误: 推理LLM未初始化!")
        return False
    
    print(f"使用模型: {get_model_name(reasoning_llm)}")
    try:
        reasoning_prompt = """
        解决以下逻辑推理问题：
        
        小明比小红年龄大。小刚比小明年龄小。小华比小明年龄大。
        请按照年龄从大到小排列这四个人。
        """
        messages = [{"role": "user", "content": reasoning_prompt}]
        response = reasoning_llm.chat(messages)
        print(f"回答: {response.content}")
        return True
    except Exception as e:
        print(f"生成回答时出错: {str(e)}")
        return False

def test_writing_llm(writing_llm):
    """
    测试写作LLM的创意写作能力
    
    Args:
        writing_llm: 写作LLM模型实例
    
    Returns:
        bool: 测试是否成功
    """
    print("\n3. 测试写作LLM (创意写作):")
    if writing_llm is None:
        print("错误: 写作LLM未初始化!")
        return False
    
    print(f"使用模型: {get_model_name(writing_llm)}")
    try:
        writing_prompt = "请写一段关于人工智能与人类未来关系的短文，约1000字。"
        messages = [{"role": "user", "content": writing_prompt}]
        response = writing_llm.chat(messages)
        print(f"回答: {response.content}")
        return True
    except Exception as e:
        print(f"生成回答时出错: {str(e)}")
        return False

def main():
    """
    主函数：初始化配置并测试三种不同的LLM模型
    """
    try:
        # 启用开发模式，以便能够看到调试日志
        log.set_dev_mode(True)
        log.set_level(log.logging.DEBUG)
        
        # 初始化配置
        yaml_file = os.path.join(current_dir, "..", "config.rbase.yaml")
        print(f"加载配置文件: {yaml_file}")
        if not os.path.exists(yaml_file):
            print(f"错误: 配置文件 {yaml_file} 不存在!")
            return
            
        config = Configuration(yaml_file)
        init_config(config)
        from deepsearcher.configuration import llm, reasoning_llm, writing_llm
        
        print("=" * 50)
        print("LLM模型演示")
        print("=" * 50)
        
        # 测试三种不同的LLM模型
        test_standard_llm(llm)
        test_reasoning_llm(reasoning_llm)
        test_writing_llm(writing_llm)
        
        print("\n" + "=" * 50)
        print("演示完成")
        print("=" * 50)
    except Exception as e:
        print(f"程序执行出错: {str(e)}")

if __name__ == "__main__":
    main() 