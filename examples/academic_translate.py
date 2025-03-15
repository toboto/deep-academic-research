"""
Script for testing the Academic Translator Agent.

This script is used to test the functionality of the AcademicTranslator class, including translation between Chinese and English.
"""

import os
import sys
import logging

# Add project root directory to system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deepsearcher import configuration
from deepsearcher.agent.academic_translator import AcademicTranslator
from deepsearcher.tools.log import color_print, error

# Suppress unnecessary log output
logging.getLogger("httpx").setLevel(logging.WARNING)

def test_zh_to_en():
    """
    Test translation from Chinese to English.
    """
    # Chinese test texts
    zh_texts = [
        "请分析临床能力对于临床医学专业学位研究生培养的重要性。",
        "量子计算利用量子力学原理进行信息处理，可以解决传统计算机难以处理的问题。",
        "基因编辑技术CRISPR-Cas9被誉为生物技术领域的革命性突破。"
    ]
    
    # Initialize configuration and translator
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.rbase.yaml")
    config = configuration.Configuration(config_path)
    configuration.config = config
    configuration.init_config(config)
    
    translator = AcademicTranslator()
    
    # Test translation
    color_print("\n===== Chinese to English Translation Test =====")
    for i, text in enumerate(zh_texts):
        color_print(f"\nOriginal {i+1}: {text}")
        try:
            translated = translator.translate(text, 'en')
            color_print(f"Translation {i+1}: {translated}")
        except Exception as e:
            error(f"Translation failed: {e}")

def test_en_to_zh():
    """
    Test translation from English to Chinese.
    """
    # English test texts
    en_texts = [
        "Please analyze the importance of clinical competence in the training of graduate students pursuing a professional degree in clinical medicine.",
        "Quantum computing uses quantum mechanical principles for information processing, solving problems that are difficult for traditional computers.",
        "The gene editing technology CRISPR-Cas9 is considered a revolutionary breakthrough in biotechnology."
    ]
    
    # Initialize configuration and translator
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.rbase.yaml")
    config = configuration.Configuration(config_path)
    configuration.config = config
    configuration.init_config(config)
    
    translator = AcademicTranslator()
    
    # Test translation
    color_print("\n===== English to Chinese Translation Test =====")
    for i, text in enumerate(en_texts):
        color_print(f"\nOriginal {i+1}: {text}")
        try:
            translated = translator.translate(text, 'zh')
            color_print(f"Translation {i+1}: {translated}")
        except Exception as e:
            error(f"Translation failed: {e}")

def test_same_language():
    """
    Test cases where source and target languages are the same.
    """
    # Test texts
    texts = [
        "深度学习是机器学习的一个分支。",  # Chinese
        "Deep learning is a branch of machine learning."  # English
    ]
    
    # Initialize configuration and translator
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.rbase.yaml")
    config = configuration.Configuration(config_path)
    configuration.config = config
    configuration.init_config(config)
    
    translator = AcademicTranslator()
    
    # Test translation
    color_print("\n===== Same Language Test =====")
    
    # Chinese text translated to Chinese
    color_print(f"\nOriginal: {texts[0]}")
    try:
        translated = translator.translate(texts[0], 'zh')
        color_print(f"Translation: {translated}")
        color_print(f"Is identical: {texts[0] == translated}")
    except Exception as e:
        error(f"Translation failed: {e}")
    
    # English text translated to English
    color_print(f"\nOriginal: {texts[1]}")
    try:
        translated = translator.translate(texts[1], 'en')
        color_print(f"Translation: {translated}")
        color_print(f"Is identical: {texts[1] == translated}")
    except Exception as e:
        error(f"Translation failed: {e}")

def test_mixed_language():
    """
    Test cases with mixed language content.
    """
    # Mixed language test texts
    mixed_texts = [
        "深度学习(Deep Learning)是机器学习的一个分支，基于artificial neural networks。",
        "Quantum computing（量子计算）uses quantum mechanical principles for information processing."
    ]
    
    # Initialize configuration and translator
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.rbase.yaml")
    config = configuration.Configuration(config_path)
    configuration.config = config
    configuration.init_config(config)
    
    translator = AcademicTranslator()
    
    # Test translation
    color_print("\n===== Mixed Language Test =====")
    
    # Mixed language translated to English
    color_print(f"\nOriginal: {mixed_texts[0]}")
    try:
        translated = translator.translate(mixed_texts[0], 'en')
        color_print(f"Translation (English): {translated}")
    except Exception as e:
        error(f"Translation failed: {e}")
    
    # Mixed language translated to Chinese
    color_print(f"\nOriginal: {mixed_texts[1]}")
    try:
        translated = translator.translate(mixed_texts[1], 'zh')
        color_print(f"Translation (Chinese): {translated}")
    except Exception as e:
        error(f"Translation failed: {e}")

if __name__ == "__main__":
    # Run tests
    test_zh_to_en()
    test_en_to_zh()
    test_same_language()
    test_mixed_language() 