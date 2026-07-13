"""
统一配置模块 —— 从 .env 文件读取 API Key
使用方法: from config import DEEPSEEK_API_KEY, get_client
"""
import os
from openai import OpenAI

# 尝试加载 .env 文件
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv 未安装时静默跳过，使用系统环境变量

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")


def get_client():
    """获取配置好的 DeepSeek OpenAI 兼容客户端"""
    if not DEEPSEEK_API_KEY:
        raise RuntimeError(
            "未设置 DEEPSEEK_API_KEY！\n"
            "1. 复制 .env.example 为 .env\n"
            "2. 填入你的 API Key\n"
            "3. 或设置系统环境变量 DEEPSEEK_API_KEY"
        )
    return OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
