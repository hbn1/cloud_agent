"""澶氭櫤鑳戒綋浜戝鏈嶇郴缁熺殑涓诲叆鍙ｃ€?

璇ユā鍧楁彁渚涗簡涓€涓?CLI 鎺ュ彛锛岀敤浜庝笌鍩轰簬 LangGraph 鐨勫鏅鸿兘浣撶郴缁熻繘琛屼氦浜掞紝
骞堕泦鎴愪簡 FastMCP 宸ュ叿鍜岄暱/鐭湡鍐呭瓨銆?

鐢ㄦ硶:
    python main.py                    # 浜や簰妯″紡
    python main.py --query "浠€涔堟槸VPC"  # 鍗曟鏌ヨ妯″紡
"""

import argparse
import asyncio
import logging
import os
import sys
import uuid
from pathlib import Path
from typing import Any

# 鎶戝埗 macOS 涓婁笌 gRPC fork 鐩稿叧鐨勮鍛婏紙鏃犲锛屾潵鑷?pymilvus/grpcio锛?
os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
os.environ.setdefault("GRPC_TRACE", "")

# 灏嗙埗鐩綍娣诲姞鍒板鍏ヨ矾寰?
sys.path.insert(0, str(Path(__file__).parent))

# 纭繚鎵€鏈夊钩鍙颁笂鐨?stdin/stdout 閮戒娇鐢?UTF-8锛堜慨澶?macOS 缁堢涓枃杈撳叆闂锛?
if hasattr(sys.stdin, 'reconfigure'):
    sys.stdin.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from langchain_openai import ChatOpenAI
from pydantic import SecretStr
from config import get_settings
from core.memory import MemoryManager
from core.workflow.graph_manager import AgentGraphManager
from core.workflow.state import AgentState


def get_chat_llm():
    """Get a shared ChatOpenAI instance for memory extraction tasks."""
    settings = get_settings()
    api_key = settings.dashscope_api_key
    return ChatOpenAI(
        api_key=SecretStr(api_key) if api_key else None,
        model=settings.model,
        base_url=settings.base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1",
        temperature=0.0,
    )
