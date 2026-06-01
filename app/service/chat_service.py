import asyncio
import json
import sys
import os

# 初始化 Agent 和 Graph
AGENT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "agent")
if AGENT_DIR not in sys.path:
    sys.path.insert(0, AGENT_DIR)

from core.workflow.graph_manager import AgentGraphManager
from core.memory.memory_manager import MemoryManager
from infra.cache import semantic_cache

# 会话取消事件：session_id -> asyncio.Event
_cancel_events: dict[str, asyncio.Event] = {}

def cancel_chat(session_id: str):
    """触发指定会话的取消信号。"""
    event = _cancel_events.get(session_id)
    if event:
        event.set()

def clear_cancel(session_id: str):
    _cancel_events.pop(session_id, None)

# Global variables for graph and memory
graph = None
memory = None

async def init_agent_system():
    global graph, memory
    if graph is None:
        print("[Start] 初始化 Multi-Agent 图编排...")
        graph_manager = AgentGraphManager()
        graph = graph_manager.build_graph()
        
        print("[Memory] 初始化 Memory 系统...")
        from config import get_settings
        settings = get_settings()
        memory = MemoryManager(
            redis_url=settings.redis_url,
            redis_ttl=settings.redis_ttl,
            milvus_host=settings.milvus_host,
            milvus_port=settings.milvus_port,
            milvus_api_key=settings.milvus_api_key,
            embedding_api_key=settings.dashscope_api_key,
        )
        await memory.initialize()
        await semantic_cache.initialize()
        print("[OK] Agent 系统初始化完成！")

async def _extract_memory_context(user_id: str, session_id: str, query: str) -> str:
    context_parts = []
    if memory and memory.short_term.available:
        history = await memory.short_term.get_messages(user_id, session_id)
        if history:
            recent_history = history[-10:] if len(history) > 10 else history
            context_parts.append("【近期对话历史】:")
            for msg in recent_history:
                role = "User" if msg["role"] == "user" else "Assistant"
                context_parts.append(f"{role}: {msg['content']}")
    
    if memory and memory.long_term.available:
        prefs = await memory.long_term.retrieve_relevant(user_id, query)
        if prefs:
            context_parts.append("\n【用户长期偏好/背景】:")
            for p in prefs:
                context_parts.append(f"- {p}")
                
    return "\n".join(context_parts)

async def stream_chat(query: str, user_id: str, session_id: str):
    # 注册取消事件
    cancel_event = asyncio.Event()
    _cancel_events[session_id] = cancel_event

    try:
        # 检查是否在开始前被取消
        if cancel_event.is_set():
            print(f"[Stop] 会话 {session_id} 在开始前被取消")
            yield f"data: {json.dumps({'done': True, 'cancelled': True})}\n\n"
            return

        cache_hit = await semantic_cache.get_cache(query, user_id)
        if cache_hit:
            response_text = cache_hit["answer"]
            print(
                f"⚡ 语义缓存命中: {cache_hit['level']} distance={cache_hit['distance']:.4f} matched='{cache_hit['matched_question']}'"
            )
        else:
            print("[Run] 进入 Agent 工作流推理...")
            mem_context = await _extract_memory_context(user_id, session_id, query)
            state = {
                "messages": [("user", query)],
                "user_id": user_id,
                "session_id": session_id,
                "memory_context": mem_context,
                "next_agent": "",
                "metadata": {}
            }
            config = {"configurable": {"user_id": user_id}}
            try:
                if cancel_event.is_set():
                    raise asyncio.CancelledError()
                result = await asyncio.to_thread(asyncio.run, graph.ainvoke(state, config=config)) if not asyncio.iscoroutinefunction(graph.ainvoke) else await graph.ainvoke(state, config=config)
                response_text = result["messages"][-1].content
            except asyncio.CancelledError:
                print(f"[Stop] 会话 {session_id} 在 Agent 推理中被取消")
                yield f"data: {json.dumps({'done': True, 'cancelled': True})}\n\n"
                return
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"[ERROR] Agent 工作流执行失败: {type(e).__name__}: {e}")
                response_text = f"抱歉，AI 服务暂时出现了异常（{type(e).__name__}），请稍后重试。"

        # 保存短时记忆
        if memory and memory.short_term.available:
            turn = [
                {"role": "user", "content": query},
                {"role": "assistant", "content": response_text},
            ]
            await memory.save_conversation(user_id, session_id, turn)

        # 流式返回大模型结果（流式过程中也检查取消信号）
        chunk_size = 5
        for i in range(0, len(response_text), chunk_size):
            if cancel_event.is_set():
                print(f"[Stop] 会话 {session_id} 在流式输出中被取消")
                yield f"data: {json.dumps({'done': True, 'cancelled': True})}\n\n"
                return
            chunk = response_text[i:i+chunk_size]
            yield f"data: {json.dumps({'content': chunk})}\n\n"
            await asyncio.sleep(0.02)

        yield f"data: {json.dumps({'done': True})}\n\n"
    finally:
        _cancel_events.pop(session_id, None)
