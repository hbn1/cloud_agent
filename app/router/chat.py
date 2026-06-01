from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from schemas.chat import ChatRequest, StopChatRequest
from service.chat_service import stream_chat, cancel_chat

router = APIRouter()

@router.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """
    处理多智能体聊天请求，并使用 SSE (Server-Sent Events) 返回流式响应。
    如果命中 L1 语义缓存，将直接返回缓存结果。
    否则进入 Agent 图编排流程。
    """
    return StreamingResponse(
        stream_chat(request.query, request.user_id, request.session_id),
        media_type="text/event-stream"
    )

@router.post("/chat/stop")
async def stop_chat(request: StopChatRequest):
    """取消指定会话的正在进行的 AI 生成。"""
    cancel_chat(request.session_id)
    return {"status": "cancelled", "session_id": request.session_id}
