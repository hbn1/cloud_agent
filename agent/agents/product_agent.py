import os
import asyncio
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
import typing

# create_react_agent 在 langgraph<1.0 位于 langgraph.prebuilt，
# langchain>=1.3 中已迁移并重命名为 langchain.agents.create_agent
if typing.TYPE_CHECKING:
    from langchain.agents import create_agent as create_react_agent
else:
    try:
        from langchain.agents import create_agent as create_react_agent
    except Exception:
        from langgraph.prebuilt import create_react_agent  # fallback（已弃用但仍可用）
from pydantic import SecretStr

from tools.vector_tool import query_vector_db
from tools.graph_tool import query_knowledge_graph
from tools.user_profile_tool import get_user_profile
from core.workflow.state import AgentState
from typing import Dict, Any

# ── 静态提示模板（仅含变量占位符）─────────────────────────────
BASE_SYSTEM_PROMPT = """你是一个专业的云服务平台【产品咨询Agent】。
你的任务是解答用户关于云产品（如云服务器ECS、专有网络VPC等）的疑问。
你有两个强大的检索工具可供使用：

1. query_vector_db（向量数据库检索）:
   - 适用场景：查询大段的概念解释、操作步骤说明、详细的规则政策。
   - 特点：擅长处理模糊的语义匹配和长文本阅读。

2. query_knowledge_graph（知识图谱检索）:
   - 适用场景：查询云产品的架构、实体包含关系、具体的配置数值与限制、组合查询等结构化数据。
   - 特点：擅长处理精确的属性、关系和多跳拓扑查询。

工作要求：
- 仔细分析用户的问题，自主决定使用哪个工具，或者结合使用两个工具（如果问题很复杂）。
- 如果问题偏结构化参数（如网卡数、带宽、实例关系），优先尝试 query_knowledge_graph；若图谱查询超时或失败，自动降级为 query_vector_db 并继续完成回答。
- 如果问题同时包含结构化参数和规则解释，建议组合使用两个工具；但以可用性优先，不强制必须调用图谱。
- 优先通过工具获取事实依据，不要凭空捏造（幻觉）。
- 获取到信息后，请以专业、清晰、友好的客服口吻组织回答。
- 如果工具返回没有找到相关信息，请诚实地告诉用户目前知识库中没有相关记录。
- 答案来源只能引用工具原始返回中明确出现的来源名；禁止编造任何文档名、版本号、白皮书名称。
- 如果某工具未调用或调用失败，不要在"答案来源"中提及该工具，也不要解释为什么失败。
- 每次最终回答的结尾，只需列出实际获取到数据的来源，不要输出"可信度"，也不要输出"未使用"的工具。
  格式示例：
  答案来源：
  - 向量检索：xxx.md

【系统提供的用户记忆/背景上下文】:
{memory_context}
"""


class ProductAgentNode:
    """包装了 LangGraph create_react_agent 的节点类，供主图编排时直接调用。"""

    def __init__(self):
        dotenv_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"
        )
        load_dotenv(dotenv_path)

        raw_api_key = os.getenv("DASHSCOPE_API_KEY")
        api_key_secret = SecretStr(raw_api_key) if raw_api_key else None

        self.llm = ChatOpenAI(
            api_key=api_key_secret,
            model=os.getenv("MODEL", "qwen-plus"),
            base_url=os.getenv(
                "BASE_URL",
                "https://dashscope.aliyuncs.com/compatible-mode/v1",
            ),
            temperature=0.1,
        )
        self.tools = [query_vector_db, query_knowledge_graph, get_user_profile]

        # 编译一次，重复使用
        self.inner_agent = create_react_agent(
            model=self.llm,
            tools=self.tools,
        )

    async def __call__(self, state: AgentState) -> Dict[str, Any]:
        memory_context = state.get("memory_context", "") or "暂无背景上下文。"
        system_prompt = BASE_SYSTEM_PROMPT.format(memory_context=memory_context)

        # 将动态 SystemPrompt 注入消息头部，避免每次重建 inner_agent
        messages = [SystemMessage(content=system_prompt), *state["messages"]]

        try:
            result = await self.inner_agent.ainvoke({"messages": messages})
        except Exception as e:
            print(f"[ProductAgent] Agent 调用失败: {e}")
            raise

        final_message = result["messages"][-1]
        return {"messages": [final_message]}


def get_product_agent() -> ProductAgentNode:
    """获取 ProductAgentNode 实例（工厂函数）。"""
    return ProductAgentNode()


# ── 独立测试入口 ─────────────────────────────────────────────
async def _test_main():
    agent = get_product_agent()

    print("[Agent] ProductAgent 已启动！（输入 'quit' 或 'exit' 退出）")
    print("=" * 50)
    print("您可以尝试问我：")
    print("1. [图谱测试] ecs.g8a.4xlarge 实例能挂载多少块弹性网卡？")
    print("2. [向量测试] 五天无理由退款有什么限制条件吗？")
    print("3. [混合测试] 什么是专有网络VPC？另外华北2（北京）地域有哪些实例规格族？")
    print("=" * 50)

    while True:
        user_input = input("\n用户: ")
        if user_input.lower() in ("quit", "exit"):
            break
        if not user_input.strip():
            continue

        # 构造测试状态
        test_state: AgentState = {
            "messages": [HumanMessage(content=user_input)],
            "next_agent": "product",
            "user_id": "test_user",
            "session_id": "test_session",
            "memory_context": "",
            "metadata": {},
        }

        try:
            result = await agent(test_state)
            final_msg = result["messages"][-1]
            print(f"\nAgent: {final_msg.content}")
        except Exception as e:
            print(f"\n[ERROR] {e}")


if __name__ == "__main__":
    asyncio.run(_test_main())
