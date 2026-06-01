import os
import sys
import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from typing import Dict, Any
from pydantic import SecretStr
from langchain_core.runnables import RunnableConfig

from core.workflow.state import AgentState
from core.mcp.mcp_pool import get_mcp_pool

class PromotionAgentNode:
    """Promotion Agent for product sharing, commissions, and marketing materials."""
    def __init__(self):
        dotenv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), '.env')
        load_dotenv(dotenv_path)

        api_key = os.getenv("DASHSCOPE_API_KEY")
        
        self.llm = ChatOpenAI(
            api_key=SecretStr(api_key) if api_key else None,
            model=os.getenv("MODEL", "qwen-plus"),
            base_url=os.getenv("BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
            temperature=0.3,
        )

    async def __call__(self, state: AgentState) -> Dict[str, Any]:
        run_config: RunnableConfig = {
            "configurable": {"user_id": state.get("user_id", "unknown")}
        }
        
        pool = await get_mcp_pool()
        tools = await pool.get_tools(["get_promotable_products", "search_product_catalog", "get_promotion_materials", "generate_ai_poster"], caller_agent="promotion_agent")

        memory_context = state.get("memory_context", "")
        
        system_prompt = f"""You are an enthusiastic cloud platform promotion/marketing Agent.

Workflow:
1. If user says "I want to promote products" or "what products can I share", FIRST call get_promotable_products.
2. If user selects a specific product, call search_product_catalog with the product name.
3. Then call get_promotion_materials with the product_id.
4. Then call generate_ai_poster with a creative prompt.

Note: System auto-injects user_id.
Final response must include:
1. Enthusiastic opening mentioning commission rate.
2. Core product selling points.
3. Clear exclusive promotion link.
4. Markdown image showing the poster.

[Background Context]:
{memory_context if memory_context else "No background context."}
"""
        inner_agent = create_react_agent(
            model=self.llm,
            tools=tools,
            prompt=system_prompt
        )
        
        print("[PromotionAgent] Generating marketing and promotion materials...")
        
        result = await inner_agent.ainvoke(
            {"messages": state["messages"]}, 
            config=run_config
        )
        
        final_message = result["messages"][-1]
        return {"messages": [final_message]}