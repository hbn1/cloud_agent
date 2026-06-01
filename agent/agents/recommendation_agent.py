import os
import sys
import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from core.workflow.state import AgentState
from typing import Dict, Any, cast
from pydantic import SecretStr
from langchain_core.runnables import RunnableConfig
from core.mcp.mcp_pool import get_mcp_pool

class RecommendationAgent:
    """Intelligent recommendation Agent for cloud product selection."""
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
        memory_context = state.get("memory_context", "")
        
        runnable_config: RunnableConfig = {
            "configurable": {"user_id": state.get("user_id", "unknown")}
        }
        
        pool = await get_mcp_pool()
        mcp_tools = await pool.get_tools(["get_promotable_products", "search_product_catalog", "get_promotion_materials"], caller_agent="recommendation_agent")
        
        from tools.vector_tool import query_vector_db
        tools = [query_vector_db] + mcp_tools

        system_prompt = f"""You are a senior cloud architect and intelligent recommendation Agent.
Your task is to recommend the most suitable cloud products based on user business scenarios.

Workflow:
1. Analyze user requirements (business type, concurrency, budget, region).
2. Call get_promotable_products or search_product_catalog for current catalog.
3. For selection recommendations, call query_vector_db for technical specs.
4. Recommend 1-3 best-fit products with professional reasoning.
5. Call get_promotion_materials for purchase links.

Requirements:
- Professional and enthusiastic tone.
- Include specific instance types/product names.
- Never recommend products not in the catalog.
- List sources at the end.

[Background Context]:
{memory_context if memory_context else "No background context."}
"""
        inner_agent = create_react_agent(
            model=self.llm,
            tools=tools,
            prompt=system_prompt
        )
        
        print("[Search] [RecommendationAgent] Performing intelligent product selection...")
        
        result = await inner_agent.ainvoke(
            {"messages": state["messages"]},
            config=runnable_config
        )
        final_message = result["messages"][-1]
        return {"messages": [final_message]}