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

class FinOpsAgentNode:
    """
    FinOps Agent: cost optimization and architectural diagnostics.
    Receives context from BillingAgent and analyzes monitoring data.
    """
    def __init__(self):
        dotenv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), '.env')
        load_dotenv(dotenv_path)

        api_key = os.getenv("DASHSCOPE_API_KEY")
        
        self.llm = ChatOpenAI(
            api_key=SecretStr(api_key) if api_key else None,
            model=os.getenv("MODEL", "qwen-plus"),
            base_url=os.getenv("BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
            temperature=0.1, 
        )
        
        # No longer creating MCP client in __init__ or __call__

    async def __call__(self, state: AgentState) -> Dict[str, Any]:
        config: RunnableConfig = {"configurable": {"user_id": state.get("user_id", "unknown")}}

        # Use shared MCP pool instead of creating a new client each time
        pool = await get_mcp_pool()
        tools = await pool.get_tools(["query_user_instances", "analyze_instance_usage"], caller_agent="finops_agent")
        
        system_prompt = f"""You are a professional FinOps cost optimization expert.
You just received context from the Billing Agent.

Your task:
1. Carefully read the conversation history, prioritize extracting the instance ID the user wants to optimize.
2. If no instance_id in context, first call query_user_instances to get the user instance list, prioritizing Running ECS instances for analysis.
3. Call analyze_instance_usage tool to get recent CPU, memory etc. monitoring data.
4. Analyze whether the instance has "RESOURCES_IDLE" situation.
5. Give professional cost reduction recommendations. If CPU is very low long-term, suggest downgrading.

Note: System auto-injects user_id, pass placeholder "auto" when calling tools.
- Never fabricate instance IDs, monitoring metrics or cost savings.
"""
        inner_agent = create_react_agent(
            model=self.llm,
            tools=tools,
            prompt=system_prompt
        )
        
        print("[Idea] [FinOpsAgent] Analyzing instance monitoring metrics, generating cost optimization report...")
        
        result = await inner_agent.ainvoke(
            {"messages": state["messages"]}, 
            config=config
        )
        
        final_message = result["messages"][-1]
        return {"messages": [final_message], "next_agent": ""}