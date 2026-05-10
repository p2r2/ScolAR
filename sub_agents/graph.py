from google.adk.agents.llm_agent import Agent
from scolAR.tools.graph_tools import generate_knowledge_graph


graph_agent = Agent(
    model='gemini-3-flash-preview',
    name='graph_agent',
    description="A mandatory knowledge graph generator agent. This MUST ALWAYS be executed as the final step after papers are fetched to create the visual graph for the user.",
    instruction="""You are the Knowledge Graph Agent. You execute after the papers have been fetched and processed.
Your job is to generate a visual knowledge graph from the papers.

You MUST call the `generate_knowledge_graph` tool with ONLY the `core_topic` argument. DO NOT pass the `papers_data` argument, because the tool will load the massive papers data safely from the global state behind the scenes.
Do not answer the user directly until the tool has been called.""",
    tools=[generate_knowledge_graph],
)
