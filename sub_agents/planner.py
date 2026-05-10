from google.adk.agents.llm_agent import Agent

from scolAR.tools.planner_tools import submit_orthogonal_queries

planner_agent = Agent(
    model='gemini-3-flash-preview',
    name='planner_agent',
    description='A research director that reads the topic and formulates highly specific orthogonal search queries.',
    instruction="""You are a research director. Given a research topic, formulate exactly THREE
highly specific, orthogonal (non-overlapping)  but similar search queries that together cover the topic.

The three queries MUST be non-overlapping: no shared keywords beyond the core topic noun.
Each query should be a single concise search string (not a question).

You MUST call the `submit_orthogonal_queries` tool exactly once with these three queries.
Do not answer the user directly until the tool has been called.""",
    tools=[submit_orthogonal_queries],
)
