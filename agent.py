from google.adk.agents.sequential_agent import SequentialAgent
from scolAR.sub_agents.fetcher import fetcher_agent
from scolAR.sub_agents.planner import planner_agent
from scolAR.sub_agents.graph import graph_agent



root_agent = SequentialAgent(
    name="root_agent",
    sub_agents=[planner_agent,fetcher_agent,graph_agent],

)
