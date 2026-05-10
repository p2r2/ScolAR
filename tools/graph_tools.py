import ast
import json
import logging
import networkx as nx
import matplotlib.pyplot as plt
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

from google.adk.tools.tool_context import ToolContext

def generate_knowledge_graph(core_topic: str, papers_data: str = "", output_path: str = "knowledge_graph.png", tool_context: ToolContext = None) -> str:
    """
    An intelligent agent responsible for parsing structured academic summaries 
    and constructing a specialized Knowledge Graph.

    This agent ingests categorized summaries of academic papers (structured 
    around core topics, orthogonal queries, and paper metadata) and translates 
    them into formal nodes and edges representing conceptual and bibliometric relationships.

    The generated graph schema adheres to the following ontology:
        - (CoreTopic) -[HAS_SUBTOPIC]-> (Category)
        - (Category) -[INCLUDES_PAPER]-> (Paper)
        - (Paper) -[PUBLISHED_IN]-> (Journal)
        - (Paper) -[HAS_DOI]-> (DOI)
        
    Attributes:
        graph_client (Any): The connection client for the target graph database 
            or rendering engine (e.g., Neo4j driver, NetworkX graph, Graphify client).
        strict_mode (bool): If True, the agent will reject papers missing critical 
            metadata (like DOIs). Defaults to False.

    Args:
        core_topic: The core topic or research area the papers revolve around.
        papers_data: Optionally pass stringified papers, however the state will be prioritized.
        output_path: File path to save the generated graph image.
    """
    try:
        papers = []
        if tool_context and "fetched_papers" in tool_context.state:
            papers = tool_context.state["fetched_papers"]
        
        if not papers and papers_data:
            # Safely parse the papers data since LLMs often output stringified lists/JSON
            try:
                papers = json.loads(papers_data)
            except json.JSONDecodeError:
                try:
                    papers = ast.literal_eval(papers_data)
                except (ValueError, SyntaxError):
                    papers = eval(papers_data)  # Fallback if literal_eval fails on minor formatting issues
                    
        if not isinstance(papers, list):
            return "Error: Could not retrieve a valid list of papers from state or arguments."

        # Initialize NetworkX Directed Graph
        G = nx.DiGraph()

        # Add Core Topic node
        G.add_node(core_topic, type="CoreTopic")

        # Iterate over papers and map them based on our ontology
        for paper in papers:
            title = paper.get('title', 'Unknown Title')
            query = paper.get('query', 'General Concept')
            journal = paper.get('journal_name', '')
            doi = paper.get('doi', '')
            
            # (Category) -[INCLUDES_PAPER]-> (Paper)
            G.add_node(query, type="Category")
            G.add_node(title, type="Paper")
            
            G.add_edge(core_topic, query, label="HAS_SUBTOPIC")
            G.add_edge(query, title, label="INCLUDES_PAPER")
            
            # (Paper) -[PUBLISHED_IN]-> (Journal)
            if journal:
                G.add_node(journal, type="Journal")
                G.add_edge(title, journal, label="PUBLISHED_IN")
            
            # (Paper) -[HAS_DOI]-> (DOI)
            if doi:
                G.add_node(doi, type="DOI")
                G.add_edge(title, doi, label="HAS_DOI")

        # Instead of matplotlib, use Pyvis for an interactive HTML graph
        from pyvis.network import Network
        
        # Create a pyvis network and load the networkx graph
        # notebook=False ensures it produces a standalone HTML file
        output_path_html = output_path.replace(".png", ".html")
        net = Network(height="800px", width="100%", bgcolor="#222222", font_color="white", directed=True)
        
        # We need to map our custom hex colors to Pyvis attributes
        for n, data in G.nodes(data=True):
            node_type = data.get('type', '')
            color = '#cccccc'
            size = 15
            
            if node_type == "CoreTopic":
                color = '#ff9999'
                size = 35
            elif node_type == "Category":
                color = '#ffcc99'
                size = 25
            elif node_type == "Paper":
                color = '#99ccff'
                size = 20
            elif node_type == "Journal":
                color = '#99ff99'
                size = 15
            elif node_type == "DOI":
                color = '#ffff99'
                size = 10
                
            # Pyvis node configuration
            data['color'] = color
            data['size'] = size
            data['title'] = str(n) # tooltip
            
            # Shorten labels for aesthetic
            label_text = str(n)
            data['label'] = label_text[:35] + "..." if len(label_text) > 38 else label_text

        # Inherit all configured nodes and edges into pyvis
        net.from_nx(G)
        
        # Add physics for interactive bouncing
        net.toggle_physics(True)
        
        # Save output to HTML
        net.save_graph(output_path_html)

        html_iframe_notice = f"Successfully generated interactive HTML Knowledge Graph with {G.number_of_nodes()} nodes. You can view the full interactive file at {output_path_html} in your browser!"
        
        return html_iframe_notice

    except Exception as e:
        logger.error(f"Failed to generate knowledge graph: {e}")
        return f"Error generation knowledge graph: {str(e)}"
