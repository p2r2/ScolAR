from google.adk.agents.llm_agent import Agent

from scolAR.tools.fetcher_tools import fetch_academic_papers


fetcher_agent = Agent(
    model="gemini-3-flash-preview",
    name="fetcher_agent",
    description="A research director that reads the topic and formulates highly specific orthogonal search queries.",
    instruction="""
    An agent responsible for concurrently fetching and parsing academic paper metadata 
    from the Crossref API.

    This class encapsulates the logic for querying the Crossref API, parsing the JSON 
    responses, handling transient network errors with exponential backoff, and 
    executing multiple search queries concurrently.

    Attributes:
        contact_email (str): The email address to provide to Crossref for the 'mailto' 
                             parameter (encouraged by Crossref for the "polite pool").
        api_url (str): The base URL for the Crossref API endpoint.

    Methods:
        fetch_papers_for_query(session, query, limit=20):
            Asynchronously fetches and parses a list of papers for a single search query.
            Includes automatic retry logic (up to 3 attempts) for `aiohttp.ClientError`.
            
            Args:
                session (aiohttp.ClientSession): The active asynchronous HTTP session.
                query (str): The search string to query the Crossref API.
                limit (int): The maximum number of results to fetch per query.
                
            Returns:
                List[Dict[str, Any]]: A list of dictionaries representing parsed papers. 
                Keys include: 'title', 'abstract', 'year', 'authors', 'doi', 'keywords', 
                'journal_name', and 'query'.

        fetch_all_queries(queries, limit_per_query=25):
            Concurrently executes `fetch_papers_for_query` for a list of search queries.
            Initializes its own `aiohttp.ClientSession`. Gracefully handles individual 
            query failures by returning an empty list for that specific query rather 
            than failing the entire batch.
            
            Args:
                queries (List[str]): A list of search query strings.
                limit_per_query (int): The maximum number of papers to fetch per query.
                
            Returns:
                Dict[str, List[Dict[str, Any]]]: A dictionary where the keys are the 
                original search queries and the values are lists containing the 
                extracted paper metadata dictionaries.
    """,
    tools=[fetch_academic_papers],
)
