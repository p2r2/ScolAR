# fetcher.py
import os
import aiohttp
import asyncio
import logging
from typing import List, Dict, Any
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

logger = logging.getLogger(__name__)

CONTACT_EMAIL = os.environ.get("CONTACT_EMAIL", "system@example.com")
CROSSREF_API_URL = os.environ.get("CROSSREF_API_URL", "https://api.crossref.org/works")

# =====================================================================
# INTERNAL LOGIC (Do NOT pass this to the LLM)
# Notice the underscore prefix to mark it as internal.
# =====================================================================
@retry(
    wait=wait_exponential(multiplier=1, min=2, max=10),
    stop=stop_after_attempt(3),
    retry=retry_if_exception_type(aiohttp.ClientError)
)
async def _internal_fetch_papers(session: aiohttp.ClientSession, query: str, limit: int = 20) -> List[Dict[str, Any]]:
    url = CROSSREF_API_URL
    params = {
        "query": query,
        "rows": limit,
        "mailto": CONTACT_EMAIL
    }
    
    logger.info(f"Fetching papers for query: '{query}'")
    
    try:
        async with session.get(url, params=params) as response:
            response.raise_for_status()
            data = await response.json()
            
            items = data.get("message", {}).get("items", [])
            extracted = []
            
            for item in items:
                title_list = item.get("title", [])
                title = title_list[0] if title_list else "Unknown Title"
                
                abstract = item.get("abstract", "")
                
                year = None
                pub_date = item.get("published-print") or item.get("published-online")
                if pub_date and "date-parts" in pub_date:
                    year = pub_date["date-parts"][0][0]
                    
                authors = []
                for author in item.get("author", []):
                    fam = author.get("family", "")
                    given = author.get("given", "")
                    authors.append(f"{given} {fam}".strip())
                    
                doi = item.get("DOI", "")
                
                subjects = item.get("subject", [])
                keywords = ", ".join(subjects) if subjects else ""
                
                container_titles = item.get("container-title", [])
                journal_name = container_titles[0] if container_titles else ""
                
                extracted.append({
                    "title": title,
                    "abstract": abstract,
                    "year": year,
                    "authors": authors,
                    "doi": doi,
                    "keywords": keywords,
                    "journal_name": journal_name,
                    "query": query 
                })
                
            return extracted
    except Exception as e:
        logger.error(f"Failed to fetch papers for '{query}': {e}")
        raise 


async def fetch_all_queries(queries: List[str], limit_per_query: int = 25) -> Dict[str, List[Dict[str, Any]]]:
    """
    Fetches papers for all queries concurrently. 
    (Usually used for backend processing, not directly by the LLM).
    """
    async with aiohttp.ClientSession() as session:
        tasks = [_internal_fetch_papers(session, q, limit_per_query) for q in queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        query_results = {}
        for q, res in zip(queries, results):
            if isinstance(res, Exception):
                logger.error(f"Error fetching for query '{q}': {res}")
                query_results[q] = []
            else:
                query_results[q] = res
                
        return query_results


# =====================================================================
# LLM TOOL FUNCTION (Pass ONLY this to your LLM)
# =====================================================================
from google.adk.tools.tool_context import ToolContext

def fetch_academic_papers(query: str, limit: int = 20, tool_context: ToolContext = None) -> str:
    """
    Searches for academic papers on Crossref based on a query.
    
    Args:
        query: The search topic or keywords.
        limit: The maximum number of papers to return.
    """
    import asyncio
    import threading

    async def _run():
        async with aiohttp.ClientSession() as session:
            return await _internal_fetch_papers(session, query, limit)
            
    # Use a separate thread to run the asyncio loop so it doesn't conflict
    # if the adk framework is already running inside an event loop (e.g. uvicorn).
    result = None
    exception = None

    def _thread_worker():
        nonlocal result, exception
        try:
            result = asyncio.run(_run())
        except Exception as e:
            exception = e

    thread = threading.Thread(target=_thread_worker)
    thread.start()
    thread.join()

    if exception:
        raise exception

    # Save to state so the next agent (graph_agent) can access it!
    if tool_context is not None:
        if "fetched_papers" not in tool_context.state:
            tool_context.state["fetched_papers"] = []
        if isinstance(result, list):
            tool_context.state["fetched_papers"].extend(result)

    # LLMs prefer stringified outputs over native Python lists of dicts
    return str(result)