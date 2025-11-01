#hybrid_chat_langgraph.py
import json
import asyncio
from typing import TypedDict, List, Dict, Any
from functools import lru_cache
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from neo4j import AsyncGraphDatabase, GraphDatabase
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate

import config


# -----------------------------
# Config
# -----------------------------
EMBED_MODEL = "nvidia/llama-3.2-nemoretriever-300m-embed-v2"
CHAT_MODEL = "meta/llama-3.3-70b-instruct"
TOP_K = 5

INDEX_NAME = config.PINECONE_INDEX_NAME


# -----------------------------
# Initialize clients
# -----------------------------
client = OpenAI(
  api_key=config.NVIDIA_API_KEY,
  base_url="https://integrate.api.nvidia.com/v1"
)

pc = Pinecone(api_key=config.PINECONE_API_KEY)

# Connect to Pinecone index
if INDEX_NAME not in pc.list_indexes().names():
    print(f"Creating managed index: {INDEX_NAME}")
    pc.create_index(
        name=INDEX_NAME,
        dimension=config.PINECONE_VECTOR_DIM,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(INDEX_NAME)

# Connect to Neo4j
driver = AsyncGraphDatabase.driver(
    config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASSWORD)
)


### ===== CACHED EMBEDDINGS =====

@lru_cache(maxsize=128)   # IMPROVEMENT 1: Caching embeddings
def embed_text(text: str) -> List[float]:
    """Get embedding for a text string with caching."""
    resp = client.embeddings.create(
        model=EMBED_MODEL, 
        input=[text],
        extra_body={"input_type": "query", "truncate": "NONE"})   # for the type of embedding model

    return resp.data[0].embedding


### ===== LANGGRAPH AGENT STATE AND NODES =====

# Defining the State for our graph
class RAGState(TypedDict):
    query: str
    vector_matches: List[Dict]
    summary: str
    graph_facts: List[Dict]
    generation: str


# Node 1: Retrive from Pinecone
def retrieve_vector_context(state: RAGState) -> RAGState:
    """Query Pinecone for semantic matches."""
    print("--- Step 1: Retrieving vector context ---")
    vec = embed_text(state['query'])
    res = index.query(
        vector=vec,
        top_k=TOP_K,
        include_metadata=True,
        include_values=False
    )
    print(f"DEBUG: Found {len(res['matches'])} vector matches.")
    return {"vector_matches": res['matches']}


# Node 2: Summarizing
def summarize_retrieved_context(state: RAGState) -> Dict[str, str]:
    """Summarize the initial retrieved nodes to distill the key themes."""
    print("--- Step 2: Summarizing retrieved context  ---")

    summary_prompt=f"""
    Based on the following search results, please provide a very brief, one paragraph summary.
    Identify the main city, theme, or topic that seems most relevant to the user's original query: "{state['query']}".

    Search Results:
    {[match['metadata']['name'] for match in state['vector_matches']]}
    """

    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{'role': 'user', 'content': summary_prompt}],
        max_tokens=150,
        temperature=0.1
    )
    summary = resp.choices[0].message.content
    print("DEBUG: Generate summary of top nodes.")
    return {'summary': summary}


# Node 3: Retrieve from Neo4j
async def retrieve_graph_context(state: RAGState) -> Dict[str, Any]:
    """Fetch neighbouring nodes from Neo4j based on vector matches."""
    print("---Step 3: Retrieving graph context ---")
    node_ids = [m['id'] for m in state['vector_matches']]
    
    async def fetch_facts_for_id(nid):
        async with driver.session() as session:
            q = (
                "MATCH (n: Entity {id:$nid})-[r]-(m:Entity) "
                "RETURN type(r) AS rel, labels(m) AS labels, m.id AS id, "
                "m.name AS name, m.type AS type, m.description AS description "
                "LIMIT 5"   # Limit to 5 neighbours per node to keep context concise
            )
            result = await session.run(q, nid=nid)
            records = await result.data()
            return [
                {
                    "source": nid, "rel": r["rel"], "target_id": r["id"],
                    "target_name": r["name"],
                    "target_desc": (r["description"] or "")[:200]
                } for r in records
            ]
        
    # Create a list of async tasks and run them all with asyncio.gather
    tasks = [fetch_facts_for_id(nid) for nid in node_ids]
    results_list = await asyncio.gather(*tasks)

    # Flatten the list of lists into a single list of facts
    facts = [item for sublist in results_list for item in sublist]

    print(f"DEBUG: Found {len(facts)} graph facts.")
    return {"graph_facts": facts}


# Node 4: Generate the final response
def generate_response(state: RAGState) -> RAGState:
    """Generate a response using LLM with both contexts."""
    print("---Step 4: Generate final response ---")

    # IMPROVEMENT 2: Chain-of Thought Prompt
    system_prompt = """
    You are an expert travel assistant. Your goal is to create a helpful and concise travel itinerary.
    Follow these steps to generate your response:
    1.  **Analyze the Goal:** First, understand the user's core travel request from their query.
    2.  **Synthesize Context:** Review the 'Semantic Matches' from the vector database and the 'Connected Facts' from the graph database. Identify the key locations and attractions.
    3.  **Construct the Itinerary:** Based on your analysis, create a clear, step-by-step itinerary or a list of recommendations.
    4.  **Cite Sources:** Crucially, you MUST cite the node ids (e.g., attraction_123, city_hanoi) in parentheses after mentioning any place. This is a strict requirement.
    5.  **Add Tips:** Conclude with 2-3 practical tips for the traveler.
    """

    vec_context_str = "\n".join([f"- id: {m['id']}, name: {m['metadata'].get('name', '')}, type: {m['metadata'].get('type', '')}" for m in state['vector_matches']])
    graph_context_str = "\n".join([f"- ({f['source']}) is {f['rel']} '{f['target_name']}' ({f['target_id']})" for f in state['graph_facts']])

    user_prompt = f"""
    User Query: {state['query']}

    An initial analysis of the most relevant places produced this summary:
    "{state['summary']}"

    Use this summary, along with the detailed context below, to construct your answer.

    Semantic Matches:
    {vec_context_str}

    Connected Facts:
    {graph_context_str}

    Now, generate the response using the system instructions.
    """

    prompt = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    resp = client.chat.completions.create(
        model = CHAT_MODEL,
        messages=prompt,
        max_tokens=600,
        temperature=0.3
    )
    generation = resp.choices[0].message.content
    return {"generation": generation}


# ===== MAIN CHAT APPLICATION =====

async def main():
    # Define the graph workflow
    workflow = StateGraph(RAGState)
    workflow.add_node("vector_retriever", retrieve_vector_context)
    workflow.add_node("summarizer", summarize_retrieved_context)
    workflow.add_node("graph_retriever", retrieve_graph_context)
    workflow.add_node("generator", generate_response)

    # Define the graph edges
    workflow.set_entry_point("vector_retriever")
    workflow.add_edge("vector_retriever", "summarizer")
    workflow.add_edge("summarizer", "graph_retriever")
    workflow.add_edge("graph_retriever", "generator")
    workflow.add_edge("generator", END)

    # Compile the graph into runnable app
    app = workflow.compile()

    print("Hybrid travel assistant (Async LangGraph Eddition). Type 'exit' to quit.")
    while True:
        query = input('\n Enter your travel question: ').strip()
        if not query or query.lower() in ("exit", "quit"):
            break
        
        # Create the initial input
        initial_input = {"query": query}

        # Run the graph
        final_state = await app.ainvoke(initial_input)

        print("\n ===== ASSISTANT ANSWER =====\n")
        print(final_state['generation'])
        print("\n ===== END ===== \n")

if __name__ == "__main__":
    asyncio.run(main())