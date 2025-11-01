# config_example.py — copy to config.py and fill with real values.
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password"

NVIDIA_API_KEY = "nvapi-..."   # Your NVIDIA API key

PINECONE_API_KEY = "pcsk_..." # your Pinecone API key
PINECONE_ENV = "us-east-1"   # example
PINECONE_INDEX_NAME = "vietnam-travel"
PINECONE_VECTOR_DIM = 2048       # adjust to embedding model used (we are using "nvidia/llama-3.2-nemoretriever-300m-embed-v2" which has 2048 dimentions) -change if needed.
