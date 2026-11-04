import os


# Load all data from env
EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
EMBEDDING_MODEL_REVISION = os.environ.get("EMBEDDING_MODEL_REVISION", "main")


# Constants for LLM provider and model
TOP_K=3
MAX_AGENT_STEPS=8
MAX_TOOL_CALLS=6