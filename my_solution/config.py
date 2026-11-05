import os


# Load all data from env
EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
EMBEDDING_MODEL_REVISION = os.environ.get("EMBEDDING_MODEL_REVISION", "main")
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "groq")
MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b")


# Constants for LLM provider and model
TOP_K=3
MAX_AGENT_STEPS=10
MAX_TOOL_CALLS=8
MAX_RETRIES=2


# Instructions
BASE_INSTRUCTIONS = (
            " You are an operations assistant that helps support agents handle customer tickets."
            " When given a prompt, start by gathering relevant information using the provided tools."
            " If the given prompt refers to a prior conversation, use access_memory() to retrieve information about the prior conversation."
            " Use get_ticket() to get details about the ticket in question."
            " Gather further on the customer by calling get_customer_context() and retrieve relevant policies by calling search_policy()."
            " The policies provide guidance on how to handle tickets and requests. Do not invent policies or make up information."
            " If the ticket requires escalation, check the availability of the responsible owner using check_availability()."
            " The relevant queue for escalation with check_availability() is mentioned in the relevant policies."
            " When you have gathered enough information, draft a response using draft_ticket_update() or draft_escalation()."
            " If you are unable to answer the request, you may abstain from answering by leaving the answer field empty and setting the status to 'abstained'."
)