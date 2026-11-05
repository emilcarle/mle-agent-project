""" The operations data layer provides a local backend for the agent's tools, using JSON fixtures and Markdown documents. """


from typing import Any
from pathlib import Path
from pydantic_ai import ModelRetry
from my_solution.knowledge_base import embed_texts
from my_solution.config import TOP_K


class OperationsData:
    """Local data access layer that acts like a small tool backend."""

    def __init__(self, 
                 tickets: list[dict[str, Any]],
                 customers: list[dict[str, Any]],
                 availability: list[dict[str, Any]],
                 memory: dict[str, Any],
                 chunks: list[dict[str, Any]],
                 threshold: float
                 ):
        
        self.tickets = tickets
        self.customers = customers
        self.availability = availability
        self.memory = memory
        self.document_chunks = chunks
        self.threshold = threshold
        self.top_k = TOP_K

    def get_ticket(self, ticket_id: str) -> dict[str, Any]:
        """Return the ticket details from a given ticket_id."""

        for ticket in self.tickets:
            if ticket["ticket_id"] == ticket_id:
                return ticket
        raise ModelRetry(
            f"Unknown ticket_id: {ticket_id}. Use an ID from the request or history, not a guess."
        )
        
    def get_customer_context(self, customer_id: str) -> dict[str, Any]:
        """Return the customer details from a given customer_id."""

        for customer in self.customers:
            if customer["customer_id"] == customer_id:
                return customer
        raise ModelRetry(
            f"Unknown customer_id: {customer_id}. Use an ID from the request or history, not a guess."
        )
        
    def search_policy(self, query: str) -> list[dict[str, Any]]:
        """Return the top_k most relevant document chunks for a given ticket text body."""

        query_embedding = embed_texts(query)[0]
        for chunk in self.document_chunks:
            chunk["similarity"] = float(
                sum(a * b for a, b in zip(query_embedding, chunk["vector"]))
            )
        if not any(chunk["similarity"] > self.threshold for chunk in self.document_chunks):
            raise ModelRetry(
                f"No relevant policy documents found for query: {query}."
            )
        selected_chunks = [chunk for chunk in sorted(self.document_chunks, key=lambda x: x["similarity"], reverse=True)[:self.top_k] if chunk["similarity"] > self.threshold]
        return [
            {
                key: value for key, value in chunk.items() if key != "vector"}
                for chunk in selected_chunks
            ]

        
    def check_availability(self, queue: str) -> dict[str, Any]:
        """Find the responsible owner and their details for a handoff based on the queue."""

        for item in self.availability:
            if item["queue"] == queue:
                return item
        raise ModelRetry(
            f"Unknown queue: {queue}. Use a queue from the availability list, not a guess."
        )

    def draft_ticket_update(
        self,
        ticket_id: str,
        customer_name: str,
        citations: list[str],
    ) -> str:
        """Draft a customer-visible update without writing to a real helpdesk."""

        citation_text = ", ".join(f"[{citation}]" for citation in citations)
        return (
            f"Draft update for {customer_name} on {ticket_id}: we can proceed "
            f"with the documented next step based on {citation_text}."
        )

    def draft_escalation(
        self,
        ticket_id: str,
        queue: str,
        owner: str,
        citations: list[str],
    ) -> str:
        """Draft an internal escalation for a human queue."""

        availability = next(
            (item for item in self.availability if item["queue"] == queue), None
        )
        if not availability or not availability["available"]:
            raise RuntimeError(f"No available owner for {queue}; request review.")
        if owner != availability["owner"]:
            raise RuntimeError(
                f"The proposed owner does not match {queue}; request review."
            )
        citation_text = ", ".join(f"[{citation}]" for citation in citations)
        return (
            f"Escalate {ticket_id} to {queue} ({owner}) with evidence {citation_text}."
        )
        
    def access_memory(self, customer_id: str) -> dict[str, Any]:
        """Extracts a memory-log of a conversation based on a costumer_id."""
        if customer_id != self.memory['last_customer_id']:
            raise RuntimeError(
                f"No cases relating to customer_id {customer_id} found in memory."
            )
        return self.memory