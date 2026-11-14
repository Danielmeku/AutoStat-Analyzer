from typing import Protocol


class StatisticalProvider(Protocol):
    def respond(self, question: str, context: dict) -> dict:
        """Return qualitative prose and evidence identifiers, never perform calculations."""
        ...
