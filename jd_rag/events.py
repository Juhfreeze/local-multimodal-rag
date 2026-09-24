from typing import Callable

ProgressCallback = Callable[[str], None]

def ignore_progress(message: str) -> None:
    """Default for callers that do not need progress messages."""
