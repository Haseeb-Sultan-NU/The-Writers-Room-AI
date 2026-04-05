from typing import TypedDict, List, Dict, Any

class GraphState(TypedDict):
    """
    This defines the central state that gets passed from agent to agent.
    Every agent will read from this, do its job, and append its results back here.
    """
    input_mode: str          # "manual" or "auto"
    raw_prompt: str          # The user's initial idea or uploaded text
    script: Dict[str, Any]   # The structured scene_manifest.json data
    characters: List[Dict]   # The character_db.json data
    images: List[str]        # Paths to the generated images
    status: str              # "processing", "needs_review", "approved", or "failed"
    errors: List[str]        # Keep track of formatting or validation errors