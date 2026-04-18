from typing import Annotated, TypedDict, List, Dict, Any
import operator

# --- PHASE 2: Parallel Task State ---
# This state is passed ONLY to the parallel worker nodes for a specific scene
class SceneTaskState(TypedDict):
    scene_id: int
    scene_data: Dict[str, Any]
    character_profiles: List[Dict[str, Any]]

# --- MAIN GRAPH STATE ---
class GraphState(TypedDict):
    # Phase 1 Artifacts
    input_mode: str
    raw_prompt: str
    script: Dict[str, Any]
    characters: List[Dict[str, Any]]
    images: List[str]
    
    # Phase 2 Artifacts (Using operator.add to safely merge parallel outputs)
    manifest_path: str
    audio_tracks: Annotated[List[Dict[str, Any]], operator.add]
    base_videos: Annotated[List[Dict[str, Any]], operator.add]
    swapped_videos: Annotated[List[Dict[str, Any]], operator.add]
    final_scenes: Annotated[List[Dict[str, Any]], operator.add]
    
    errors: Annotated[List[str], operator.add]
    status: str