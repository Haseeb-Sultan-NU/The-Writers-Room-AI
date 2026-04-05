import json
from typing import Callable, Dict, Any, List
from pydantic import BaseModel, Field

class ToolRegistry:
    """
    Simulates the Model Context Protocol (MCP) tool discovery mechanism.
    Agents query this registry at runtime to find and execute tools dynamically.
    """
    def __init__(self):
        self._tools: Dict[str, Dict[str, Any]] = {}

    def register_tool(self, name: str, description: str, schema: type[BaseModel], func: Callable):
        """Registers a tool along with its Pydantic schema for dynamic discovery."""
        # We extract the JSON schema from the Pydantic model
        self._tools[name] = {
            "description": description,
            "schema": schema.model_json_schema(),
            "func": func
        }

    def discover_tools(self) -> List[Dict[str, Any]]:
        """
        Agents call this to find out what they can do. 
        Returns a list of available tools and their required JSON schemas.
        """
        return [{"name": name, "description": data["description"], "schema": data["schema"]} 
                for name, data in self._tools.items()]

    def execute_tool(self, tool_name: str, **kwargs) -> Any:
        """Executes a tool dynamically by name, preventing hardcoded API calls."""
        if tool_name not in self._tools:
            raise ValueError(f"Tool '{tool_name}' not found in MCP registry.")
        return self._tools[tool_name]["func"](**kwargs)

# =====================================================================
# TOOL SCHEMAS (Defining the structured JSON inputs agents must provide)
# =====================================================================

class GenerateScriptInput(BaseModel):
    prompt: str = Field(..., description="The user's prompt or story idea.")
    num_scenes: int = Field(default=5, description="Number of scenes to generate.")

class CommitMemoryInput(BaseModel):
    collection_name: str = Field(..., description="The ChromaDB collection to save to (e.g., 'script_history').")
    data: Dict[str, Any] = Field(..., description="The JSON data or metadata to save.")

# =====================================================================
# TOOL IMPLEMENTATIONS (Dummy functions for now, we will connect LLMs later)
# =====================================================================

def generate_script_segment(prompt: str, num_scenes: int = 5) -> Dict[str, Any]:
    """Placeholder logic for the Scriptwriter Agent."""
    return {"status": "success", "message": f"Generated {num_scenes} scenes for: {prompt}"}

def commit_memory(collection_name: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """Placeholder logic for interacting with ChromaDB."""
    return {"status": "success", "message": f"Saved agent data to {collection_name}"}

# =====================================================================
# INITIALIZE & REGISTER
# =====================================================================

# Create the global registry instance
registry = ToolRegistry()

# Dynamically register the tools
registry.register_tool(
    name="generate_script_segment", 
    description="Generates a multi-scene screenplay segment based on user intent.", 
    schema=GenerateScriptInput, 
    func=generate_script_segment
)

registry.register_tool(
    name="commit_memory", 
    description="Saves state or metadata to the persistent Vector DB.", 
    schema=CommitMemoryInput, 
    func=commit_memory
)

if __name__ == "__main__":
    # Test 1: Can we discover tools dynamically?
    print("--- MCP TOOL DISCOVERY ---")
    available_tools = registry.discover_tools()
    print(json.dumps(available_tools, indent=2))
    
    print("\n--- MCP TOOL EXECUTION ---")
    # Test 2: Can we execute a tool without hardcoding the function call?
    # This simulates an agent sending a JSON payload.
    agent_json_payload = {
        "tool": "generate_script_segment",
        "input": {
            "prompt": "A cyberpunk detective investigates a neon city",
            "num_scenes": 3
        }
    }
    
    result = registry.execute_tool(agent_json_payload["tool"], **agent_json_payload["input"])
    print(f"Agent Execution Result: {result}")