import os
import chromadb

# Dynamically find the project root (one folder up from the 'src' directory)
# This ensures our database is always created in the main project folder.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_data")

def initialize_memory():
    """
    Initializes a local ChromaDB instance to store script history and character metadata.
    Data is saved to the 'chroma_data' directory in the project root.
    """
    # Create the storage directory if it doesn't exist
    os.makedirs(CHROMA_PATH, exist_ok=True)
    
    # Initialize the client with persistent storage
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    
    # Create or get collections (tables in vector DB terms)
    script_collection = client.get_or_create_collection(name="script_history")
    character_collection = client.get_or_create_collection(name="character_metadata")
    
    print(f"Memory Vector Database initialized successfully at: {CHROMA_PATH}")
    return script_collection, character_collection

if __name__ == "__main__":
    # Run this file directly to test the initialization
    initialize_memory()