import os
from dotenv import load_dotenv

# Load environment variables from .env file located in the parent directory
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(dotenv_path=dotenv_path)

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', os.getenv("MODEL_PATH", "models/isolation_forest_model.joblib"))

if NEO4J_PASSWORD is None:
    print("Warning: NEO4J_PASSWORD not found in .env file.") 