from pathlib import Path

BASE_DIR = Path(__file__).parent

OLLAMA_MODEL = "gemma2:2b"
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_TIMEOUT = 30
OLLAMA_RETRIES = 2

CHROMA_DB_PATH = str(BASE_DIR / "chroma_db")
CHROMA_COLLECTION_NAME = "meddra_terms"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

CONFIDENCE_THRESHOLD = 0.75
MAX_NARRATIVE_WORDS = 400
TOP_K_CANDIDATES = 5

FAERS_API_URL = "https://api.fda.gov/drug/event.json"
FAERS_SAMPLE_SIZE = 300

DATA_DIR = BASE_DIR / "data"
FAERS_SAMPLES_PATH = DATA_DIR / "faers_samples.json"
MEDDRA_TERMS_PATH = DATA_DIR / "meddra_terms.json"
TEST_NARRATIVES_PATH = DATA_DIR / "test_narratives.json"
DEMO_RESULTS_PATH = DATA_DIR / "demo_results.json"
