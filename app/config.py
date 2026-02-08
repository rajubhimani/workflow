from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")

    OLLAMA_MODEL: str = "llama3.2"
    DEBUG: bool = False
    MAX_TOKENS: int = 1024

    # Embedding
    OLLAMA_EMBEDDING_MODEL: str = "nomic-embed-text"

    # Vector store
    VECTOR_STORE_BACKEND: str = "chroma"
    CHROMA_PERSIST_DIR: str = "./chroma_data"

    # Chunking
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200

    # Retrieval
    RAG_TOP_K: int = 4

    # File uploads
    UPLOAD_DIR: str = "./uploads"


settings = Settings()
