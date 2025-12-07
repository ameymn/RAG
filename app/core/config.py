from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):

    ## Apis

    PINECONE_API_KEY: str = Field(default="", description="Pinecone API key")
    PINECONE_ENVIRONMENT: str = Field(default="us-east-1", description="Pinecone environment region")
    PINECONE_INDEX_NAME: str = Field(default="visionrag", description="Pinecone index name")


    GROQ_API_KEY: str = Field(default="", description="GROQ API key")

    JINA_API_KEY: str = Field(default="", description="Jina API Key")

    class Config:
        env_file = ".env"
        extra = "allow"




settings = Settings()