from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):

    ## Apis

    PINECONE_API_KEY: str = Field(default="", description="Pinecone API key")
    PINECONE_ENVIRONMENT: str = Field(default="us-east-1", description="Pinecone environment region")
    PINECONE_INDEX_NAME: str = Field(default="visionrag", description="Pinecone index name")


    GROQ_API_KEY: str = Field(default="", description="GROQ API key")

    JINA_API_KEY: str = Field(default="", description="Jina API Key")

    S3_ENDPOINT: str = Field(default="https://s3.amazonaws.com", description="S3 Endpoint URL")
    S3_BUCKET: str = Field(default="visionrag-bucket", description="S3 Bucket Name")
    S3_ACCESS_KEY: str = Field(default="", description="S3 Access Key")
    S3_SECRET_KEY: str = Field(default="", description="S3 Secret Key")
    S3_REGION: str = Field(default="us-east-2", description="S3 Region")

    class Config:
        env_file = ".env"
        extra = "allow"




settings = Settings()