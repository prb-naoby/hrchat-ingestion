import os
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

# Set environment variables for testing
os.environ["QDRANT_URL"] = "http://localhost:6333"
os.environ["QDRANT_API_KEY"] = "test-key"
os.environ["QDRANT_COLLECTION"] = "test-collection"
os.environ["GOOGLE_API_KEY"] = "test-google-key"

@pytest.fixture
def mock_qdrant_client():
    with patch("src.api.qdrant_client") as mock:
        client_instance = MagicMock()
        mock.return_value = client_instance
        yield client_instance

@pytest.fixture
def mock_pipeline_process_file():
    with patch("src.api.process_file") as mock:
        yield mock

@pytest.fixture
def client():
    from src.api import app
    return TestClient(app)

@pytest.fixture
def mock_doc_parser():
    with patch("src.parser_utils.DocParser") as mock:
        yield mock

@pytest.fixture
def mock_genai_embedding():
    with patch("src.embed.get_genai_embedding") as mock:
        mock.return_value = [0.1] * 768
        yield mock
