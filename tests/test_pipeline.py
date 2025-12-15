import pytest
from unittest.mock import MagicMock, patch
from src.pipeline import process_file

@patch("src.pipeline.parse_document")
@patch("src.pipeline.chunk_markdown")
@patch("src.pipeline.qdrant_client")
@patch("src.pipeline.get_genai_embedding")
def test_process_file_flow(
    mock_get_embedding,
    mock_qdrant,
    mock_chunk,
    mock_parse
):
    # Setup mocks
    mock_parse.return_value = ("# Markdown", {"page_count": 1})
    mock_chunk.return_value = ["Chunk 1", "Chunk 2"]
    mock_get_embedding.return_value = [0.1] * 10
    
    mock_client = MagicMock()
    mock_qdrant.return_value = mock_client
    
    result = process_file(
        data=b"pdf-content",
        filename="test.pdf",
        ext=".pdf",
        category="test",
        collection="col",
        job_key="job1"
    )
    
    assert result["file_name"] == "test.pdf"
    assert result["uploaded_chunks"] == 2
    
    # Verify calls
    mock_parse.assert_called_once()
    mock_chunk.assert_called_once()
    assert mock_get_embedding.call_count >= 2  # At least once for marker, once for each chunk
    mock_client.upsert.assert_called()
