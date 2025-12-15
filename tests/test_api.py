from fastapi import status

def test_healthz(client):
    response = client.get("/healthz")
    assert response.status_code == 200
    data = response.json()
    assert "python" in data
    assert data["api_key_present"] is True  # Set in conftest

def test_process_endpoint_async(client, mock_pipeline_process_file, mock_qdrant_client):
    # Mock Qdrant checks to simulate "not found" so it proceeds to processing
    mock_qdrant_client.scroll.return_value = ([], None)
    
    files = {"file": ("test.pdf", b"dummy content", "application/pdf")}
    data = {"category": "test-cat", "collection": "test-col"}
    
    response = client.post("/process?async_mode=1", files=files, data=data)
    assert response.status_code == status.HTTP_202_ACCEPTED
    assert response.json()["status"] == "accepted"

def test_process_endpoint_sync(client, mock_pipeline_process_file, mock_qdrant_client):
    # Mock Qdrant checks
    mock_qdrant_client.scroll.return_value = ([], None)
    
    # Mock pipeline return
    mock_pipeline_process_file.return_value = {
        "file_name": "test.pdf",
        "status": "completed"
    }
    
    files = {"file": ("test.pdf", b"dummy content", "application/pdf")}
    data = {"category": "test-cat", "collection": "test-col"}
    
    response = client.post("/process?async_mode=0", files=files, data=data)
    assert response.status_code == 200
    assert response.json()["file_name"] == "test.pdf"

def test_filenames_endpoint(client, mock_qdrant_client):
    # Mock scroll response
    from qdrant_client.models import Record
    
    mock_record = Record(
        id="1",
        payload={"metadata": {"filename": "doc1.pdf"}},
        vector=None
    )
    mock_qdrant_client.scroll.side_effect = [([mock_record], None)]
    
    response = client.get("/filenames?category=test")
    assert response.status_code == 200
    data = response.json()
    assert "doc1.pdf" in data["filenames"]
