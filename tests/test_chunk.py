from src.chunk import chunk_markdown

def test_chunk_markdown_simple():
    text = "# Heading\n\nSome content here."
    chunks = chunk_markdown(text, max_chars=100)
    assert len(chunks) == 1
    assert "Some content here" in chunks[0]

def test_chunk_markdown_split():
    # Create text larger than max_chars to force split
    text = "# Heading 1\n\n" + "a" * 60 + "\n\n# Heading 2\n\n" + "b" * 60
    chunks = chunk_markdown(text, max_chars=80)
    assert len(chunks) >= 2
    assert "Heading 1" in chunks[0]
    assert "Heading 2" in chunks[1]

def test_chunk_markdown_heading_only():
    text = "# Heading Only"
    chunks = chunk_markdown(text)
    assert len(chunks) == 1
    assert chunks[0] == "# Heading Only"

def test_chunk_markdown_preface():
    text = "Preface content.\n\n# Heading 1\n\nBody 1."
    chunks = chunk_markdown(text, max_chars=1000)
    # Depending on logic, preface might be its own block or merged if small enough
    # The current logic treats preface as a block.
    # Since max_chars is large, they should be merged into one chunk or kept separate blocks in one chunk.
    assert len(chunks) == 1
    assert "Preface content" in chunks[0]
    assert "Heading 1" in chunks[0]
