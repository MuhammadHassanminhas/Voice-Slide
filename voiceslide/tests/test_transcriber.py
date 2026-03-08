import sys
import os
import pytest
from unittest.mock import patch, MagicMock

# Add backend to sys.path to resolve imports in transcriber.py
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend')))

from transcriber import transcribe_chunk, get_model

def test_transcribe_chunk_valid_audio():
    # Read the raw float32 PCM fixture
    fixture_path = os.path.join(os.path.dirname(__file__), 'fixtures', 'sample_audio.raw')
    with open(fixture_path, 'rb') as f:
        audio_bytes = f.read()

    # Mock the WhisperModel to avoid requiring actual model inference during unit testing
    mock_model = MagicMock()
    
    # Create mock segments to return
    mock_segment = MagicMock()
    mock_segment.text = "Hello world."
    
    mock_info = MagicMock()
    mock_info.language = "en"
    
    # model.transcribe() returns an iterable of segments and an info object
    mock_model.transcribe.return_value = ([mock_segment], mock_info)

    # Patch get_model to return our mocked WhisperModel
    with patch('transcriber.get_model', return_value=mock_model):
        
        # Test the function with valid audio
        # Note: We must also patch numpy if it hasn't been installed, but assuming it exists
        # in the CI/test environment.
        with patch('transcriber.np.frombuffer', return_value=[0.0, 0.0]):
            result = transcribe_chunk(audio_bytes)
            
            assert isinstance(result, dict)
            assert result.get('text') == "Hello world."
            assert result.get('is_final') is True
            assert result.get('language') == "en"
            
            # Ensure model.transcribe was called
            mock_model.transcribe.assert_called_once()

def test_transcribe_chunk_empty_audio():
    # Test with empty bytes
    result = transcribe_chunk(b"")
    
    assert isinstance(result, dict)
    assert result.get('text') == ""
    assert result.get('is_final') is True
    assert result.get('language') == "en"
