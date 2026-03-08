import struct
import os

def generate_pcm_fixture(filepath: str, duration_sec: int = 1, sample_rate: int = 16000):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "wb") as f:
        # Write 0.0f (4 bytes) sample_rate * duration_sec times
        for _ in range(int(sample_rate * duration_sec)):
            f.write(struct.pack('f', 0.0))
    print(f"Fixture created at {filepath}")

if __name__ == "__main__":
    generate_pcm_fixture("tests/fixtures/sample_audio.raw")
