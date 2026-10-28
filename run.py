"""Run locally: python run.py. No API keys or external inference service."""
import os

# Keep CPU usage predictable on laptops. Set before importing numerical libraries.
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

if __name__ == "__main__":
    import uvicorn
    print("\nMozhi lab: http://127.0.0.1:8000\nStop with Ctrl+C.\n")
    uvicorn.run("toxicity.server:app", host="127.0.0.1", port=8000, log_level="info")
