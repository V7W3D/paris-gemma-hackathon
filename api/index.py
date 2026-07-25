"""Vercel Python entrypoint: re-export the FastAPI app.

The package lives at the repository root, so the import path matches local
`uvicorn backend.main:app` once the project root is on PYTHONPATH (Vercel
puts it there for this layout).
"""

from backend.main import app

__all__ = ["app"]
