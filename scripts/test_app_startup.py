"""
Quick test script to verify FastAPI app starts correctly.
Run this before implementing endpoints.
"""

# ruff : noqa : E402

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import settings
from src.main import app


def test_app_startup():
    """Test that the app initializes without errors."""
    print("✓ FastAPI app imported successfully")
    print(f"✓ App title: {app.title}")
    print(f"✓ App version: {app.version}")
    print(f"✓ Database URL configured: {settings.database_url[:20]}...")
    print(f"✓ CORS origins: {settings.cors_origins_list}")
    print(f"✓ Dataset path: {settings.dataset_path}")
    print(f"✓ Models path: {settings.models_path}")
    print("\n✅ All basic checks passed!")


if __name__ == "__main__":
    test_app_startup()
