"""Shared fixtures for AI service tests."""

import os
import sys

import pytest

# Add service dir to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
