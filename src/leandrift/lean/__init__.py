"""Lean verification backends: real REPL pool and a semantic mock for local dev."""

from leandrift.lean.backend import LeanBackend, LeanResult, get_backend

__all__ = ["LeanBackend", "LeanResult", "get_backend"]
