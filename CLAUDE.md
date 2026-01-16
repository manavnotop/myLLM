# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python project using modern pyproject.toml-based packaging with Python 3.13+.

## Development Commands

```bash
# Install dependencies
pip install -e .

# Run the application
python main.py

# Run with uv (if using uv for package management)
uv run python main.py
```

## Project Structure

- `main.py` - Entry point with `main()` function
- `pyproject.toml` - Project configuration (PEP 517/518 standard)
- `.python-version` - Specifies Python 3.13
