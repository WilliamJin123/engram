# Technology Stack

**Analysis Date:** 2026-01-31

## Languages

**Primary:**
- Python 3.9+ - Primary language for all implementation
  - Tested on Python 3.10, 3.11, 3.12, 3.13, 3.14
  - Package: `engram` v0.1.0

## Runtime

**Environment:**
- Python 3.9 or higher (requires-python >= 3.9)
- CPython (standard Python interpreter)

**Package Manager:**
- uv - Fast Python package installer
- Lockfile: `uv.lock` (present)
- Build system: setuptools >= 61.0

## Frameworks

**Core:**
- PyTorch 2.0+ - Deep learning framework
  - Version 2.8.0 for Python < 3.10
  - Version 2.10.0 for Python >= 3.10
  - Purpose: Tensor operations, pattern representation, and numerical computation

**Testing:**
- pytest 7.0+ - Test runner and framework
  - Version 8.4.2 for Python < 3.10
  - Version 9.0.2 for Python >= 3.10
  - Config: `pytest.ini_options` in `pyproject.toml`
  - Test path: `tests/`

**Build/Dev:**
- setuptools - Python packaging
- No linting/formatting tools configured in dependencies (not enforced)

## Key Dependencies

**Critical:**
- torch >= 2.0.0 - Core computational framework for pattern operations
  - Used in `src/quantum_substrate/patterns.py` for tensor operations
  - Used in pattern generation, similarity computation, and pattern manipulation
  - Includes optional CUDA support via cuda-bindings (12.9.4) on Linux x86_64

**Infrastructure:**
- cuda-bindings 12.9.4 - CUDA toolkit support (optional, Linux only)
  - Conditional: `python_full_version >= '3.10' and platform_machine == 'x86_64' and sys_platform == 'linux'`
  - Provides GPU acceleration when available
- cuda-pathfinder 1.3.3 - CUDA environment detection
- filelock - File locking for thread-safe operations
- fsspec - Abstract filesystem for file operations
- networkx - Graph operations (imported by torch)
- jinja2 - Template engine (imported by torch)
- exceptiongroup - Exception handling (Python < 3.12 backport)
- colorama - Terminal color support (Windows optional)

## Configuration

**Environment:**
- Python path configuration: `conftest.py` adds `src/` to sys.path automatically
- No `.env` files detected in project
- No environment-specific configuration files

**Build:**
- `pyproject.toml`: Project metadata and dependencies
  - `[tool.pytest.ini_options]`: Test configuration
  - `[tool.setuptools.packages.find]`: Package discovery from `src/`

**Package Discovery:**
- setuptools configured to find packages in `src/` directory
- All modules under `src/agentic/` and `src/quantum_substrate/` automatically discovered

## Platform Requirements

**Development:**
- Operating System: Linux, macOS, Windows (Windows requires colorama for pytest output)
- Python: 3.9 through 3.14
- Virtual environment: Recommended (uses `.venv/`)

**Production:**
- Operating System: Linux (for CUDA support), macOS, Windows
- Python: 3.9 or higher
- GPU Support: Optional (CUDA 12.9+ on x86_64 Linux)
- CPU-only mode: Fully supported with PyTorch CPU backend

## Optional Dependencies

**Dev group:**
- pytest (see Testing section above)
- Installed with: `uv pip install -e ".[dev]"`

## Version Matrix

| Component | Python < 3.10 | Python 3.10 | Python 3.11+ |
|-----------|---|---|---|
| torch | 2.8.0 | 2.10.0 | 2.10.0 |
| pytest | 8.4.2 | 9.0.2 | 9.0.2 |
| filelock | 3.19.1 | 3.20.3 | 3.20.3 |
| fsspec | 2025.10.0 | 2026.1.0 | 2026.1.0 |
| networkx | - | 3.4.2 | 3.6.1 |

---

*Stack analysis: 2026-01-31*
