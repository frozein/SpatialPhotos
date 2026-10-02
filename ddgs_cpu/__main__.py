"""Build the CPU renderer: python -m ddgs_cpu."""

from . import _load_extension


if __name__ == "__main__":
    print("Building ddgs_cpu (CPU-only C++ extension)...", flush=True)
    extension = _load_extension(verbose=True)
    print(f"ddgs_cpu ready: {extension.__file__}", flush=True)
