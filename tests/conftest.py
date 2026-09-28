# Configure test environment before any imports
import os
import sys

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Set Modin environment variables before any Modin imports anywhere
# This prevents tests from setting conflicting storage formats
os.environ["MODIN_STORAGE_FORMAT"] = "pandas"

# Try to use Ray backend first, fall back to Dask if Ray is not available (Windows)
try:
    # Try Ray first (preferred for Linux/Mac)
    import ray
    ray.init(ignore_reinit_error=True, num_cpus=2)
    os.environ["MODIN_ENGINE"] = "ray"

    from modin.config import Engine, StorageFormat
    Engine.put("ray")
    StorageFormat.put("pandas")
    print("Modin configured with Ray backend")
except ImportError:
    # Ray not available, try Dask (works on Windows)
    try:
        os.environ["MODIN_ENGINE"] = "dask"

        from modin.config import Engine, StorageFormat
        Engine.put("dask")
        StorageFormat.put("pandas")
        print("Modin configured with Dask backend (Ray not available)")
    except ImportError:
        # Neither Ray nor Dask available - Modin tests will be skipped
        print("Neither Ray nor Dask available - Modin tests will be skipped")
        pass
