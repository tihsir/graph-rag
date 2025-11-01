"""GraphRAG package."""

# Import exceptiongroup very early for Python 3.9 compatibility
# This must happen before any anyio-dependent imports
import sys
if sys.version_info < (3, 11):
    try:
        from exceptiongroup import ExceptionGroup
        # Monkey-patch into builtins so anyio can find it
        if not hasattr(sys.modules.get('builtins', object()), 'ExceptionGroup'):
            import builtins
            builtins.ExceptionGroup = ExceptionGroup
    except ImportError:
        pass

__version__ = "0.1.0"
