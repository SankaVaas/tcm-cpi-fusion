# Lazy — avoids crashing on import when torch_scatter is missing/broken
def get_dual_stream():
    from .dual_stream import DualStreamCPI
    return DualStreamCPI

def get_baseline():
    from .baseline import BaselineCPI
    return BaselineCPI

# Direct for normal usage
from .baseline import BaselineCPI
