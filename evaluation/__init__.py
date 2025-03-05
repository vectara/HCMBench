from .Axcel import AXCEL
from .HHEM import HHEM
from .Minicheck import Minicheck
from .Rouge import Rouge

# List all the metrics that should be a part of the hallucination majority voting process
__vote__ = ['AXCEL', 'HHEM', 'Minicheck']
__all__ = __vote__ + ['Rouge']