from .Axcel import AXCEL
from .HHEM import HHEM
from .Minicheck import Minicheck
from .Rouge import Rouge

# List all the metrics that should be a part of the hallucination majority voting process
__factuality__ = ['AXCEL', 'HHEM', 'Minicheck']
__similarity__ = ['Rouge']
__all__ = __factuality__ + __similarity__