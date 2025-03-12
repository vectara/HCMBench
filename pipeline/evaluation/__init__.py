from .axcel import AXCEL
from .hhem import HHEM
from .minicheck import Minicheck
from .rouge import Rouge

# List all the metrics that should be a part of the hallucination majority voting process
__factuality__ = ['AXCEL', 'HHEM', 'Minicheck']
__similarity__ = ['Rouge']
__all__ = __factuality__ + __similarity__