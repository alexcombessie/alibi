"""
The 'alibi.explainers' module includes feature importance, counterfactual and anchor-based explainers.
"""

from .ale import ALE, plot_ale
from .anchor_tabular import AnchorTabular, DistributedAnchorTabular
from .anchor_text import AnchorText
from .anchor_image import AnchorImage

__all__ = ["ALE",
           "AnchorTabular",
           "DistributedAnchorTabular",
           "AnchorText",
           "AnchorImage",
           "plot_ale",
           ]

try:
    from .shap_wrappers import KernelShap, TreeShap
    __all__ += ["KernelShap", "TreeShap"]
except ImportError:
    pass
