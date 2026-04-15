"""
Base abstraction for 'for' loop substitution strategies.

Extends the generic control structure strategy with semantics
specific to 'for' loop transformations.
"""

from ..base_control_structure_strategy import BaseControlStructureStrategy


class BaseForLoopStrategy(BaseControlStructureStrategy):
    """
    Specialization of BaseControlStructureStrategy for 'for' loops.

    This class does not add new methods but serves as a semantic layer
    to group all 'for' loop strategies together.
    """

    pass
