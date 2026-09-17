"""RSJG Joint Dependency V2 building blocks.

The package is deliberately independent from the legacy GDTS path.  Importing
it does not construct modules or consume random numbers; the parent model only
instantiates these classes when ``jdv2_active`` is true.
"""

from .dependency_corrector import DependencyCorrector
from .dynamic_relation import DynamicHypothesisRelation
from .future_teacher import SceneFutureTeacher
from .joint_energy import RelationSpecificJointEnergy
from .joint_sampler import ParallelConditionalSampler
from .scene_latent import SceneLatentPrior
from .unary_goal import UnaryGoalResidual

__all__ = [
    "DependencyCorrector",
    "DynamicHypothesisRelation",
    "ParallelConditionalSampler",
    "RelationSpecificJointEnergy",
    "SceneFutureTeacher",
    "SceneLatentPrior",
    "UnaryGoalResidual",
]
