"""Verifiable reward computation, candidate generation, and lightweight verification."""

from gui_grounding.reward.lightweight_verifier import (
    DEFAULT_LIGHTWEIGHT_VERIFIER_CONFIG,
    DUAL_PATH_CANDIDATE_SCHEMA,
    LIGHTWEIGHT_VERIFIER_SCHEMA,
    build_dual_path_candidates,
    score_dual_path_candidates,
)
from gui_grounding.reward.verifiable_reward import VerifiableRewardCalculator

__all__ = [
    "DEFAULT_LIGHTWEIGHT_VERIFIER_CONFIG",
    "DUAL_PATH_CANDIDATE_SCHEMA",
    "LIGHTWEIGHT_VERIFIER_SCHEMA",
    "VerifiableRewardCalculator",
    "build_dual_path_candidates",
    "score_dual_path_candidates",
]
