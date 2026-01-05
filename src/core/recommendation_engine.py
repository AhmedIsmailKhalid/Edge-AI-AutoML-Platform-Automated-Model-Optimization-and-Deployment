"""
Recommendation Engine for intelligent optimization technique selection.

This module analyzes optimization results and provides intelligent recommendations
based on user constraints and optimization goals.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any

from src.models.experiment import OptimizationGoal
from src.models.optimization_run import OptimizationRun


class RecommendationReason(str, Enum):
    """Reasons for recommendations."""

    BEST_COMPRESSION = "best_compression"
    BEST_ACCURACY = "best_accuracy"
    BEST_BALANCE = "best_balance"
    FASTEST_INFERENCE = "fastest_inference"
    MEETS_ALL_CONSTRAINTS = "meets_all_constraints"
    CLOSEST_TO_CONSTRAINTS = "closest_to_constraints"


@dataclass
class TechniqueScore:
    """Scoring for a single technique."""

    technique_name: str
    total_score: float
    compression_score: float
    accuracy_score: float
    speed_score: float
    constraint_score: float
    meets_constraints: bool
    reasons: list[str]


@dataclass
class Recommendation:
    """A single recommendation."""

    rank: int
    technique_name: str
    score: float
    primary_reason: RecommendationReason
    explanation: str
    metrics: dict[str, Any]
    meets_constraints: bool
    strengths: list[str]
    tradeoffs: list[str]


class RecommendationEngine:
    """
    Intelligent recommendation engine for optimization techniques.

    Analyzes completed optimization runs and provides ranked recommendations
    based on user constraints and optimization goals.
    """

    def __init__(
        self,
        target_size_mb: float | None = None,
        target_accuracy_percent: float | None = None,
        target_latency_ms: float | None = None,
        optimization_goal: OptimizationGoal = OptimizationGoal.BALANCED,
    ):
        """
        Initialize recommendation engine.

        Args:
            target_size_mb: Maximum model size in MB
            target_accuracy_percent: Minimum accuracy percentage
            target_latency_ms: Maximum inference latency in ms
            optimization_goal: Primary optimization goal
        """
        self.target_size_mb = target_size_mb
        self.target_accuracy_percent = target_accuracy_percent
        self.target_latency_ms = target_latency_ms
        self.optimization_goal = optimization_goal

        # Scoring weights based on optimization goal
        self.weights = self._get_weights_for_goal(optimization_goal)

    def _get_weights_for_goal(self, goal: OptimizationGoal) -> dict[str, float]:
        """
        Get scoring weights based on optimization goal.

        Args:
            goal: Optimization goal

        Returns:
            Dictionary of weights for different metrics
        """
        # Handle both enum and string values
        goal_str = goal.value if isinstance(goal, OptimizationGoal) else goal

        if goal_str == "minimize_size":
            return {"compression": 0.6, "accuracy": 0.2, "speed": 0.1, "constraint": 0.1}
        elif goal_str == "maximize_accuracy":
            return {"compression": 0.1, "accuracy": 0.6, "speed": 0.2, "constraint": 0.1}
        elif goal_str == "minimize_latency":
            return {"compression": 0.2, "accuracy": 0.2, "speed": 0.5, "constraint": 0.1}
        else:  # "balanced"
            return {"compression": 0.3, "accuracy": 0.3, "speed": 0.3, "constraint": 0.1}

    def generate_recommendations(
        self, optimization_runs: list[OptimizationRun]
    ) -> list[Recommendation]:
        """
        Generate ranked recommendations from optimization runs.

        Args:
            optimization_runs: List of completed optimization runs

        Returns:
            List of recommendations ranked by score
        """
        # Filter completed runs
        completed_runs = [
            run for run in optimization_runs if run.status == "completed" and run.result is not None
        ]

        if not completed_runs:
            return []

        # Score each technique
        scores = [self._score_technique(run) for run in completed_runs]

        # Sort by total score (descending)
        scores.sort(key=lambda x: x.total_score, reverse=True)

        # Generate recommendations
        recommendations = []
        for rank, score in enumerate(scores, start=1):
            rec = self._create_recommendation(rank, score, completed_runs)
            recommendations.append(rec)

        return recommendations

    def _score_technique(self, run: OptimizationRun) -> TechniqueScore:
        """
        Calculate comprehensive score for a technique.

        Args:
            run: Optimization run to score

        Returns:
            TechniqueScore with detailed scoring
        """
        result = run.result

        # Calculate individual scores (0-1 scale)
        compression_score = self._calculate_compression_score(result)
        accuracy_score = self._calculate_accuracy_score(result)
        speed_score = self._calculate_speed_score(result)
        constraint_score, meets_constraints = self._calculate_constraint_score(result)

        # Calculate weighted total score
        total_score = (
            compression_score * self.weights["compression"]
            + accuracy_score * self.weights["accuracy"]
            + speed_score * self.weights["speed"]
            + constraint_score * self.weights["constraint"]
        )

        # Generate reasons
        reasons = self._generate_score_reasons(
            run.technique_name, compression_score, accuracy_score, speed_score, meets_constraints
        )

        return TechniqueScore(
            technique_name=run.technique_name,
            total_score=total_score,
            compression_score=compression_score,
            accuracy_score=accuracy_score,
            speed_score=speed_score,
            constraint_score=constraint_score,
            meets_constraints=meets_constraints,
            reasons=reasons,
        )

    def _calculate_compression_score(self, result: Any) -> float:
        """
        Calculate compression quality score.

        Higher compression ratio = higher score
        """
        if not result.original_size_mb or result.original_size_mb == 0:
            return 0.0

        compression_ratio = result.original_size_mb / result.optimized_size_mb

        # Normalize: 1x = 0, 4x = 0.5, 10x+ = 1.0
        normalized = min(compression_ratio / 10.0, 1.0)

        return normalized

    def _calculate_accuracy_score(self, result: Any) -> float:
        """
        Calculate accuracy retention score.

        Higher accuracy retention = higher score
        """
        if result.original_accuracy == 0:
            return 1.0  # Can't measure, assume perfect

        accuracy_retention = result.optimized_accuracy / result.original_accuracy

        # Penalize accuracy loss heavily
        if accuracy_retention >= 0.99:
            return 1.0
        elif accuracy_retention >= 0.95:
            return 0.9
        elif accuracy_retention >= 0.90:
            return 0.7
        elif accuracy_retention >= 0.85:
            return 0.5
        else:
            return max(accuracy_retention - 0.2, 0.0)

    def _calculate_speed_score(self, result: Any) -> float:
        """
        Calculate inference speed score.

        Smaller models are generally faster.
        """
        # For now, use model size as proxy for speed
        # Later: use actual latency measurements
        size_mb = result.optimized_size_mb

        # Normalize: <1MB = 1.0, 5MB = 0.5, 10MB+ = 0
        if size_mb < 1:
            return 1.0
        elif size_mb < 5:
            return 1.0 - (size_mb - 1) / 8.0
        else:
            return max(0.5 - (size_mb - 5) / 10.0, 0.0)

    def _calculate_constraint_score(self, result: Any) -> tuple[float, bool]:
        """
        Calculate how well the result meets constraints.

        Returns:
            (score, meets_all_constraints)
        """
        violations = 0
        total_constraints = 0

        # Check size constraint
        if self.target_size_mb is not None:
            total_constraints += 1
            if result.optimized_size_mb > self.target_size_mb:
                violations += 1

        # Check accuracy constraint
        if self.target_accuracy_percent is not None:
            total_constraints += 1
            accuracy_percent = result.optimized_accuracy * 100
            if accuracy_percent < self.target_accuracy_percent:
                violations += 1

        # Check latency constraint (not implemented yet, placeholder)
        if self.target_latency_ms is not None:
            total_constraints += 1
            # TODO: Implement latency estimation
            # For now, assume latency constraint is met

        if total_constraints == 0:
            return 1.0, True

        meets_all = violations == 0
        score = 1.0 - (violations / total_constraints)

        return score, meets_all

    def _generate_score_reasons(
        self,
        technique_name: str,
        compression_score: float,
        accuracy_score: float,
        speed_score: float,
        meets_constraints: bool,
    ) -> list[str]:
        """Generate human-readable reasons for the score."""
        reasons = []

        if compression_score >= 0.8:
            reasons.append("Excellent compression ratio")
        elif compression_score >= 0.5:
            reasons.append("Good compression ratio")

        if accuracy_score >= 0.95:
            reasons.append("Minimal accuracy loss")
        elif accuracy_score >= 0.85:
            reasons.append("Acceptable accuracy retention")
        elif accuracy_score < 0.7:
            reasons.append("Significant accuracy loss")

        if speed_score >= 0.8:
            reasons.append("Very fast inference")
        elif speed_score >= 0.5:
            reasons.append("Fast inference")

        if meets_constraints:
            reasons.append("Meets all constraints")
        else:
            reasons.append("Does not meet all constraints")

        return reasons

    def _create_recommendation(
        self, rank: int, score: TechniqueScore, all_runs: list[OptimizationRun]
    ) -> Recommendation:
        """
        Create a full recommendation from a score.

        Args:
            rank: Ranking position
            score: Technique score
            all_runs: All optimization runs for context

        Returns:
            Complete recommendation
        """
        # Find the corresponding run
        run = next(r for r in all_runs if r.technique_name == score.technique_name)
        result = run.result

        # Determine primary reason
        primary_reason = self._determine_primary_reason(score)

        # Generate explanation
        explanation = self._generate_explanation(score, primary_reason, result)

        # Identify strengths
        strengths = self._identify_strengths(score, result)

        # Identify tradeoffs
        tradeoffs = self._identify_tradeoffs(score, result)

        # Gather metrics
        metrics = {
            "original_size_mb": result.original_size_mb,
            "optimized_size_mb": result.optimized_size_mb,
            "compression_ratio": result.original_size_mb / result.optimized_size_mb
            if result.optimized_size_mb > 0
            else 0,
            "original_accuracy": result.original_accuracy,
            "optimized_accuracy": result.optimized_accuracy,
            "accuracy_retention": result.optimized_accuracy / result.original_accuracy
            if result.original_accuracy > 0
            else 1.0,
            "execution_time_seconds": run.execution_time_seconds,
        }

        return Recommendation(
            rank=rank,
            technique_name=score.technique_name,
            score=score.total_score,
            primary_reason=primary_reason,
            explanation=explanation,
            metrics=metrics,
            meets_constraints=score.meets_constraints,
            strengths=strengths,
            tradeoffs=tradeoffs,
        )

    def _determine_primary_reason(self, score: TechniqueScore) -> RecommendationReason:
        """Determine the primary reason for recommendation."""
        if score.meets_constraints:
            if score.compression_score >= 0.8 and score.accuracy_score >= 0.9:
                return RecommendationReason.MEETS_ALL_CONSTRAINTS
            elif score.compression_score >= 0.8:
                return RecommendationReason.BEST_COMPRESSION
            elif score.accuracy_score >= 0.95:
                return RecommendationReason.BEST_ACCURACY
            else:
                return RecommendationReason.BEST_BALANCE
        else:
            return RecommendationReason.CLOSEST_TO_CONSTRAINTS

    def _generate_explanation(
        self, score: TechniqueScore, reason: RecommendationReason, result: Any
    ) -> str:
        """Generate human-readable explanation."""
        compression_ratio = (
            result.original_size_mb / result.optimized_size_mb
            if result.optimized_size_mb > 0
            else 0
        )
        accuracy_retention = (
            (result.optimized_accuracy / result.original_accuracy * 100)
            if result.original_accuracy > 0
            else 100
        )

        if reason == RecommendationReason.MEETS_ALL_CONSTRAINTS:
            return (
                f"This technique achieves {compression_ratio:.1f}x compression "
                f"while retaining {accuracy_retention:.1f}% of original accuracy, "
                f"meeting all your constraints."
            )
        elif reason == RecommendationReason.BEST_COMPRESSION:
            return (
                f"Best compression with {compression_ratio:.1f}x size reduction. "
                f"Accuracy: {accuracy_retention:.1f}% of original."
            )
        elif reason == RecommendationReason.BEST_ACCURACY:
            return (
                f"Best accuracy retention at {accuracy_retention:.1f}% with "
                f"{compression_ratio:.1f}x compression."
            )
        elif reason == RecommendationReason.BEST_BALANCE:
            return (
                f"Well-balanced approach: {compression_ratio:.1f}x compression "
                f"with {accuracy_retention:.1f}% accuracy retention."
            )
        else:
            return (
                f"Closest to meeting constraints. {compression_ratio:.1f}x compression, "
                f"{accuracy_retention:.1f}% accuracy retention."
            )

    def _identify_strengths(self, score: TechniqueScore, result: Any) -> list[str]:
        """Identify key strengths of the technique."""
        strengths = []

        if score.compression_score >= 0.8:
            compression_ratio = (
                result.original_size_mb / result.optimized_size_mb
                if result.optimized_size_mb > 0
                else 0
            )
            strengths.append(f"Excellent {compression_ratio:.1f}x compression")
        elif score.compression_score >= 0.3:
            compression_ratio = (
                result.original_size_mb / result.optimized_size_mb
                if result.optimized_size_mb > 0
                else 0
            )
            strengths.append(f"Good {compression_ratio:.1f}x compression")

        if score.accuracy_score >= 0.95:
            strengths.append("Minimal accuracy loss (<5%)")
        elif score.accuracy_score >= 0.90:
            strengths.append("Low accuracy loss (<10%)")
        elif score.accuracy_score >= 0.70:
            strengths.append("Moderate accuracy retention")

        if result.optimized_size_mb < 1:
            strengths.append("Very small model size (<1MB)")
        elif result.optimized_size_mb < 5:
            strengths.append("Small model size (<5MB)")

        if score.speed_score >= 0.8:
            strengths.append("Fast inference expected")
        elif score.speed_score >= 0.5:
            strengths.append("Reasonable inference speed")

        # Ensure at least one strength
        if not strengths:
            strengths.append("Successfully completed optimization")

        return strengths

    def _identify_tradeoffs(self, score: TechniqueScore, result: Any) -> list[str]:
        """Identify tradeoffs and limitations."""
        tradeoffs = []

        if score.accuracy_score < 0.85:
            accuracy_loss = (
                (1 - result.optimized_accuracy / result.original_accuracy) * 100
                if result.original_accuracy > 0
                else 0
            )
            tradeoffs.append(f"Significant accuracy loss ({accuracy_loss:.1f}%)")

        if score.compression_score < 0.3:
            tradeoffs.append("Limited compression achieved")

        if result.optimized_size_mb > 5:
            tradeoffs.append("Relatively large model size")

        if not score.meets_constraints:
            tradeoffs.append("Does not meet all specified constraints")

        return tradeoffs
