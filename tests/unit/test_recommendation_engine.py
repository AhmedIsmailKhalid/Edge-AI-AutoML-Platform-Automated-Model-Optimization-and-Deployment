"""
Unit tests for the Recommendation Engine.
"""

from unittest.mock import Mock

import pytest

from src.core.recommendation_engine import RecommendationEngine
from src.models.experiment import OptimizationGoal


@pytest.fixture
def mock_optimization_runs():
    """Create mock optimization runs with different characteristics."""
    runs = []

    # Run 1: Excellent compression, good accuracy
    run1 = Mock()
    run1.technique_name = "ptq_int8"
    run1.status = "completed"
    run1.execution_time_seconds = 5.0
    run1.result = Mock(
        original_size_mb=10.0,
        optimized_size_mb=1.0,
        original_accuracy=0.95,
        optimized_accuracy=0.93,
        original_params_count=1000000,
        optimized_params_count=1000000,
    )
    runs.append(run1)

    # Run 2: Good compression, excellent accuracy
    run2 = Mock()
    run2.technique_name = "qat"
    run2.status = "completed"
    run2.execution_time_seconds = 20.0
    run2.result = Mock(
        original_size_mb=10.0,
        optimized_size_mb=2.5,
        original_accuracy=0.95,
        optimized_accuracy=0.95,
        original_params_count=1000000,
        optimized_params_count=1000000,
    )
    runs.append(run2)

    # Run 3: Moderate compression, poor accuracy
    run3 = Mock()
    run3.technique_name = "pruning"
    run3.status = "completed"
    run3.execution_time_seconds = 10.0
    run3.result = Mock(
        original_size_mb=10.0,
        optimized_size_mb=5.0,
        original_accuracy=0.95,
        optimized_accuracy=0.80,
        original_params_count=1000000,
        optimized_params_count=500000,
    )
    runs.append(run3)

    # Run 4: Extreme compression, moderate accuracy
    run4 = Mock()
    run4.technique_name = "distillation"
    run4.status = "completed"
    run4.execution_time_seconds = 30.0
    run4.result = Mock(
        original_size_mb=10.0,
        optimized_size_mb=0.5,
        original_accuracy=0.95,
        optimized_accuracy=0.88,
        original_params_count=1000000,
        optimized_params_count=100000,
    )
    runs.append(run4)

    return runs


def test_recommendation_engine_initialization():
    """Test recommendation engine initialization."""
    print("\n🧪 Testing recommendation engine initialization...")

    engine = RecommendationEngine(
        target_size_mb=2.0,
        target_accuracy_percent=90.0,
        target_latency_ms=10.0,
        optimization_goal=OptimizationGoal.BALANCED,
    )

    assert engine.target_size_mb == 2.0
    assert engine.target_accuracy_percent == 90.0
    assert engine.target_latency_ms == 10.0
    assert engine.optimization_goal == OptimizationGoal.BALANCED

    print("✅ Initialization test passed")


def test_weights_for_different_goals():
    """Test that weights change based on optimization goal."""
    print("\n🧪 Testing weights for different goals...")

    # Min size goal
    engine_size = RecommendationEngine(optimization_goal=OptimizationGoal.MINIMIZE_SIZE)
    assert engine_size.weights["compression"] > engine_size.weights["accuracy"]
    print(f"   MIN_SIZE weights: {engine_size.weights}")

    # Max accuracy goal
    engine_accuracy = RecommendationEngine(optimization_goal=OptimizationGoal.MAXIMIZE_ACCURACY)
    assert engine_accuracy.weights["accuracy"] > engine_accuracy.weights["compression"]
    print(f"   MAX_ACCURACY weights: {engine_accuracy.weights}")

    # Balanced goal
    engine_balanced = RecommendationEngine(optimization_goal=OptimizationGoal.BALANCED)
    assert engine_balanced.weights["compression"] == engine_balanced.weights["accuracy"]
    print(f"   BALANCED weights: {engine_balanced.weights}")

    print("✅ Weights test passed")


def test_generate_recommendations_basic(mock_optimization_runs):
    """Test basic recommendation generation."""
    print("\n🧪 Testing basic recommendation generation...")

    engine = RecommendationEngine(optimization_goal=OptimizationGoal.BALANCED)
    recommendations = engine.generate_recommendations(mock_optimization_runs)

    # Should have 4 recommendations
    assert len(recommendations) == 4

    # Should be ranked
    for i in range(len(recommendations) - 1):
        assert recommendations[i].score >= recommendations[i + 1].score

    # Check structure
    for rec in recommendations:
        assert rec.rank > 0
        assert rec.technique_name
        assert 0 <= rec.score <= 1
        assert rec.primary_reason
        assert rec.explanation
        assert rec.metrics
        assert isinstance(rec.strengths, list)
        assert isinstance(rec.tradeoffs, list)

    print("✅ Basic recommendation generation test passed")


def test_recommendations_with_size_goal(mock_optimization_runs):
    """Test recommendations prioritize compression with MIN_SIZE goal."""
    print("\n🧪 Testing MIN_SIZE goal recommendations...")

    engine = RecommendationEngine(optimization_goal=OptimizationGoal.MINIMIZE_SIZE)
    recommendations = engine.generate_recommendations(mock_optimization_runs)

    # Top recommendation should be the one with best compression (distillation: 20x)
    top_rec = recommendations[0]
    print(f"   Top recommendation: {top_rec.technique_name}")
    print(f"   Score: {top_rec.score:.3f}")
    print(f"   Compression: {top_rec.metrics['compression_ratio']:.1f}x")

    # Distillation (0.5MB) should rank higher than QAT (2.5MB)
    distillation_rank = next(r.rank for r in recommendations if r.technique_name == "distillation")
    qat_rank = next(r.rank for r in recommendations if r.technique_name == "qat")

    assert distillation_rank < qat_rank, "Distillation should rank higher with MIN_SIZE goal"

    print("✅ MIN_SIZE goal test passed")


def test_recommendations_with_accuracy_goal(mock_optimization_runs):
    """Test recommendations prioritize accuracy with MAX_ACCURACY goal."""
    print("\n🧪 Testing MAX_ACCURACY goal recommendations...")

    engine = RecommendationEngine(optimization_goal=OptimizationGoal.MAXIMIZE_ACCURACY)
    recommendations = engine.generate_recommendations(mock_optimization_runs)

    # Top recommendation should prioritize accuracy retention
    top_rec = recommendations[0]
    print(f"   Top recommendation: {top_rec.technique_name}")
    print(f"   Score: {top_rec.score:.3f}")
    print(f"   Accuracy retention: {top_rec.metrics['accuracy_retention']:.1%}")

    # QAT (95% retention) should rank higher than pruning (84% retention)
    qat_rank = next(r.rank for r in recommendations if r.technique_name == "qat")
    pruning_rank = next(r.rank for r in recommendations if r.technique_name == "pruning")

    assert qat_rank < pruning_rank, "QAT should rank higher with MAX_ACCURACY goal"

    print("✅ MAX_ACCURACY goal test passed")


def test_recommendations_with_constraints(mock_optimization_runs):
    """Test constraint checking in recommendations."""
    print("\n🧪 Testing constraint checking...")

    # Set strict constraints
    engine = RecommendationEngine(
        target_size_mb=2.0,
        target_accuracy_percent=92.0,
        optimization_goal=OptimizationGoal.BALANCED,
    )

    recommendations = engine.generate_recommendations(mock_optimization_runs)

    # Check which techniques meet constraints
    meets_constraints = [r for r in recommendations if r.meets_constraints]
    violates_constraints = [r for r in recommendations if not r.meets_constraints]

    print(f"   Techniques meeting constraints: {len(meets_constraints)}")
    print(f"   Techniques violating constraints: {len(violates_constraints)}")

    for rec in meets_constraints:
        print(
            f"   ✅ {rec.technique_name}: {rec.metrics['optimized_size_mb']:.1f}MB, "
            f"{rec.metrics['accuracy_retention']:.1%} accuracy"
        )

    for rec in violates_constraints:
        print(
            f"   ❌ {rec.technique_name}: {rec.metrics['optimized_size_mb']:.1f}MB, "
            f"{rec.metrics['accuracy_retention']:.1%} accuracy"
        )

    # PTQ INT8 should meet constraints (1MB, 97.9% accuracy)
    ptq_rec = next(r for r in recommendations if r.technique_name == "ptq_int8")
    assert ptq_rec.meets_constraints, "PTQ INT8 should meet constraints"

    # Pruning should violate constraints (5MB, 84% accuracy)
    pruning_rec = next(r for r in recommendations if r.technique_name == "pruning")
    assert not pruning_rec.meets_constraints, "Pruning should violate constraints"

    print("✅ Constraint checking test passed")


def test_recommendation_explanations(mock_optimization_runs):
    """Test that explanations are generated."""
    print("\n🧪 Testing recommendation explanations...")

    engine = RecommendationEngine(optimization_goal=OptimizationGoal.BALANCED)
    recommendations = engine.generate_recommendations(mock_optimization_runs)

    for rec in recommendations:
        print(f"\n   {rec.rank}. {rec.technique_name.upper()}")
        print(f"      Explanation: {rec.explanation}")
        print(f"      Strengths: {', '.join(rec.strengths)}")
        if rec.tradeoffs:
            print(f"      Tradeoffs: {', '.join(rec.tradeoffs)}")

    # All should have explanations
    for rec in recommendations:
        assert len(rec.explanation) > 0, f"{rec.technique_name} missing explanation"
        assert len(rec.strengths) > 0, f"{rec.technique_name} missing strengths"

    print("\n✅ Explanation test passed")


def test_empty_runs():
    """Test handling of empty optimization runs."""
    print("\n🧪 Testing empty runs...")

    engine = RecommendationEngine(optimization_goal=OptimizationGoal.BALANCED)
    recommendations = engine.generate_recommendations([])

    assert len(recommendations) == 0
    print("✅ Empty runs test passed")


def test_incomplete_runs():
    """Test filtering of incomplete runs."""
    print("\n🧪 Testing incomplete runs filtering...")

    # Create mix of completed and failed runs
    runs = []

    # Completed run
    run1 = Mock()
    run1.technique_name = "ptq_int8"
    run1.status = "completed"
    run1.execution_time_seconds = 5.0
    run1.result = Mock(
        original_size_mb=10.0,
        optimized_size_mb=1.0,
        original_accuracy=0.95,
        optimized_accuracy=0.93,
        original_params_count=1000000,
        optimized_params_count=1000000,
    )
    runs.append(run1)

    # Failed run
    run2 = Mock()
    run2.technique_name = "distillation"
    run2.status = "failed"
    run2.result = None
    runs.append(run2)

    # Running run
    run3 = Mock()
    run3.technique_name = "qat"
    run3.status = "running"
    run3.result = None
    runs.append(run3)

    engine = RecommendationEngine(optimization_goal=OptimizationGoal.BALANCED)
    recommendations = engine.generate_recommendations(runs)

    # Should only have 1 recommendation (completed one)
    assert len(recommendations) == 1
    assert recommendations[0].technique_name == "ptq_int8"

    print("✅ Incomplete runs filtering test passed")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
