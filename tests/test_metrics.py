# Tests for the metrics framework

import pytest
from src.metrics import (
    MetricsFramework,
    DimensionRegistry,
    DimensionPlugin,
    FraudCorrectness,
    Novelty,
    BehavioralPlausibility,
    SupportingEvidence,
    ExplanationQuality,
    ConfidenceAssessment,
)


# ---- helpers ----

def make_hypothesis(hypothesis_id="h1", confidence=0.7):
    return {
        "hypothesis_id": hypothesis_id,
        "mechanism_description": "Shell company layering to conceal beneficial ownership through nominee directors",
        "entity_roles": ["nominee_director", "shell_company", "beneficial_owner"],
        "relationship_flows": ["ownership_chain", "fund_transfer"],
        "point_of_deviation": "concealment of beneficial ownership",
        "cited_entity_ids": ["e1", "e2", "e3"],
        "cited_relationship_ids": ["r1", "r2"],
        "cited_event_ids": ["ev1"],
        "explanation": "The mechanism operates by layering shell companies across multiple jurisdictions, appointing nominee directors to obscure the true beneficial owner. Fund transfers flow through these intermediary entities in a pattern designed to break the audit trail and evade regulatory detection.",
        "confidence": confidence,
    }


def make_context():
    return {
        "ground_truth": {
            "entity_roles": ["nominee_director", "shell_company", "beneficial_owner"],
            "relationship_flows": ["ownership_chain", "fund_transfer"],
            "point_of_deviation": "concealment of beneficial ownership",
        },
        "knowledge_space_lookup": {
            "max_structural_proximity": 0.3,
            "max_behavioral_similarity": 0.2,
        },
        "visible_information": {"entity_count": 10},
    }


# ---- registry tests ----

class TestRegistry:
    def test_register_all_defaults(self):
        reg = DimensionRegistry()
        reg.register_all_defaults()
        assert reg.count() == 6

    def test_all_dimension_names(self):
        reg = DimensionRegistry()
        reg.register_all_defaults()
        names = reg.dimension_names()
        assert "fraud_correctness" in names
        assert "novelty" in names
        assert "behavioral_plausibility" in names
        assert "supporting_evidence" in names
        assert "explanation_quality" in names
        assert "confidence_assessment" in names

    def test_validate_manifest_all_present(self):
        reg = DimensionRegistry()
        reg.register_all_defaults()
        missing = reg.validate_manifest(["fraud_correctness", "novelty"])
        assert missing == []

    def test_validate_manifest_missing(self):
        reg = DimensionRegistry()
        reg.register_all_defaults()
        missing = reg.validate_manifest(["fraud_correctness", "telekinesis"])
        assert "telekinesis" in missing

    def test_get_specific_plugin(self):
        reg = DimensionRegistry()
        reg.register_all_defaults()
        plugin = reg.get("novelty")
        assert plugin is not None
        assert plugin.name == "novelty"


# ---- individual dimension tests ----

class TestFraudCorrectness:
    def test_full_correspondence(self):
        dim = FraudCorrectness()
        result = dim.assess(make_hypothesis(), make_context())
        assert result.graded_outcome == "full_correspondence"
        assert result.score >= 0.7

    def test_no_correspondence(self):
        dim = FraudCorrectness()
        h = {"hypothesis_id": "h2", "entity_roles": ["alien"], "relationship_flows": ["warp"]}
        result = dim.assess(h, make_context())
        assert result.graded_outcome == "no_correspondence"
        assert result.score < 0.3


class TestNovelty:
    def test_high_novelty(self):
        dim = Novelty()
        ctx = {"knowledge_space_lookup": {"max_structural_proximity": 0.1, "max_behavioral_similarity": 0.1}}
        result = dim.assess(make_hypothesis(), ctx)
        assert result.graded_outcome == "highly_novel"
        assert result.score >= 0.7

    def test_low_novelty(self):
        dim = Novelty()
        ctx = {"knowledge_space_lookup": {"max_structural_proximity": 0.9, "max_behavioral_similarity": 0.9}}
        result = dim.assess(make_hypothesis(), ctx)
        assert result.graded_outcome == "low_novelty"
        assert result.score < 0.4


class TestBehavioralPlausibility:
    def test_complete_hypothesis_is_plausible(self):
        dim = BehavioralPlausibility()
        result = dim.assess(make_hypothesis(), {})
        assert result.graded_outcome == "highly_plausible"

    def test_empty_hypothesis_is_implausible(self):
        dim = BehavioralPlausibility()
        result = dim.assess({"hypothesis_id": "empty"}, {})
        assert result.graded_outcome == "implausible"
        assert result.score == 0.0


class TestSupportingEvidence:
    def test_well_cited(self):
        dim = SupportingEvidence()
        result = dim.assess(make_hypothesis(), {})
        assert result.graded_outcome == "strongly_supported"

    def test_no_citations(self):
        dim = SupportingEvidence()
        result = dim.assess({"hypothesis_id": "h"}, {})
        assert result.graded_outcome == "weakly_supported"
        assert result.score == 0.0


class TestExplanationQuality:
    def test_detailed_explanation(self):
        dim = ExplanationQuality()
        result = dim.assess(make_hypothesis(), {})
        assert result.score >= 0.6
        assert result.status == "pending_expert_review"

    def test_no_explanation(self):
        dim = ExplanationQuality()
        result = dim.assess({"hypothesis_id": "h"}, {})
        assert result.graded_outcome == "no_explanation"


class TestConfidenceAssessment:
    def test_well_calibrated(self):
        dim = ConfidenceAssessment()
        h = make_hypothesis(confidence=1.0)
        result = dim.assess(h, make_context())
        assert result.graded_outcome == "well_calibrated"

    def test_poorly_calibrated(self):
        dim = ConfidenceAssessment()
        h = {"hypothesis_id": "h", "entity_roles": ["x"], "confidence": 1.0}
        ctx = {"ground_truth": {"entity_roles": ["a", "b", "c"]}}
        result = dim.assess(h, ctx)
        assert result.graded_outcome == "poorly_calibrated"


# ---- framework integration tests ----

class TestMetricsFramework:
    def test_evaluate_returns_profile_with_6_dimensions(self):
        mf = MetricsFramework()
        profile = mf.evaluate(make_hypothesis(), make_context(), "ep-1")
        assert profile.dimension_count == 6
        assert profile.episode_id == "ep-1"
        assert profile.hypothesis_id == "h1"

    def test_no_composite_score(self):
        mf = MetricsFramework()
        profile = mf.evaluate(make_hypothesis(), make_context(), "ep-1")
        assert isinstance(profile.dimension_results, list)
        assert all(r.dimension_name for r in profile.dimension_results)

    def test_dimension_independence(self):
        mf = MetricsFramework()
        profile = mf.evaluate(make_hypothesis(), make_context(), "ep-1")
        names = [r.dimension_name for r in profile.dimension_results]
        assert len(names) == len(set(names))

    def test_single_dimension_evaluation(self):
        mf = MetricsFramework()
        result = mf.evaluate_single_dimension("novelty", make_hypothesis(), make_context(), "ep-1")
        assert result is not None
        assert result.dimension_name == "novelty"

    def test_single_dimension_nonexistent(self):
        mf = MetricsFramework()
        result = mf.evaluate_single_dimension("telekinesis", make_hypothesis(), {}, "ep-1")
        assert result is None

    def test_failure_isolation(self):
        mf = MetricsFramework()
        profile = mf.evaluate(make_hypothesis(), {}, "ep-1")
        assert profile.dimension_count == 6
        assert all(r.dimension_name for r in profile.dimension_results)


# ---- aggregation tests ----

class TestAggregation:
    def test_episode_aggregation(self):
        mf = MetricsFramework()
        p1 = mf.evaluate(make_hypothesis("h1"), make_context(), "ep-1")
        p2 = mf.evaluate(make_hypothesis("h2", confidence=0.3), make_context(), "ep-1")
        summary = mf.aggregate_episode([p1, p2], "ep-1")
        assert summary["hypothesis_count"] == 2
        assert "fraud_correctness" in summary["dimensions"]
        assert summary["dimensions"]["fraud_correctness"]["count"] == 2

    def test_empty_aggregation(self):
        mf = MetricsFramework()
        summary = mf.aggregate_episode([], "ep-1")
        assert summary["hypothesis_count"] == 0
        assert summary["dimensions"] == {}

    def test_aggregation_no_composite(self):
        mf = MetricsFramework()
        p1 = mf.evaluate(make_hypothesis(), make_context(), "ep-1")
        summary = mf.aggregate_episode([p1], "ep-1")
        assert len(summary["dimensions"]) == 6
