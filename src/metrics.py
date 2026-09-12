# Metrics for evaluating fraud detection hypotheses.
# 6 dimensions, each scores independently. No composite scores ever.

from src.models import DimensionResult, EvaluationProfile


# base class for all dimensions
class DimensionPlugin:
    name = ""
    description = ""
    requires_expert_review = False

    def assess(self, hypothesis, context):
        raise NotImplementedError


class FraudCorrectness(DimensionPlugin):
    name = "fraud_correctness"
    description = "How well the hypothesis matches the actual fraud mechanism"

    def assess(self, hypothesis, context):
        ground_truth = context.get("ground_truth", {})
        gt_roles = set(ground_truth.get("entity_roles", []))
        gt_flows = set(ground_truth.get("relationship_flows", []))
        gt_deviation = ground_truth.get("point_of_deviation", "")

        h_roles = set(hypothesis.get("entity_roles", []))
        h_flows = set(hypothesis.get("relationship_flows", []))
        h_deviation = hypothesis.get("point_of_deviation", "")

        # jaccard-ish overlap for roles and flows, exact match on deviation
        role_score = len(gt_roles & h_roles) / max(len(gt_roles | h_roles), 1)
        flow_score = len(gt_flows & h_flows) / max(len(gt_flows | h_flows), 1)
        deviation_score = 1.0 if gt_deviation and gt_deviation.lower() in h_deviation.lower() else 0.0

        score = (role_score + flow_score + deviation_score) / 3.0

        if score >= 0.7:
            outcome = "full_correspondence"
        elif score >= 0.3:
            outcome = "partial_correspondence"
        else:
            outcome = "no_correspondence"

        return DimensionResult(
            dimension_name=self.name,
            hypothesis_id=hypothesis.get("hypothesis_id", ""),
            graded_outcome=outcome,
            score=round(score, 3),
            interpretation=f"Correctness {score:.2f} based on role/flow/deviation overlap",
            status="resolved",
        )


class Novelty(DimensionPlugin):
    name = "novelty"
    description = "How different this is from already documented fraud patterns"

    def assess(self, hypothesis, context):
        lookup = context.get("knowledge_space_lookup", {})
        max_structural = lookup.get("max_structural_proximity", 0.0)
        max_behavioral = lookup.get("max_behavioral_similarity", 0.0)

        # novelty = inverse of how similar it is to known patterns
        novelty_score = 1.0 - (max_structural + max_behavioral) / 2.0

        if novelty_score >= 0.7:
            outcome = "highly_novel"
        elif novelty_score >= 0.4:
            outcome = "moderately_novel"
        else:
            outcome = "low_novelty"

        return DimensionResult(
            dimension_name=self.name,
            hypothesis_id=hypothesis.get("hypothesis_id", ""),
            graded_outcome=outcome,
            score=round(novelty_score, 3),
            interpretation=f"Novelty {novelty_score:.2f} (1.0 = completely new, 0.0 = already known)",
            status="resolved",
        )


class BehavioralPlausibility(DimensionPlugin):
    name = "behavioral_plausibility"
    description = "Whether the mechanism is financially plausible"

    def assess(self, hypothesis, context):
        # simple heuristic: just check if the main fields are filled in
        entity_roles = hypothesis.get("entity_roles", [])
        flows = hypothesis.get("relationship_flows", [])
        deviation = hypothesis.get("point_of_deviation", "")

        completeness = 0.0
        if entity_roles:
            completeness += 0.4
        if flows:
            completeness += 0.3
        if deviation:
            completeness += 0.3

        if completeness >= 0.8:
            outcome = "highly_plausible"
        elif completeness >= 0.4:
            outcome = "partially_plausible"
        else:
            outcome = "implausible"

        return DimensionResult(
            dimension_name=self.name,
            hypothesis_id=hypothesis.get("hypothesis_id", ""),
            graded_outcome=outcome,
            score=round(completeness, 3),
            interpretation=f"Plausibility {completeness:.2f} based on mechanism completeness",
            status="resolved",
        )


class SupportingEvidence(DimensionPlugin):
    name = "supporting_evidence"
    description = "Whether the hypothesis cites enough evidence"

    def assess(self, hypothesis, context):
        cited_entities = hypothesis.get("cited_entity_ids", [])
        cited_relationships = hypothesis.get("cited_relationship_ids", [])
        cited_events = hypothesis.get("cited_event_ids", [])

        total = len(cited_entities) + len(cited_relationships) + len(cited_events)

        # more citations = better evidence, cap at 1.0
        score = min(total / 5.0, 1.0)

        if score >= 0.8:
            outcome = "strongly_supported"
        elif score >= 0.4:
            outcome = "partially_supported"
        else:
            outcome = "weakly_supported"

        return DimensionResult(
            dimension_name=self.name,
            hypothesis_id=hypothesis.get("hypothesis_id", ""),
            graded_outcome=outcome,
            score=round(score, 3),
            interpretation=f"Evidence score {score:.2f} ({total} citations)",
            status="resolved",
        )


class ExplanationQuality(DimensionPlugin):
    name = "explanation_quality"
    description = "How well the hypothesis explains its reasoning"
    requires_expert_review = True

    def assess(self, hypothesis, context):
        explanation = hypothesis.get("explanation", "")
        mechanism_desc = hypothesis.get("mechanism_description", "")

        # use word count as a rough proxy for explanation quality
        text = explanation or mechanism_desc
        word_count = len(text.split()) if text else 0

        if word_count >= 50:
            score = 0.9
            outcome = "high_quality"
        elif word_count >= 20:
            score = 0.6
            outcome = "adequate_quality"
        elif word_count > 0:
            score = 0.3
            outcome = "low_quality"
        else:
            score = 0.0
            outcome = "no_explanation"

        return DimensionResult(
            dimension_name=self.name,
            hypothesis_id=hypothesis.get("hypothesis_id", ""),
            graded_outcome=outcome,
            score=round(score, 3),
            interpretation=f"Explanation quality {score:.2f} ({word_count} words)",
            status="pending_expert_review" if self.requires_expert_review else "resolved",
        )


class ConfidenceAssessment(DimensionPlugin):
    name = "confidence_assessment"
    description = "Whether the stated confidence matches actual correctness"

    def assess(self, hypothesis, context):
        stated_confidence = hypothesis.get("confidence", 0.5)
        ground_truth = context.get("ground_truth", {})

        # compare stated confidence vs how correct the hypothesis actually is
        gt_roles = set(ground_truth.get("entity_roles", []))
        h_roles = set(hypothesis.get("entity_roles", []))
        actual_correctness = len(gt_roles & h_roles) / max(len(gt_roles | h_roles), 1)

        calibration_error = abs(stated_confidence - actual_correctness)
        score = 1.0 - calibration_error

        if score >= 0.8:
            outcome = "well_calibrated"
        elif score >= 0.5:
            outcome = "moderately_calibrated"
        else:
            outcome = "poorly_calibrated"

        return DimensionResult(
            dimension_name=self.name,
            hypothesis_id=hypothesis.get("hypothesis_id", ""),
            graded_outcome=outcome,
            score=round(score, 3),
            interpretation=f"Calibration {score:.2f} (stated={stated_confidence}, actual={actual_correctness:.2f})",
            status="resolved",
        )


# all 6 dimensions
ALL_DIMENSIONS = [
    FraudCorrectness,
    Novelty,
    BehavioralPlausibility,
    SupportingEvidence,
    ExplanationQuality,
    ConfidenceAssessment,
]


class DimensionRegistry:
    # Keeps track of which dimensions are active.

    def __init__(self):
        self._plugins = {}

    def register(self, plugin):
        self._plugins[plugin.name] = plugin

    def register_all_defaults(self):
        for cls in ALL_DIMENSIONS:
            self.register(cls())

    def get(self, name):
        return self._plugins.get(name)

    def all_plugins(self):
        return list(self._plugins.values())

    def dimension_names(self):
        return list(self._plugins.keys())

    def count(self):
        return len(self._plugins)

    def validate_manifest(self, required_dimensions):
        # Returns list of required dimensions that aren't registered.
        return [d for d in required_dimensions if d not in self._plugins]


class MetricsFramework:
    # Main entry point - evaluates hypotheses across all dimensions.

    def __init__(self):
        self.registry = DimensionRegistry()
        self.registry.register_all_defaults()

    def evaluate(self, hypothesis, context, episode_id):
        # Score a hypothesis on all registered dimensions.
        results = []
        for plugin in self.registry.all_plugins():
            result = plugin.assess(hypothesis, context)
            results.append(result)

        return EvaluationProfile(
            profile_id=f"{episode_id}_{hypothesis.get('hypothesis_id', 'unknown')}",
            hypothesis_id=hypothesis.get("hypothesis_id", ""),
            episode_id=episode_id,
            dimension_results=results,
        )

    def evaluate_single_dimension(self, dimension_name, hypothesis, context, episode_id):
        # Score just one dimension. Returns None if dimension doesn't exist.
        plugin = self.registry.get(dimension_name)
        if plugin is None:
            return None
        return plugin.assess(hypothesis, context)

    def aggregate_episode(self, profiles, episode_id):
        # Get per-dimension min/mean/max across multiple hypotheses.
        if not profiles:
            return {"episode_id": episode_id, "dimensions": {}, "hypothesis_count": 0}

        dimension_scores = {}
        for profile in profiles:
            for result in profile.dimension_results:
                dimension_scores.setdefault(result.dimension_name, [])
                if result.score is not None:
                    dimension_scores[result.dimension_name].append(result.score)

        summary = {}
        for dim_name, scores in dimension_scores.items():
            if scores:
                summary[dim_name] = {
                    "mean": round(sum(scores) / len(scores), 3),
                    "min": round(min(scores), 3),
                    "max": round(max(scores), 3),
                    "count": len(scores),
                }

        return {
            "episode_id": episode_id,
            "dimensions": summary,
            "hypothesis_count": len(profiles),
        }
