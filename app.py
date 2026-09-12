# Flask frontend for EvolveBench. Run: python app.py

import json
from flask import Flask, render_template, request, jsonify
from src.metrics import MetricsFramework
from src.engine import ExecutionEngine, EpisodeTerminated
from src.graph import GraphEnvironment
from src.knowledge import KnowledgeSpace
from src.models import (
    EnvironmentInstance, Entity, Relationship, TemporalEvent,
    Submission, VersionManifest,
)

app = Flask(__name__)
mf = MetricsFramework()


DEFAULT_HYPOTHESIS = {
    "hypothesis_id": "h1",
    "mechanism_description": "Shell company layering to conceal beneficial ownership through nominee directors",
    "entity_roles": ["nominee_director", "shell_company", "beneficial_owner"],
    "relationship_flows": ["ownership_chain", "fund_transfer"],
    "point_of_deviation": "concealment of beneficial ownership",
    "cited_entity_ids": ["e1", "e2", "e3"],
    "cited_relationship_ids": ["r1", "r2"],
    "cited_event_ids": ["ev1"],
    "explanation": "The mechanism operates by layering shell companies across multiple jurisdictions, appointing nominee directors to obscure the true beneficial owner. Fund transfers flow through these intermediary entities in a pattern designed to break the audit trail and evade regulatory detection.",
    "confidence": 0.8,
}

DEFAULT_CONTEXT = {
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

DEFAULT_EPISODE_CONFIG = {
    "manifest": {
        "manifest_id": "m1",
        "name": "v1.0-demo",
        "environment_schema_version": "1.0",
        "knowledge_space_version": "1.0",
        "dimension_registry_version": "1.0",
        "protocol_version": "1.0",
        "is_active": True,
    },
    "candidate_id": "candidate_A",
    "adapter_id": "adapter_1",
    "instance": {
        "instance_id": "inst_1",
        "schema_version": "1.0",
        "entities": [
            {"entity_id": "e1", "entity_type": "account",
             "attributes": {"account_type": "checking", "currency": "USD"},
             "observation_category": "visible"},
            {"entity_id": "e2", "entity_type": "person",
             "attributes": {"name": "Bob"},
             "observation_category": "visible"},
            {"entity_id": "e3", "entity_type": "company",
             "attributes": {"name": "ShellCorp Ltd", "jurisdiction": "BVI"},
             "observation_category": "visible"},
        ],
        "relationships": [
            {"relationship_id": "r1", "relationship_type": "owns",
             "source_entity_id": "e2", "target_entity_id": "e1",
             "attributes": {}, "observation_category": "visible"},
            {"relationship_id": "r2", "relationship_type": "controls",
             "source_entity_id": "e2", "target_entity_id": "e3",
             "attributes": {}, "observation_category": "visible"},
        ],
        "temporal_events": [
            {"event_id": "ev1", "event_type": "transaction",
             "timestamp": "2024-01-01T00:00:00Z",
             "source_entity_id": "e1", "attributes": {},
             "observation_category": "visible"},
        ],
        "metadata": {
            "ground_truth": {
                "entity_roles": ["nominee_director", "shell_company", "beneficial_owner"],
                "relationship_flows": ["ownership_chain", "fund_transfer"],
                "point_of_deviation": "concealment of beneficial ownership",
            }
        },
    },
    "submission": {
        "submission_id": "sub_1",
        "candidate_system_id": "candidate_A",
        "hypotheses": [
            {
                "hypothesis_id": "h1",
                "proposed_mechanism": "Shell company layering to conceal beneficial ownership",
                "mechanism_entity_roles": ["shell", "beneficiary"],
                "mechanism_relationships": ["ownership"],
                "point_of_deviation": "concealment",
                "cited_evidence": ["e1", "r1"],
                "cited_entity_ids": ["e1", "e2", "e3"],
                "cited_relationship_ids": ["r1", "r2"],
                "cited_event_ids": ["ev1"],
                "explanation": "Layering through shell entities across jurisdictions",
                "confidence": 0.8,
            }
        ],
    },
}


@app.route("/")
def index():
    return render_template(
        "index.html",
        default_hypothesis=json.dumps(DEFAULT_HYPOTHESIS, indent=2),
        default_context=json.dumps(DEFAULT_CONTEXT, indent=2),
        default_episode_config=json.dumps(DEFAULT_EPISODE_CONFIG, indent=2),
        dimensions=mf.registry.dimension_names(),
    )


@app.route("/api/evaluate", methods=["POST"])
def evaluate():
    data = request.get_json()
    hypothesis = data.get("hypothesis", {})
    context = data.get("context", {})
    episode_id = data.get("episode_id", "ep-1")

    profile = mf.evaluate(hypothesis, context, episode_id)
    results = []
    for r in profile.dimension_results:
        results.append({
            "dimension": r.dimension_name,
            "outcome": r.graded_outcome,
            "score": r.score,
            "interpretation": r.interpretation,
            "status": r.status,
        })

    return jsonify({
        "profile_id": profile.profile_id,
        "episode_id": profile.episode_id,
        "hypothesis_id": profile.hypothesis_id,
        "dimension_count": profile.dimension_count,
        "all_resolved": profile.all_resolved,
        "results": results,
    })


@app.route("/api/evaluate_single", methods=["POST"])
def evaluate_single():
    data = request.get_json()
    dimension = data.get("dimension", "")
    hypothesis = data.get("hypothesis", {})
    context = data.get("context", {})

    result = mf.evaluate_single_dimension(dimension, hypothesis, context, "ep-1")
    if result is None:
        return jsonify({"error": f"Dimension '{dimension}' not found"}), 404

    return jsonify({
        "dimension": result.dimension_name,
        "outcome": result.graded_outcome,
        "score": result.score,
        "interpretation": result.interpretation,
        "status": result.status,
    })


@app.route("/api/aggregate", methods=["POST"])
def aggregate():
    data = request.get_json()
    hypotheses = data.get("hypotheses", [])
    context = data.get("context", {})
    episode_id = data.get("episode_id", "ep-1")

    profiles = [mf.evaluate(h, context, episode_id) for h in hypotheses]
    summary = mf.aggregate_episode(profiles, episode_id)
    return jsonify(summary)


@app.route("/api/run_episode", methods=["POST"])
def run_episode():
    data = request.get_json()

    graph_env = GraphEnvironment()
    ks = KnowledgeSpace()
    metrics = MetricsFramework()
    engine = ExecutionEngine(graph_env, ks, metrics)

    try:
        inst_data = data["instance"]
        instance = EnvironmentInstance(
            instance_id=inst_data["instance_id"],
            schema_version=inst_data["schema_version"],
            entities=[Entity(**e) for e in inst_data.get("entities", [])],
            relationships=[Relationship(**r) for r in inst_data.get("relationships", [])],
            temporal_events=[TemporalEvent(**t) for t in inst_data.get("temporal_events", [])],
            metadata=inst_data.get("metadata", {}),
        )
        ingest_result = graph_env.ingest(instance)
        if not ingest_result.admitted:
            return jsonify({"error": f"Instance rejected: {ingest_result.errors}",
                            "stage": "ingest"}), 400

        manifest = VersionManifest(**data["manifest"])
        eid = engine.initiate_episode(
            manifest, data.get("candidate_id", "unknown"),
            data.get("adapter_id", "unknown"), episode_id="ep-1",
        )

        delivery = engine.assign_and_deliver(eid, instance_id=instance.instance_id)

        sub_data = data["submission"]
        sub = Submission(
            submission_id=sub_data["submission_id"],
            episode_id=eid,
            candidate_system_id=sub_data["candidate_system_id"],
            hypotheses=sub_data["hypotheses"],
        )
        engine.submit(sub)
        result = engine.evaluate(eid)

        record = engine.get_record(eid)

        dim_results = []
        for profile in record.evaluation_results:
            for r in profile.dimension_results:
                dim_results.append({
                    "hypothesis_id": profile.hypothesis_id,
                    "dimension": r.dimension_name,
                    "outcome": r.graded_outcome,
                    "score": r.score,
                    "interpretation": r.interpretation,
                    "status": r.status,
                })

        return jsonify({
            "episode_id": eid,
            "final_state": record.state.value,
            "lifecycle_log": record.log,
            "validated_count": len(record.validated_hypotheses),
            "excluded_count": len(record.excluded_hypotheses),
            "failures": [f.model_dump() for f in record.failures],
            "dimension_results": dim_results,
            "aggregation": result,
            "delivery_entity_count": len(delivery.visible_entities) if delivery else 0,
        })

    except EpisodeTerminated as e:
        record = e.record
        return jsonify({
            "episode_id": record.episode_id,
            "final_state": record.state.value,
            "lifecycle_log": record.log,
            "failures": [f.model_dump() for f in record.failures],
            "error": str(e),
        }), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("EvolveBench UI: http://127.0.0.1:5000")
    app.run(debug=True, port=5000)
