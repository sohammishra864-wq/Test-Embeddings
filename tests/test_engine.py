# Tests for the execution engine

import pytest

from src.models import (
    EpisodeState,
    VersionManifest,
    EnvironmentInstance,
    Entity,
    Relationship,
    TemporalEvent,
    Submission,
    EpisodeRecord,
)
from src.engine import ExecutionEngine, EpisodeTerminated, route_failure, SubmissionReceiver
from src.graph import GraphEnvironment
from src.knowledge import KnowledgeSpace
from src.metrics import MetricsFramework


# ---- helpers ----

def make_manifest():
    return VersionManifest(
        manifest_id="m1",
        name="v1.0-test",
        environment_schema_version="1.0",
        knowledge_space_version="1.0",
        dimension_registry_version="1.0",
        protocol_version="1.0",
        is_active=True,
    )


def make_instance(instance_id="inst_1"):
    return EnvironmentInstance(
        instance_id=instance_id,
        schema_version="1.0",
        entities=[
            Entity(entity_id="e1", entity_type="account",
                   attributes={"account_type": "checking", "currency": "USD"},
                   observation_category="visible"),
            Entity(entity_id="e2", entity_type="person",
                   attributes={"name": "Bob"},
                   observation_category="visible"),
        ],
        relationships=[
            Relationship(relationship_id="r1", relationship_type="owns",
                         source_entity_id="e2", target_entity_id="e1",
                         attributes={}, observation_category="visible"),
        ],
        temporal_events=[
            TemporalEvent(event_id="ev1", event_type="transaction",
                          timestamp="2024-01-01T00:00:00Z",
                          source_entity_id="e1", attributes={},
                          observation_category="visible"),
        ],
    )


def make_submission(episode_id="ep1"):
    return Submission(
        submission_id="sub_1",
        episode_id=episode_id,
        candidate_system_id="candidate_A",
        hypotheses=[
            {
                "hypothesis_id": "h1",
                "proposed_mechanism": "Shell company layering",
                "mechanism_entity_roles": ["shell", "beneficiary"],
                "mechanism_relationships": ["ownership"],
                "point_of_deviation": "concealment",
                "cited_evidence": ["e1", "r1"],
                "cited_entity_ids": ["e1", "e2"],
                "cited_relationship_ids": ["r1"],
                "cited_event_ids": ["ev1"],
                "explanation": "Layering through shell entities",
                "confidence": 0.8,
            }
        ],
    )


def make_engine():
    graph_env = GraphEnvironment()
    ks = KnowledgeSpace()
    metrics = MetricsFramework()
    engine = ExecutionEngine(graph_env, ks, metrics)
    return engine, graph_env, ks, metrics


# ---- episode record tests ----

class TestEpisodeRecord:
    def test_initial_state(self):
        record = EpisodeRecord(episode_id="ep1")
        assert record.state == EpisodeState.IDLE

    def test_transition_logs(self):
        record = EpisodeRecord(episode_id="ep1")
        record.transition(EpisodeState.INITIALIZED)
        assert record.state == EpisodeState.INITIALIZED
        assert "idle -> initialized" in record.log

    def test_failure_recording(self):
        record = EpisodeRecord(episode_id="ep1")
        record.record_failure("test_stage", "test_category", "something broke")
        assert len(record.failures) == 1
        assert record.failures[0].stage == "test_stage"


# ---- exception router tests ----

class TestExceptionRouter:
    def test_non_terminating_failure(self):
        record = EpisodeRecord(episode_id="ep1")
        route_failure(record, "test", "missing_information", "optional field absent")
        assert len(record.failures) == 1
        assert record.state == EpisodeState.IDLE

    def test_terminating_failure(self):
        record = EpisodeRecord(episode_id="ep1")
        with pytest.raises(EpisodeTerminated):
            route_failure(record, "test", "protocol_violation",
                          "fatal error", scope="episode", terminate=True)
        assert record.state == EpisodeState.FAILED


# ---- submission receiver tests ----

class TestSubmissionReceiver:
    def test_receive_and_collect(self):
        receiver = SubmissionReceiver()
        sub = make_submission("ep1")
        receiver.receive(sub)
        assert receiver.has_pending("ep1")
        collected = receiver.collect("ep1")
        assert len(collected) == 1
        assert not receiver.has_pending("ep1")

    def test_collect_empty(self):
        receiver = SubmissionReceiver()
        assert receiver.collect("nonexistent") == []


# ---- lifecycle tests ----

class TestLifecycle:
    def test_initiate_episode(self):
        engine, *_ = make_engine()
        eid = engine.initiate_episode(make_manifest(), "cand_1", "adapter_1",
                                       episode_id="ep1")
        assert eid == "ep1"
        record = engine.get_record("ep1")
        assert record.state == EpisodeState.REGISTERED
        assert record.candidate_id == "cand_1"
        assert record.manifest.name == "v1.0-test"

    def test_assign_and_deliver(self):
        engine, graph_env, *_ = make_engine()
        instance = make_instance()
        graph_env.ingest(instance)
        eid = engine.initiate_episode(make_manifest(), "c1", "a1", episode_id="ep1")

        delivery = engine.assign_and_deliver(eid, instance_id="inst_1")
        assert delivery is not None
        assert delivery.episode_id == "ep1"
        assert len(delivery.visible_entities) > 0

        record = engine.get_record(eid)
        assert record.state == EpisodeState.AWAITING_SUBMISSION

    def test_assign_nonexistent_instance_terminates(self):
        engine, *_ = make_engine()
        eid = engine.initiate_episode(make_manifest(), "c1", "a1", episode_id="ep1")
        with pytest.raises(EpisodeTerminated):
            engine.assign_and_deliver(eid, instance_id="no_such_instance")


# ---- pipeline tests ----

class TestPipeline:
    def test_submission_episode_mismatch_terminates(self):
        engine, graph_env, *_ = make_engine()
        instance = make_instance()
        graph_env.ingest(instance)
        eid = engine.initiate_episode(make_manifest(), "c1", "a1", episode_id="ep1")
        engine.assign_and_deliver(eid, instance_id="inst_1")

        bad_sub = Submission(
            submission_id="sub_bad", episode_id="wrong_episode",
            candidate_system_id="c1", hypotheses=[{"hypothesis_id": "h1"}]
        )
        engine.submit(bad_sub)
        with pytest.raises(EpisodeTerminated):
            engine.evaluate(eid)

    def test_empty_hypotheses_terminates(self):
        engine, graph_env, *_ = make_engine()
        instance = make_instance()
        graph_env.ingest(instance)
        eid = engine.initiate_episode(make_manifest(), "c1", "a1", episode_id="ep1")
        engine.assign_and_deliver(eid, instance_id="inst_1")

        empty_sub = Submission(
            submission_id="sub_empty", episode_id="ep1",
            candidate_system_id="c1", hypotheses=[]
        )
        engine.submit(empty_sub)
        with pytest.raises(EpisodeTerminated):
            engine.evaluate(eid)

    def test_malformed_hypothesis_excluded(self):
        engine, graph_env, *_ = make_engine()
        instance = make_instance()
        graph_env.ingest(instance)
        eid = engine.initiate_episode(make_manifest(), "c1", "a1", episode_id="ep1")
        engine.assign_and_deliver(eid, instance_id="inst_1")

        sub = Submission(
            submission_id="sub_1", episode_id="ep1",
            candidate_system_id="c1",
            hypotheses=[
                {"hypothesis_id": "h_bad"},
                {
                    "hypothesis_id": "h_good",
                    "proposed_mechanism": "test mech",
                    "cited_evidence": ["e1"],
                    "cited_entity_ids": ["e1"],
                    "cited_relationship_ids": ["r1"],
                    "cited_event_ids": ["ev1"],
                    "mechanism_entity_roles": ["role1"],
                    "mechanism_relationships": ["rel1"],
                },
            ],
        )
        engine.submit(sub)
        result = engine.evaluate(eid)

        record = engine.get_record(eid)
        assert len(record.excluded_hypotheses) == 1
        assert record.excluded_hypotheses[0]["hypothesis_id"] == "h_bad"
        assert len(record.evaluation_results) == 1


# ---- full episode flow tests ----

class TestFullEpisodeFlow:
    def test_happy_path(self):
        engine, graph_env, ks, metrics = make_engine()
        instance = make_instance()
        graph_env.ingest(instance)

        eid = engine.initiate_episode(make_manifest(), "cand_1", "adapter_1",
                                       episode_id="ep1")
        delivery = engine.assign_and_deliver(eid, instance_id="inst_1")
        assert delivery is not None

        engine.submit(make_submission("ep1"))
        result = engine.evaluate(eid)

        record = engine.get_record(eid)
        assert record.state == EpisodeState.ARCHIVED
        assert record.aggregated_results is not None
        assert result["hypothesis_count"] == 1

    def test_no_submissions_terminates(self):
        engine, graph_env, *_ = make_engine()
        instance = make_instance()
        graph_env.ingest(instance)

        eid = engine.initiate_episode(make_manifest(), "c1", "a1", episode_id="ep1")
        engine.assign_and_deliver(eid, instance_id="inst_1")

        with pytest.raises(EpisodeTerminated):
            engine.evaluate(eid)

    def test_observation_inconsistency_excludes_hypothesis(self):
        engine, graph_env, *_ = make_engine()
        instance = make_instance()
        graph_env.ingest(instance)

        eid = engine.initiate_episode(make_manifest(), "c1", "a1", episode_id="ep1")
        engine.assign_and_deliver(eid, instance_id="inst_1")

        sub = Submission(
            submission_id="sub_1", episode_id="ep1",
            candidate_system_id="c1",
            hypotheses=[{
                "hypothesis_id": "h_oob",
                "proposed_mechanism": "test",
                "cited_evidence": ["e_nonexistent"],
                "cited_entity_ids": ["e_nonexistent"],
                "cited_relationship_ids": [],
                "cited_event_ids": [],
                "mechanism_entity_roles": ["x"],
                "mechanism_relationships": ["y"],
            }],
        )
        engine.submit(sub)
        engine.evaluate(eid)

        record = engine.get_record(eid)
        assert len(record.excluded_hypotheses) == 1
        assert record.failures[0].category == "inconsistent_output"

    def test_episode_isolation(self):
        engine, graph_env, *_ = make_engine()
        instance = make_instance("inst_a")
        graph_env.ingest(instance)

        eid1 = engine.initiate_episode(make_manifest(), "c1", "a1", episode_id="ep1")
        eid2 = engine.initiate_episode(make_manifest(), "c2", "a2", episode_id="ep2")

        r1 = engine.get_record(eid1)
        r2 = engine.get_record(eid2)
        assert r1.candidate_id == "c1"
        assert r2.candidate_id == "c2"
        assert r1.episode_id != r2.episode_id

    def test_lifecycle_log_completeness(self):
        engine, graph_env, *_ = make_engine()
        instance = make_instance()
        graph_env.ingest(instance)

        eid = engine.initiate_episode(make_manifest(), "c1", "a1", episode_id="ep1")
        engine.assign_and_deliver(eid, instance_id="inst_1")
        engine.submit(make_submission("ep1"))
        engine.evaluate(eid)

        record = engine.get_record(eid)
        assert len(record.log) >= 6
        assert "idle -> initialized" in record.log[0]
        assert "reporting -> archived" in record.log[-1]
