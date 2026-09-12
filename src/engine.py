# Execution engine - runs evaluation episodes from start to finish.
# Lifecycle: init -> assign -> deliver -> submit -> validate -> evaluate -> archive

import uuid

from src.models import (
    EpisodeState, VersionManifest, Submission, DeliveredContent,
    EpisodeRecord,
)
from src.graph import GraphEnvironment
from src.knowledge import KnowledgeSpace
from src.metrics import MetricsFramework


# ---- Exceptions ----

class EpisodeTerminated(Exception):
    # Raised when something goes wrong enough to kill the episode.
    def __init__(self, record, reason):
        self.record = record
        self.reason = reason
        super().__init__(reason)


def route_failure(record, stage, category, detail, scope="hypothesis", terminate=False):
    # Log a failure. If terminate=True, move to FAILED and raise.
    record.record_failure(stage, category, detail, scope)
    if terminate:
        record.transition(EpisodeState.FAILED)
        raise EpisodeTerminated(record, detail)


# ---- Submission receiver ----

class SubmissionReceiver:
    # Holds submissions until we're ready to process them.

    def __init__(self):
        self._pending = {}

    def receive(self, submission):
        self._pending.setdefault(submission.episode_id, []).append(submission)

    def collect(self, episode_id):
        return self._pending.pop(episode_id, [])

    def has_pending(self, episode_id):
        return bool(self._pending.get(episode_id))


# ---- Pipeline: validation + evaluation ----

def _validate_submission(record, submission):
    if not submission.hypotheses:
        route_failure(record, "submission_validation", "incomplete_submission",
                      "Submission contains no hypotheses", scope="submission", terminate=True)
        return False

    if submission.episode_id != record.episode_id:
        route_failure(record, "submission_validation", "invalid_submission",
                      f"Episode ID mismatch: {submission.episode_id} != {record.episode_id}",
                      scope="submission", terminate=True)
        return False

    record.submissions.append(submission.model_dump())
    return True


def _verify_structure(record, hypothesis):
    # Check that required fields are present.
    required = ["hypothesis_id", "proposed_mechanism", "cited_evidence"]
    missing = [f for f in required if f not in hypothesis or not hypothesis[f]]
    if missing:
        route_failure(record, "structural_verification", "invalid_submission",
                      f"Hypothesis missing fields: {missing}")
        record.excluded_hypotheses.append(hypothesis)
        return False
    return True


def _check_observation_consistency(record, hypothesis, graph_env):
    # Make sure cited evidence was actually visible to the candidate.
    cited_entities = hypothesis.get("cited_entity_ids", [])
    cited_relationships = hypothesis.get("cited_relationship_ids", [])
    cited_events = hypothesis.get("cited_event_ids", [])

    violations = graph_env.check_observation_consistency(
        record.episode_id,
        cited_entity_ids=cited_entities or None,
        cited_relationship_ids=cited_relationships or None,
        cited_event_ids=cited_events or None,
    )
    if violations:
        route_failure(record, "observation_consistency", "inconsistent_output",
                      f"Evidence outside visible boundary: {violations}")
        record.excluded_hypotheses.append(hypothesis)
        return False
    return True


def _lookup_knowledge(record, hypothesis, knowledge_space):
    # Check how similar this hypothesis is to known fraud patterns.
    result = knowledge_space.lookup(
        mechanism_description=hypothesis.get("proposed_mechanism", ""),
        entity_roles=hypothesis.get("mechanism_entity_roles",
                                    hypothesis.get("entity_roles", [])),
        relationship_flows=hypothesis.get("mechanism_relationships",
                                          hypothesis.get("relationship_flows", [])),
        version=record.manifest.knowledge_space_version if record.manifest else None,
    )
    lookup_data = {
        "nearest_families": [f.family_id for f in result.nearest_families],
        "max_structural_proximity": result.position.max_structural_proximity,
        "max_behavioral_similarity": result.position.max_behavioral_similarity,
    }
    record.knowledge_lookups[hypothesis.get("hypothesis_id", "")] = lookup_data
    return lookup_data


def _run_pipeline(record, submission, graph_env, knowledge_space, metrics):
    # Full pipeline for one submission.
    if not _validate_submission(record, submission):
        return {}

    # get ground truth from the instance metadata if it's there
    instance = graph_env.get_instance_for_episode(record.episode_id)
    ground_truth = {}
    if instance and instance.metadata:
        ground_truth = instance.metadata.get("ground_truth", {})

    for hyp in submission.hypotheses:
        if not _verify_structure(record, hyp):
            continue
        if not _check_observation_consistency(record, hyp, graph_env):
            continue

        record.validated_hypotheses.append(hyp)
        ks_context = _lookup_knowledge(record, hyp, knowledge_space)

        context = {
            "knowledge_space_lookup": ks_context,
            "ground_truth": ground_truth,
        }
        profile = metrics.evaluate(hyp, context, record.episode_id)
        record.evaluation_results.append(profile)

    result = metrics.aggregate_episode(record.evaluation_results, record.episode_id)
    record.aggregated_results = result
    return result


# ---- Main class ----

class ExecutionEngine:
    # Drives evaluation episodes through their lifecycle.

    def __init__(self, graph_env, knowledge_space, metrics):
        self._graph_env = graph_env
        self._knowledge_space = knowledge_space
        self._metrics = metrics
        self._receiver = SubmissionReceiver()
        self._episodes = {}

    @property
    def receiver(self):
        return self._receiver

    def initiate_episode(self, manifest, candidate_id, adapter_id, episode_id=None):
        # Start a new episode. Returns the episode_id.
        eid = episode_id or str(uuid.uuid4())
        record = EpisodeRecord(episode_id=eid)

        record.manifest = manifest
        record.transition(EpisodeState.INITIALIZED)

        record.candidate_id = candidate_id
        record.adapter_id = adapter_id
        record.transition(EpisodeState.REGISTERED)

        self._episodes[eid] = record
        return eid

    def assign_and_deliver(self, episode_id, instance_id=None, session_id=None):
        # Assign an environment instance and deliver visible info to the candidate.
        record = self._get_record(episode_id)

        assigned = self._graph_env.assign(
            record.episode_id, instance_id=instance_id, session_id=session_id
        )
        if assigned is None:
            route_failure(record, "environment_assignment", "protocol_violation",
                          "No instance could be assigned", scope="episode", terminate=True)
            return None

        record.assigned_instance_ids.append(assigned)
        record.transition(EpisodeState.ENVIRONMENT_ASSIGNED)

        # deliver visible info
        visible = self._graph_env.deliver(record.episode_id)
        if visible is None:
            route_failure(record, "input_delivery", "protocol_violation",
                          "No visible information available", scope="episode", terminate=True)
            return None

        delivery = DeliveredContent(
            delivery_id=f"{record.episode_id}_d{len(record.deliveries)}",
            episode_id=record.episode_id,
            environment_instance_id=record.assigned_instance_ids[-1],
            visible_entities=[e.model_dump() for e in visible.entities],
            visible_relationships=[r.model_dump() for r in visible.relationships],
            visible_events=[ev.model_dump() for ev in visible.temporal_events],
            schema_version=record.manifest.environment_schema_version if record.manifest else "unknown",
        )
        record.deliveries.append(delivery.model_dump())
        record.transition(EpisodeState.AWAITING_SUBMISSION)
        return delivery

    def submit(self, submission):
        # Receive a submission from a candidate system.
        self._receiver.receive(submission)

    def evaluate(self, episode_id):
        # Run validation + evaluation on all pending submissions, then archive.
        record = self._get_record(episode_id)
        submissions = self._receiver.collect(episode_id)

        if not submissions:
            record.transition(EpisodeState.VALIDATING)
            route_failure(record, "submission_intake", "incomplete_submission",
                          "No submissions received", scope="episode", terminate=True)

        record.transition(EpisodeState.VALIDATING)

        for sub in submissions:
            _run_pipeline(record, sub, self._graph_env,
                          self._knowledge_space, self._metrics)

        record.transition(EpisodeState.EVALUATING)
        record.transition(EpisodeState.REPORTING)
        record.transition(EpisodeState.ARCHIVED)

        return record.aggregated_results or {}

    def get_record(self, episode_id):
        return self._episodes.get(episode_id)

    def _get_record(self, episode_id):
        record = self._episodes.get(episode_id)
        if record is None:
            raise ValueError(f"No episode with id {episode_id}")
        return record
