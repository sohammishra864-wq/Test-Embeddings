# All the data models for EvolveBench.

from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


# ---- Environment Instance stuff ----

class Entity(BaseModel):
    entity_id: str
    entity_type: str
    attributes: dict = Field(default_factory=dict)
    created_at: Optional[str] = None
    deactivated_at: Optional[str] = None
    observation_category: str = "visible"


class Relationship(BaseModel):
    relationship_id: str
    relationship_type: str
    source_entity_id: str
    target_entity_id: str
    attributes: dict = Field(default_factory=dict)
    created_at: Optional[str] = None
    deactivated_at: Optional[str] = None
    observation_category: str = "visible"


class TemporalEvent(BaseModel):
    event_id: str
    event_type: str
    timestamp: str
    source_entity_id: Optional[str] = None
    target_entity_id: Optional[str] = None
    attributes: dict = Field(default_factory=dict)
    observation_category: str = "visible"


class EnvironmentInstance(BaseModel):
    # The full environment instance - entities, relationships, and events.
    instance_id: str
    schema_version: str
    session_id: Optional[str] = None
    snapshot_index: int = 0
    entities: list[Entity]
    relationships: list[Relationship]
    temporal_events: list[TemporalEvent] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)


# ---- Submission and Delivery ----

class DeliveredContent(BaseModel):
    # What gets sent to the candidate system (only the visible info).
    delivery_id: str
    episode_id: str
    environment_instance_id: str
    visible_entities: list[dict]
    visible_relationships: list[dict]
    visible_events: list[dict] = Field(default_factory=list)
    schema_version: str
    delivery_timestamp: Optional[str] = None


class Submission(BaseModel):
    # What the candidate system sends back.
    submission_id: str
    episode_id: str
    candidate_system_id: str
    hypotheses: list[dict]
    submission_timestamp: Optional[str] = None
    metadata: dict = Field(default_factory=dict)


# ---- Evaluation results ----

class DimensionResult(BaseModel):
    # Result from scoring one hypothesis on one dimension.
    dimension_name: str
    hypothesis_id: str
    graded_outcome: str
    score: Optional[float] = None
    interpretation: str
    status: str = "resolved"


class EvaluationProfile(BaseModel):
    # All dimension scores for one hypothesis. Never collapsed into a single number.
    profile_id: str
    hypothesis_id: str
    episode_id: str
    dimension_results: list[DimensionResult]

    @property
    def dimension_count(self):
        return len(self.dimension_results)

    @property
    def all_resolved(self):
        return all(r.status == "resolved" for r in self.dimension_results)


# ---- Lifecycle ----

class EpisodeState(str, Enum):
    IDLE = "idle"
    INITIALIZED = "initialized"
    REGISTERED = "registered"
    ENVIRONMENT_ASSIGNED = "environment_assigned"
    AWAITING_SUBMISSION = "awaiting_submission"
    VALIDATING = "validating"
    EVALUATING = "evaluating"
    REPORTING = "reporting"
    ARCHIVED = "archived"
    FAILED = "failed"


class VersionManifest(BaseModel):
    manifest_id: str
    name: str
    environment_schema_version: str
    knowledge_space_version: str
    dimension_registry_version: str
    protocol_version: str
    is_active: bool = False


# ---- Episode tracking ----

class FailureRecord(BaseModel):
    stage: str
    category: str
    detail: str
    scope: str = "hypothesis"


class EpisodeRecord(BaseModel):
    # Tracks everything that happens during one evaluation episode.
    episode_id: str
    state: EpisodeState = EpisodeState.IDLE

    manifest: Optional[VersionManifest] = None
    candidate_id: Optional[str] = None
    adapter_id: Optional[str] = None

    assigned_instance_ids: list[str] = Field(default_factory=list)
    deliveries: list[dict] = Field(default_factory=list)
    submissions: list[dict] = Field(default_factory=list)

    validated_hypotheses: list[dict] = Field(default_factory=list)
    excluded_hypotheses: list[dict] = Field(default_factory=list)

    knowledge_lookups: dict[str, dict] = Field(default_factory=dict)
    evaluation_results: list[EvaluationProfile] = Field(default_factory=list)
    aggregated_results: Optional[dict] = None

    failures: list[FailureRecord] = Field(default_factory=list)
    log: list[str] = Field(default_factory=list)

    def transition(self, new_state):
        self.log.append(f"{self.state.value} -> {new_state.value}")
        self.state = new_state

    def record_failure(self, stage, category, detail, scope="hypothesis"):
        self.failures.append(FailureRecord(
            stage=stage, category=category, detail=detail, scope=scope
        ))


# ---- Knowledge Space ----

class KnowledgeItem(BaseModel):
    # A documented fraud mechanism in the knowledge base.
    item_id: str
    mechanism_description: str
    entity_roles: list[str]
    relationship_flows: list[str]
    point_of_deviation: str
    provenance: str
    version_admitted: str
