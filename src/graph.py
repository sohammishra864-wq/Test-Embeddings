# Graph environment for storing and validating financial entity graphs.
# Handles ingestion, validation, integrity checks, and observation filtering.

from dataclasses import dataclass, field

from src.models import EnvironmentInstance


# ---- Schema: allowed entity and relationship types ----

ENTITY_TYPES = {
    "person": {
        "required_attributes": ["name"],
        "optional_attributes": ["date_of_birth", "jurisdiction", "nationality"],
    },
    "company": {
        "required_attributes": ["name", "jurisdiction"],
        "optional_attributes": ["registration_number", "incorporation_date", "company_type"],
    },
    "account": {
        "required_attributes": ["account_type", "currency"],
        "optional_attributes": ["institution", "opened_at", "closed_at", "balance"],
    },
    "transaction": {
        "required_attributes": ["amount", "currency", "timestamp"],
        "optional_attributes": ["description", "reference"],
    },
    "address": {
        "required_attributes": ["country"],
        "optional_attributes": ["city", "postal_code", "street"],
    },
    "document": {
        "required_attributes": ["document_type"],
        "optional_attributes": ["issued_at", "expires_at", "issuer"],
    },
    "shell_entity": {
        "required_attributes": ["name", "jurisdiction"],
        "optional_attributes": ["registration_number", "nominee_flag"],
    },
}

RELATIONSHIP_TYPES = {
    "owns": {
        "source_entity_types": ["person", "company", "shell_entity"],
        "target_entity_types": ["account", "company", "shell_entity"],
        "required_attributes": [],
        "cardinality": "many_to_many",
        "exclusive": False,
    },
    "controls": {
        "source_entity_types": ["person", "company"],
        "target_entity_types": ["company", "shell_entity", "account"],
        "required_attributes": [],
        "cardinality": "many_to_many",
        "exclusive": False,
    },
    "transfers_to": {
        "source_entity_types": ["account"],
        "target_entity_types": ["account"],
        "required_attributes": ["amount", "currency"],
        "cardinality": "many_to_many",
        "exclusive": False,
    },
    "registered_at": {
        "source_entity_types": ["company", "shell_entity"],
        "target_entity_types": ["address"],
        "required_attributes": [],
        "cardinality": "many_to_one",
        "exclusive": True,
    },
    "resides_at": {
        "source_entity_types": ["person"],
        "target_entity_types": ["address"],
        "required_attributes": [],
        "cardinality": "many_to_one",
        "exclusive": True,
    },
    "signatory_of": {
        "source_entity_types": ["person"],
        "target_entity_types": ["account"],
        "required_attributes": [],
        "cardinality": "many_to_many",
        "exclusive": False,
    },
    "beneficial_owner_of": {
        "source_entity_types": ["person"],
        "target_entity_types": ["company", "shell_entity"],
        "required_attributes": [],
        "cardinality": "many_to_many",
        "exclusive": False,
    },
    "documented_by": {
        "source_entity_types": ["person", "company", "shell_entity", "account"],
        "target_entity_types": ["document"],
        "required_attributes": [],
        "cardinality": "many_to_many",
        "exclusive": False,
    },
}

VALID_OBSERVATION_CATEGORIES = {
    "visible", "hidden", "ground_truth", "future", "evaluation_artifact"
}

VALID_CURRENCIES = {
    "USD", "EUR", "GBP", "CHF", "JPY", "AUD", "CAD", "HKD", "SGD", "BRL",
    "INR", "CNY", "RUB", "ZAR", "AED", "KYD", "VGB", "BMD", "BHD", "MXN",
}


# ---- Integrity checks (5 categories) ----

def check_structural(instance):
    # Every relationship must connect existing entities of the right types.
    entity_map = {e.entity_id: e.entity_type for e in instance.entities}
    errors = []

    for rel in instance.relationships:
        spec = RELATIONSHIP_TYPES.get(rel.relationship_type)
        if spec is None:
            errors.append(f"Unknown relationship type: {rel.relationship_type}")
            continue
        if rel.source_entity_id not in entity_map:
            errors.append(f"Relationship {rel.relationship_id}: source entity {rel.source_entity_id} does not exist")
            continue
        if rel.target_entity_id not in entity_map:
            errors.append(f"Relationship {rel.relationship_id}: target entity {rel.target_entity_id} does not exist")
            continue

        source_type = entity_map[rel.source_entity_id]
        target_type = entity_map[rel.target_entity_id]
        if source_type not in spec["source_entity_types"]:
            errors.append(f"Relationship {rel.relationship_id}: source type {source_type} not allowed for {rel.relationship_type}")
        if target_type not in spec["target_entity_types"]:
            errors.append(f"Relationship {rel.relationship_id}: target type {target_type} not allowed for {rel.relationship_type}")

    return errors


def check_temporal(instance):
    # No deactivation before creation, no events before entity exists.
    errors = []
    entity_created = {}

    for e in instance.entities:
        if e.created_at:
            entity_created[e.entity_id] = e.created_at
        if e.created_at and e.deactivated_at and e.deactivated_at < e.created_at:
            errors.append(f"Entity {e.entity_id}: deactivated_at ({e.deactivated_at}) before created_at ({e.created_at})")

    for rel in instance.relationships:
        if rel.created_at and rel.deactivated_at and rel.deactivated_at < rel.created_at:
            errors.append(f"Relationship {rel.relationship_id}: deactivated_at before created_at")
        if rel.created_at:
            src_created = entity_created.get(rel.source_entity_id)
            if src_created and rel.created_at < src_created:
                errors.append(f"Relationship {rel.relationship_id}: created before source entity {rel.source_entity_id}")
            tgt_created = entity_created.get(rel.target_entity_id)
            if tgt_created and rel.created_at < tgt_created:
                errors.append(f"Relationship {rel.relationship_id}: created before target entity {rel.target_entity_id}")

    for evt in instance.temporal_events:
        if evt.source_entity_id:
            src_created = entity_created.get(evt.source_entity_id)
            if src_created and evt.timestamp < src_created:
                errors.append(f"Event {evt.event_id}: timestamp before source entity {evt.source_entity_id} creation")
        if evt.target_entity_id:
            tgt_created = entity_created.get(evt.target_entity_id)
            if tgt_created and evt.timestamp < tgt_created:
                errors.append(f"Event {evt.event_id}: timestamp before target entity {evt.target_entity_id} creation")

    return errors


def check_relationship_integrity(instance):
    # Exclusive relationships can only have one active target per source.
    errors = []
    active_rels = {}

    for rel in instance.relationships:
        spec = RELATIONSHIP_TYPES.get(rel.relationship_type)
        if spec is None or not spec.get("exclusive") or rel.deactivated_at:
            continue

        key = (rel.relationship_type, rel.source_entity_id)
        active_rels.setdefault(key, []).append(rel)

    for (rel_type, source_id), rels in active_rels.items():
        if len(rels) > 1:
            errors.append(f"Exclusive relationship {rel_type}: entity {source_id} has {len(rels)} active targets (max 1)")

    return errors


def check_business(instance):
    # Basic financial sanity checks.
    errors = []
    deactivated_accounts = {
        e.entity_id for e in instance.entities
        if e.entity_type == "account" and e.deactivated_at
    }

    for evt in instance.temporal_events:
        amount = evt.attributes.get("amount")
        if amount is not None and amount < 0:
            errors.append(f"Event {evt.event_id}: negative amount {amount}")
        currency = evt.attributes.get("currency")
        if currency and currency not in VALID_CURRENCIES:
            errors.append(f"Event {evt.event_id}: unknown currency {currency}")

        if evt.source_entity_id and evt.source_entity_id in deactivated_accounts:
            src_entity = next(
                (e for e in instance.entities if e.entity_id == evt.source_entity_id), None
            )
            if src_entity and src_entity.deactivated_at and evt.timestamp >= src_entity.deactivated_at:
                errors.append(f"Event {evt.event_id}: source account {evt.source_entity_id} already deactivated")

    for rel in instance.relationships:
        if rel.relationship_type == "transfers_to":
            amount = rel.attributes.get("amount")
            if amount is not None and amount < 0:
                errors.append(f"Relationship {rel.relationship_id}: negative transfer amount")

    return errors


def check_consistency(instance):
    # No duplicate IDs.
    errors = []

    seen = set()
    for e in instance.entities:
        if e.entity_id in seen:
            errors.append(f"Duplicate entity_id: {e.entity_id}")
        seen.add(e.entity_id)

    seen = set()
    for rel in instance.relationships:
        if rel.relationship_id in seen:
            errors.append(f"Duplicate relationship_id: {rel.relationship_id}")
        seen.add(rel.relationship_id)

    seen = set()
    for evt in instance.temporal_events:
        if evt.event_id in seen:
            errors.append(f"Duplicate event_id: {evt.event_id}")
        seen.add(evt.event_id)

    return errors


def validate_schema(instance):
    # Check entity and relationship types against the schema definitions.
    errors = []

    for entity in instance.entities:
        spec = ENTITY_TYPES.get(entity.entity_type)
        if spec is None:
            errors.append(f"Entity {entity.entity_id}: unknown type {entity.entity_type}")
            continue
        for attr in spec["required_attributes"]:
            if attr not in entity.attributes:
                errors.append(f"Entity {entity.entity_id}: missing required attribute '{attr}'")
        if entity.observation_category not in VALID_OBSERVATION_CATEGORIES:
            errors.append(f"Entity {entity.entity_id}: invalid observation_category '{entity.observation_category}'")

    for rel in instance.relationships:
        spec = RELATIONSHIP_TYPES.get(rel.relationship_type)
        if spec is None:
            errors.append(f"Relationship {rel.relationship_id}: unknown type {rel.relationship_type}")
            continue
        for attr in spec["required_attributes"]:
            if attr not in rel.attributes:
                errors.append(f"Relationship {rel.relationship_id}: missing required attribute '{attr}'")
        if rel.observation_category not in VALID_OBSERVATION_CATEGORIES:
            errors.append(f"Relationship {rel.relationship_id}: invalid observation_category")

    for evt in instance.temporal_events:
        if evt.observation_category not in VALID_OBSERVATION_CATEGORIES:
            errors.append(f"Event {evt.event_id}: invalid observation_category")

    return errors


# ---- Load pipeline ----

@dataclass
class LoadResult:
    admitted: bool
    instance_id: str
    errors: list = field(default_factory=list)
    failed_step: str = None


def load_instance(instance):
    # Validate schema + run all integrity checks.
    schema_errors = validate_schema(instance)
    if schema_errors:
        return LoadResult(False, instance.instance_id, schema_errors, "schema_validation")

    for name, check_fn in [
        ("structural", check_structural),
        ("temporal", check_temporal),
        ("relationship", check_relationship_integrity),
        ("business", check_business),
        ("consistency", check_consistency),
    ]:
        violations = check_fn(instance)
        if violations:
            return LoadResult(False, instance.instance_id, violations, f"integrity_{name}")

    return LoadResult(True, instance.instance_id)


# ---- Instance store ----

class InstanceStore:
    def __init__(self):
        self._instances = {}
        self._episode_assignments = {}

    def admit(self, instance):
        self._instances[instance.instance_id] = instance

    def get(self, instance_id):
        return self._instances.get(instance_id)

    def assign_to_episode(self, episode_id, instance_id):
        if instance_id not in self._instances:
            return False
        self._episode_assignments[episode_id] = instance_id
        return True

    def get_for_episode(self, episode_id):
        instance_id = self._episode_assignments.get(episode_id)
        if instance_id is None:
            return None
        return self._instances.get(instance_id)


# ---- Temporal session tracking ----

class TemporalGraphManager:
    # Tracks evolving sessions - ordered snapshots that share a session_id.

    def __init__(self):
        self._sessions = {}

    def add_snapshot(self, instance):
        session_id = instance.session_id
        if not session_id:
            return []

        prior = self._sessions.get(session_id, [])
        errors = self._check_deps(prior, instance)
        if not errors:
            self._sessions.setdefault(session_id, []).append(instance)
        return errors

    def get_latest(self, session_id):
        history = sorted(
            self._sessions.get(session_id, []),
            key=lambda i: i.snapshot_index,
        )
        return history[-1] if history else None

    def _check_deps(self, prior, new):
        # Events and relationships in new snapshot can't reference entities that don't exist yet.
        if not prior:
            return []

        known_entities = set()
        for snap in prior:
            for e in snap.entities:
                known_entities.add(e.entity_id)
        new_entity_ids = {e.entity_id for e in new.entities}

        errors = []
        for evt in new.temporal_events:
            for eid in [evt.source_entity_id, evt.target_entity_id]:
                if eid and eid not in known_entities and eid not in new_entity_ids:
                    errors.append(f"Event {evt.event_id}: references unknown entity {eid} not created in any prior snapshot")

        for rel in new.relationships:
            if rel.source_entity_id not in known_entities and rel.source_entity_id not in new_entity_ids:
                errors.append(f"Relationship {rel.relationship_id}: references unknown source {rel.source_entity_id}")
            if rel.target_entity_id not in known_entities and rel.target_entity_id not in new_entity_ids:
                errors.append(f"Relationship {rel.relationship_id}: references unknown target {rel.target_entity_id}")

        return errors


# ---- Observation model ----

def filter_visible(instance):
    # Return a copy with only visible-category items.
    return EnvironmentInstance(
        instance_id=instance.instance_id,
        schema_version=instance.schema_version,
        session_id=instance.session_id,
        snapshot_index=instance.snapshot_index,
        entities=[e for e in instance.entities if e.observation_category == "visible"],
        relationships=[r for r in instance.relationships if r.observation_category == "visible"],
        temporal_events=[t for t in instance.temporal_events if t.observation_category == "visible"],
        metadata={},
    )


def check_observation_consistency(instance, cited_entity_ids=None,
                                   cited_relationship_ids=None, cited_event_ids=None):
    # Check that all cited IDs are in the visible subset.
    visible_entities = {e.entity_id for e in instance.entities if e.observation_category == "visible"}
    visible_rels = {r.relationship_id for r in instance.relationships if r.observation_category == "visible"}
    visible_events = {t.event_id for t in instance.temporal_events if t.observation_category == "visible"}

    errors = []
    for eid in (cited_entity_ids or []):
        if eid not in visible_entities:
            errors.append(f"Cited entity {eid} not in visible subset")
    for rid in (cited_relationship_ids or []):
        if rid not in visible_rels:
            errors.append(f"Cited relationship {rid} not in visible subset")
    for evid in (cited_event_ids or []):
        if evid not in visible_events:
            errors.append(f"Cited event {evid} not in visible subset")
    return errors


# ---- Main class ----

class GraphEnvironment:
    # Manages the financial graph - ingestion, assignment, delivery, consistency checks.

    def __init__(self):
        self._store = InstanceStore()
        self._temporal = TemporalGraphManager()

    def ingest(self, instance):
        # Validate and store an environment instance.
        result = load_instance(instance)
        if not result.admitted:
            return result

        if instance.session_id:
            temporal_errors = self._temporal.add_snapshot(instance)
            if temporal_errors:
                return LoadResult(False, instance.instance_id, temporal_errors, "temporal_sequence")

        self._store.admit(instance)
        return result

    def assign(self, episode_id, instance_id=None, session_id=None):
        # Assign an instance to an episode. Returns instance_id or None.
        if instance_id:
            if self._store.assign_to_episode(episode_id, instance_id):
                return instance_id
            return None

        if session_id:
            latest = self._temporal.get_latest(session_id)
            if latest and self._store.assign_to_episode(episode_id, latest.instance_id):
                return latest.instance_id

        return None

    def deliver(self, episode_id):
        # Get the visible portion of the assigned instance.
        instance = self._store.get_for_episode(episode_id)
        if instance is None:
            return None
        return filter_visible(instance)

    def check_observation_consistency(self, episode_id, cited_entity_ids=None,
                                       cited_relationship_ids=None, cited_event_ids=None):
        # Check that cited evidence is within the visible subset.
        instance = self._store.get_for_episode(episode_id)
        if instance is None:
            return ["No instance assigned to this episode"]
        return check_observation_consistency(
            instance, cited_entity_ids, cited_relationship_ids, cited_event_ids
        )

    def get_instance_for_episode(self, episode_id):
        # Get the full (unfiltered) instance for an episode.
        return self._store.get_for_episode(episode_id)
