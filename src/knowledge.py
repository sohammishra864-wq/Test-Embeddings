# Knowledge space - stores documented fraud patterns and measures how
# similar new hypotheses are to what's already known.

import copy
from dataclasses import dataclass, field

from src.models import KnowledgeItem


@dataclass
class FraudFamily:
    family_id: str
    name: str
    description: str
    related_families: list = field(default_factory=list)


class KnowledgeRepository:
    # Simple in-memory store for knowledge items.

    def __init__(self):
        self.items = {}

    def add(self, item):
        self.items[item.item_id] = item

    def get(self, item_id):
        return self.items.get(item_id)

    def count(self):
        return len(self.items)

    def contains(self, item_id):
        return item_id in self.items


class Taxonomy:
    # Groups knowledge items into fraud families.

    def __init__(self):
        self.families = {}
        self._membership = {}  # item_id -> family_id

    def add_family(self, family):
        self.families[family.family_id] = family

    def get_family(self, family_id):
        return self.families.get(family_id)

    def assign_item(self, item_id, family_id):
        if family_id not in self.families:
            return False
        self._membership[item_id] = family_id
        return True

    def get_members(self, family_id):
        return [k for k, v in self._membership.items() if v == family_id]

    def find_nearest_families(self, mechanism_description):
        # Find families with the most keyword overlap. TODO: use embeddings instead
        words = set(mechanism_description.lower().split())
        scored = []
        for fam in self.families.values():
            fam_words = set(fam.description.lower().split())
            overlap = len(words & fam_words)
            if overlap > 0:
                scored.append((overlap, fam))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [fam for _, fam in scored[:3]]


@dataclass
class ItemPosition:
    item_id: str
    structural_proximity: float
    behavioral_similarity: float


@dataclass
class LookupPosition:
    nearest_items: list
    max_structural_proximity: float
    max_behavioral_similarity: float


class RepresentationSpace:
    # Measures similarity between mechanisms using Jaccard on keyword sets.
    # TODO: switch to actual embeddings when we need better accuracy

    def __init__(self):
        self._features = {}

    def set_position(self, item_id, features):
        self._features[item_id] = features

    def query_position(self, mechanism_features, top_k=5):
        if not self._features:
            return LookupPosition([], 0.0, 0.0)

        scored = []
        for item_id, features in self._features.items():
            if not features and not mechanism_features:
                jaccard = 1.0
            elif not features or not mechanism_features:
                jaccard = 0.0
            else:
                jaccard = len(features & mechanism_features) / len(features | mechanism_features)

            scored.append(ItemPosition(item_id, jaccard, jaccard))

        scored.sort(key=lambda p: p.structural_proximity, reverse=True)
        top = scored[:top_k]

        return LookupPosition(
            nearest_items=top,
            max_structural_proximity=max(p.structural_proximity for p in top) if top else 0.0,
            max_behavioral_similarity=max(p.behavioral_similarity for p in top) if top else 0.0,
        )


@dataclass
class LookupResult:
    nearest_families: list
    position: LookupPosition


@dataclass
class UpdateResult:
    success: bool
    error: str = None


class KnowledgeSpace:
    # Stores documented fraud knowledge and lets you query how similar a new hypothesis is.

    def __init__(self, initial_version="0.1.0"):
        self._current_version = initial_version
        self._repository = KnowledgeRepository()
        self._taxonomy = Taxonomy()
        self._representation = RepresentationSpace()
        self._released_versions = {}

    @property
    def current_version(self):
        return self._current_version

    @property
    def item_count(self):
        return self._repository.count()

    def lookup(self, mechanism_description, entity_roles, relationship_flows, version=None):
        # Where does this mechanism sit relative to what we already know?
        repo, tax, rep = self._resolve_stores(version)
        nearest_families = tax.find_nearest_families(mechanism_description)
        features = self._extract_features(mechanism_description, entity_roles, relationship_flows)
        position = rep.query_position(features)
        return LookupResult(nearest_families=nearest_families, position=position)

    def get_item(self, item_id, version=None):
        repo, _, _ = self._resolve_stores(version)
        return repo.get(item_id)

    def get_family(self, family_id, version=None):
        _, tax, _ = self._resolve_stores(version)
        return tax.get_family(family_id)

    def get_family_members(self, family_id, version=None):
        _, tax, _ = self._resolve_stores(version)
        return tax.get_members(family_id)

    def knowledge_update(self, item, family_id, new_family=None):
        # Add a new item. Rolls back if the family assignment fails.
        if self._repository.contains(item.item_id):
            return UpdateResult(False, f"Item {item.item_id} already exists")

        self._repository.add(item)

        if new_family:
            self._taxonomy.add_family(new_family)

        if not self._taxonomy.assign_item(item.item_id, family_id):
            # rollback: remove the item and family we just added
            self._repository.items.pop(item.item_id, None)
            if new_family:
                self._taxonomy.families.pop(new_family.family_id, None)
            return UpdateResult(False, f"Family {family_id} not found")

        features = self._extract_features(
            item.mechanism_description, item.entity_roles, item.relationship_flows
        )
        self._representation.set_position(item.item_id, features)
        return UpdateResult(True)

    def version_snapshot(self, new_version):
        # Freeze current state as an immutable version.
        self._released_versions[new_version] = {
            "repository": copy.deepcopy(self._repository),
            "taxonomy": copy.deepcopy(self._taxonomy),
            "representation": copy.deepcopy(self._representation),
        }
        self._current_version = new_version
        return new_version

    def _resolve_stores(self, version):
        if version and version in self._released_versions:
            state = self._released_versions[version]
            return state["repository"], state["taxonomy"], state["representation"]
        return self._repository, self._taxonomy, self._representation

    @staticmethod
    def _extract_features(description, entity_roles, flows):
        words = set(description.lower().split())
        words.update(r.lower() for r in entity_roles)
        words.update(f.lower() for f in flows)
        return words
