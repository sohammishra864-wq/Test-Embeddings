# EvolveBench - Fraud Detection Evaluation Framework

A benchmark framework for evaluating fraud detection hypotheses. Scores them across 6 dimensions, runs full evaluation episodes, and provides a web UI to see everything in action.

## Setup

Needs Python 3.10+

```bash
pip install -r requirements.txt
```

## Project Structure

```
├── app.py              # Flask web app
├── src/
│   ├── models.py       # All data models (Pydantic)
│   ├── metrics.py      # 6 scoring dimensions + framework
│   ├── engine.py       # Execution engine + episode lifecycle
│   ├── graph.py        # Graph environment + integrity checks
│   └── knowledge.py    # Knowledge space + similarity matching
├── templates/
│   └── index.html      # Web UI (4 tabs)
├── tests/
│   ├── test_metrics.py # 26 tests for metrics
│   └── test_engine.py  # 18 tests for engine
└── requirements.txt
```

## Running

### Web UI

```bash
python app.py
```

Go to http://127.0.0.1:5000 - there are 4 tabs:

- **Full Evaluation** - score a hypothesis on all 6 dimensions
- **Single Dimension** - test one dimension at a time
- **Episode Aggregation** - aggregate scores across multiple hypotheses
- **Episode Flow** - run a complete episode end-to-end

All tabs have sample data pre-filled so you can just click the button.

### Tests

```bash
# all 44 tests
python -m pytest tests/ -v

# just metrics (26 tests)
python -m pytest tests/test_metrics.py -v

# just engine (18 tests)
python -m pytest tests/test_engine.py -v
```

### Using it in code

```python
from src.metrics import MetricsFramework

mf = MetricsFramework()

hypothesis = {
    "hypothesis_id": "h1",
    "entity_roles": ["nominee_director", "shell_company"],
    "relationship_flows": ["ownership_chain"],
    "point_of_deviation": "concealment of beneficial ownership",
    "cited_entity_ids": ["e1", "e2"],
    "cited_relationship_ids": ["r1"],
    "cited_event_ids": [],
    "explanation": "Shell companies layered across jurisdictions to obscure ownership.",
    "confidence": 0.8,
}

context = {
    "ground_truth": {
        "entity_roles": ["nominee_director", "shell_company", "beneficial_owner"],
        "relationship_flows": ["ownership_chain", "fund_transfer"],
        "point_of_deviation": "concealment of beneficial ownership",
    },
    "knowledge_space_lookup": {
        "max_structural_proximity": 0.3,
        "max_behavioral_similarity": 0.2,
    },
}

profile = mf.evaluate(hypothesis, context, episode_id="ep-1")
for r in profile.dimension_results:
    print(f"{r.dimension_name:30s}  {r.graded_outcome:25s}  {r.score}")
```

## The 6 Dimensions

| Dimension | What it checks |
|---|---|
| Fraud Correctness | Does the hypothesis match the actual fraud mechanism? |
| Novelty | Is this a new pattern or already documented? |
| Behavioral Plausibility | Is the proposed mechanism financially realistic? |
| Supporting Evidence | Are there enough citations to back it up? |
| Explanation Quality | Is the reasoning clear and detailed? |
| Confidence Assessment | Is the stated confidence level accurate? |

## Episode Lifecycle

```
IDLE -> INITIALIZED -> REGISTERED -> ENVIRONMENT_ASSIGNED -> AWAITING_SUBMISSION
  -> VALIDATING -> EVALUATING -> REPORTING -> ARCHIVED
```

Or `-> FAILED` if something goes wrong at any stage.

## What each module does

| Module | Purpose |
|---|---|
| `models.py` | All the data models - entities, relationships, submissions, results, etc. |
| `metrics.py` | The 6 scoring dimensions, registry, and evaluation framework |
| `engine.py` | Runs episodes through the full lifecycle, validates submissions |
| `graph.py` | Stores entity graphs, validates integrity, filters visible info |
| `knowledge.py` | Known fraud patterns, similarity matching (Jaccard for now) |
