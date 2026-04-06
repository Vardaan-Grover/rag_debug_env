from pathlib import Path
from typing import List

from src.models import Domain, EmbeddingModel, FaultType

_CORPORA_DIR = Path(__file__).parent.parent / "corpora"

_TASK_DOMAIN = {1: Domain.SOFTWARE, 2: Domain.CLIMATE, 3: Domain.MEDICAL}

_TASK_DESCRIPTION = {
    1: (
        "Task 1 (Easy — Software): The RAG pipeline has one or two config faults "
        "on a Python-documentation corpus. Diagnose the retrieval degradation and "
        "fix the pipeline configuration to achieve high coverage of relevant chunks."
    ),
    2: (
        "Task 2 (Medium — Climate): The RAG pipeline has compound config faults "
        "on a climate-science corpus. Multiple issues interact; you must fix both "
        "to recover good retrieval quality."
    ),
    3: (
        "Task 3 (Hard — Medical): The pipeline uses the wrong embedding model on a "
        "medical corpus and has additional config faults. Identify the model mismatch "
        "and all config issues; pay special attention to multi-hop queries."
    ),
}

# Per-task episode query budget
_N_EPISODE_QUERIES = {1: 5, 2: 5, 3: 5}   # Task 3: 3 regular + 2 multi-hop
_MAX_STEPS = 10

# Coverage thresholds used for SUBMIT terminal_bonus / grading
_SUCCESS_COVERAGE = {1: 0.72, 2: 0.65, 3: 0.60}

# Model name → numpy file suffix
_MODEL_FILE = {
    EmbeddingModel.GENERAL: "general",
    EmbeddingModel.MEDICAL: "medical",
    EmbeddingModel.LEGAL:   "legal",
    EmbeddingModel.CODE:    "code",
}

# Fault sets per task
_TASK1_FAULT_SETS: List[List[FaultType]] = [
    [FaultType.CHUNK_TOO_LARGE, FaultType.NO_RERANKING],   # compound
    [FaultType.THRESHOLD_TOO_HIGH],
    [FaultType.TOP_K_TOO_SMALL],
    [FaultType.CHUNK_TOO_LARGE],
]

_TASK2_FAULT_SETS: List[List[FaultType]] = [
    [FaultType.THRESHOLD_TOO_LOW, FaultType.DUPLICATE_FLOODING],  # compound
    [FaultType.TOP_K_TOO_SMALL, FaultType.CONTEXT_OVERFLOW],      # compound
    [FaultType.DUPLICATE_FLOODING],
    [FaultType.CONTEXT_OVERFLOW],
]

_TASK3_FAULTS: List[FaultType] = [
    FaultType.WRONG_EMBEDDING_MODEL,
    FaultType.CHUNK_TOO_LARGE,
    FaultType.THRESHOLD_TOO_HIGH,
]
