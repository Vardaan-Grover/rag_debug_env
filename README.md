<h1 align="center">
  🚀 RAGDebugEnv
</h1>

<p align="center">
  <strong>An OpenEnv-compliant RL environment for training autonomous agents to debug and heal broken RAG pipelines.</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/OpenEnv-Compatible-green" alt="OpenEnv">
  <img src="https://img.shields.io/badge/Reinforcement%20Learning-Ready-orange" alt="RL">
  <img src="https://img.shields.io/badge/Domain-RAG%20Pipelines-purple" alt="RAG">
</p>

---

## 🌟 The Elevator Pitch

Retrieval-Augmented Generation (RAG) pipelines are notoriously fragile. A wrong chunk size, a misconfigured similarity threshold, or a poorly matched embedding model can silently destroy retrieval quality. 

Training an AI Agent to fix these pipelines autonomously requires Reinforcement Learning (RL). But there's a huge problem: **Real RAG pipelines are too slow for RL.** A single retrieval step takes 5-10 seconds. Training an agent over millions of episodes would take years.

**Enter RAGDebugEnv.** ⚡

RAGDebugEnv is a revolutionary simulated environment that reduces RAG execution time to **sub-milliseconds** per step. By performing the heavy lifting offline and simulating pipeline failures using pure matrix mathematics, RAGDebugEnv allows you to train robust AI debugging agents on a standard laptop. 

Best of all? **Zero Sim-to-Real Gap.** The observation schema your agent sees in simulation is *identical* to a production backend (Pinecone, Weaviate, etc.). Train your agent in milliseconds, then deploy it to fix real-world enterprise RAG pipelines!

---

## 🔥 Key Features & Immense Value

- ⚡ **Speed of Light Simulation**: By caching vector similarities into `S_true` matrices offline, episodes run in milliseconds instead of seconds.
- 🧬 **Mathematical Fault Injection**: Simulates 9 common RAG failures (e.g., `CHUNK_TOO_LARGE`, `THRESHOLD_TOO_LOW`, `DUPLICATE_FLOODING`, `WRONG_EMBEDDING_MODEL`) through clever matrix transformations rather than slow text operations.
- 🎯 **Absolute Ground Truth**: Uses Cross-Encoder grading (`R*`) independent of the embedding model to provide an undisputable "actual relevance" reward signal for the agent.
- 🧠 **Curriculum Training**: Three beautifully crafted task tiers—from simple single-fault Python docs to multi-hop medical textbook failures.
- 🚀 **OpenEnv Standard**: Fully compliant with the AgentBeats OpenEnv framework, making it ready for out-of-the-box local training and Hugging Face Spaces deployment. 

---

## 🧠 Architecture Deep Dive (How It Works)

If you're an engineer, you'll love this. **The environment never runs a real RAG pipeline during episodes.**

### The Offline-to-Online Pipeline

```text
1. OFFLINE (Run Once via build_corpus.py):
Documents → Chunks → Embed (via 4 models) → S_true_[model].npy
Chunks + Queries → Cross-encoder → ground_truth.json (R*)

2. ONLINE (During Agent Training - milliseconds):
S_faulted = apply_fault_math(S_true_general)
R_agent = threshold_filter(top_k(S_faulted, config))
coverage = |R_agent ∩ R*| / |R*|
Reward = delta(coverage) + precision - costs
```

### The Two Matrices
1. **`S_true` (Bi-encoder Similarity)**: Represents how the embedding model *perceives* similarity. Shape: `(n_queries, n_chunks)`.
2. **`R*` (Ground Truth)**: Represents *actual* relevance determined by an offline cross-encoder. Independent of embeddings. The gap between `S_true` and `R*` is the learning signal for the agent!

### Simulating Faults with Math
Instead of manipulating text chunks on the fly, faults are applied as mathematical transformations on the `S_true` matrix:
* **`CHUNK_TOO_LARGE`**: `scipy.ndimage.uniform_filter1d(S_true)` (averages neighboring scores, diluting relevance).
* **`THRESHOLD_TOO_HIGH`**: `S_true * 0.55` (deflates all scores, hiding relevant chunks).
* **`WRONG_EMBEDDING_MODEL`**: `permute_rows(S_true)` (scrambles scores, simulating a model that doesn't understand the domain).

---

## 🛠️ Local Setup & Quick Start

Want to train your own RAG healer? Follow these steps to get everything running locally on your machine.

### 1. Prerequisites
- **Python 3.10+** installed
- **Docker** (optional, but recommended for the isolated server)
- Git

### 2. Installation
Clone the repository and set up your virtual environment:
```bash
git clone https://github.com/your-username/rag_debug_env.git
cd rag_debug_env

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e ".[dev]"
```

### 3. Build the Corpus (The Offline Magic)
Before running the environment, generate the `S_true` matrices and `R*` ground truth.
```bash
# This will run Stages 1-5 (Document loading -> Chunking -> Query Gen -> Embedding)
python -m rag_debug_env.corpora.build_corpus
```

### 4. Start the Environment Server
You have two ways to run the environment: natively or via Docker.

**Option A: Native Python (Fastest for local testing)**
```bash
python -m rag_debug_env.server.app
```

**Option B: Docker (Closest to production/OpenEnv)**
```bash
docker build -t rag_debug_env-env:latest -f server/Dockerfile .
# The environment client will automatically spin up the container when initialized
```

### 5. Run the Baseline Agent
Test the environment with our baseline agent that will try to fix pipeline errors:
```bash
python -m rag_debug_env.baseline.eval_agent --task 1 --episodes 3
```

*(You should see the agent attempting to fix faults and interacting with the simulation in sub-milliseconds!)*

---

## 🎮 The Training Tasks

We designed the environment with a continuous curriculum:

| Task Tier | Domain | Fault Complexity | Max Steps | Goal |
|---|---|---|---|---|
| **Task 1 (Easy)** | Software (Python Docs) | 1 Random Fault | 10 | Fix the single point of failure |
| **Task 2 (Medium)** | Climate (Wikipedia) | 2 Interacting Faults | 15 | Fix compounded issues (e.g. Chunk Too Large + No Reranking) |
| **Task 3 (Hard)** | Medical (Textbooks) | 3 Faults (Multi-hop) | 20 | Swap to the correct domain embedding model + tune config |

---

## 💻 OpenEnv API Usage Example

Integrating with your own agent is incredibly simple:

```python
from rag_debug_env import RagDebugAction, RagDebugEnv

# Connect to the environment
env = RagDebugEnv.from_docker_image("rag_debug_env-env:latest")

# Reset to get a new broken RAG pipeline (Task 1)
result = env.reset(task_id=1)
print(f"Initial State: {result.observation}")

# Take an action to fix it
action = RagDebugAction(action_type="ADJUST_THRESHOLD", value=0.65)
result = env.step(action)

print(f"New Reward: {result.reward}")
env.close()
```

---

## 🤝 Contributing

We welcome pull requests! Whether it's adding new faults, expanding domains, or optimizing the baseline agent:
1. Fork the repo.
2. Create a feature branch: `git checkout -b feature/cool-new-fault`
3. Commit your changes: `git commit -m 'Add sparse retrieval fault'`
4. Push to the branch: `git push origin feature/cool-new-fault`
5. Open a Pull Request.

---

*Built for the AgentBeats OpenEnv Hackathon.* 🌟
