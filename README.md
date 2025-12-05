# Agent Society Challenge: Task-Specific Architectures for LLM-Based User Simulation and Ranking

This repository contains implementations of LLM-based agents for user simulation and personalized recommendation tasks within the Web Society Simulator framework. We develop and evaluate multiple agent architectures with different combinations of memory modules, reasoning strategies, and statistical profiling mechanisms.

## Overview

This project implements agents for two complementary tasks:

- **Track A: User Simulation** - Agents generate authentic reviews with appropriate star ratings that reflect individual user preferences and writing styles
- **Track B: Recommendation** - Agents rank 20 candidate items based on historical user preferences and item attributes

All agents are evaluated on the Amazon dataset using established metrics for preference estimation, review generation quality, and ranking accuracy.

## Installation

### Prerequisites

- Python >= 3.10, < 4.0
- Poetry (for dependency management) or pip

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd 245AgentSocietyChallenge
```

2. Install dependencies using Poetry:
```bash
poetry install
```

Or using pip:
```bash
pip install -e .
```

### Required Dependencies

Key dependencies include:
- `openai` - OpenAI API client
- `anthropic` - Claude API client
- `langchain` - Memory and retrieval modules
- `sentence-transformers` - Embedding models
- `torch` - PyTorch for model operations
- `tiktoken` - Token counting utilities
- `python-dotenv` - Environment variable management

See `requirements.txt` or `pyproject.toml` for the complete list.

## Configuration

### API Keys

Create a `secrets.env` file in the project root with your API keys:

```env
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

The agents will automatically load these keys when running.

### Data Setup

Ensure the processed data is available in `./data/processed/`. Task files should be organized as:
- `./example/track1/{dataset}/tasks/` - User simulation tasks
- `./example/track1/{dataset}/groundtruth/` - User simulation ground truth
- `./example/track2/{dataset}/tasks/` - Recommendation tasks
- `./example/track2/{dataset}/groundtruth/` - Recommendation ground truth

Supported datasets: `amazon`, `goodreads`, `yelp`

## Usage

### Running User Simulation Agents

#### Agent 1 (MySimulationAgent)
```bash
python example/ModelingAgent1.py
```

#### Agent 2 (AnalyzeAgent)
```bash
python example/ModelingAgent2.py
```

#### Agent 3 (ModelingAgent3)
```bash
python example/ModelingAgent3.py
```

### Running Recommendation Agents

#### Agent 1 (MyRecommendationAgent)
```bash
python example/RecAgent1.py
```

#### Agent 2 (AdvancedRecommendationAgent)
```bash
python example/RecAgent2.py
```

#### Agent 3 (CategoryAwareRecommendationAgent)
```bash
python example/RecAgent3.py
```

#### Agent 4 (SimplifiedMemoryAgent)
```bash
python example/RecAgent4.py
```

### Running Ablation Studies

For recommendation Agent 2 ablation studies:
```bash
python run_ablation_study.py
```

Or run individual variants by modifying `ABLATION_VARIANT` in `example/RecAgent2_ablation.py`:
- `"full"` - Complete agent
- `"no_memory"` - Without memory module
- `"no_profiling"` - Without statistical profiling
- `"no_consistency"` - Without consistency reasoning

### Configuration Options

In each agent file, you can modify:
- `task_set`: Dataset to use (`"amazon"`, `"goodreads"`, or `"yelp"`)
- `number_of_tasks`: Number of tasks to run (default: 400 for full evaluation)
- `model`: LLM model to use (e.g., `"gpt-4.1"`, `"gpt-4o-mini"`, `"claude-3-opus"`)
- `enable_threading`: Enable parallel processing
- `max_workers`: Number of parallel workers

## Project Structure

```
.
├── example/
│   ├── ModelingAgent1.py          # User simulation Agent 1
│   ├── ModelingAgent2.py          # User simulation Agent 2
│   ├── ModelingAgent3.py          # User simulation Agent 3
│   ├── ModelingAgent_baseline.py  # Baseline agent
│   ├── RecAgent1.py               # Recommendation Agent 1
│   ├── RecAgent2.py               # Recommendation Agent 2
│   ├── RecAgent2_ablation.py      # Ablation study for Agent 2
│   ├── RecAgent3.py               # Recommendation Agent 3
│   ├── RecAgent4.py               # Recommendation Agent 4
│   ├── RecAgent_baseline.py       # Baseline recommendation agent
│   ├── track1/                     # User simulation tasks and ground truth
│   └── track2/                     # Recommendation tasks and ground truth
├── results/
│   ├── evaluation/                 # Evaluation results (JSON format)
│   └── generation_detail/          # Detailed LLM outputs (if enabled)
├── websocietysimulator/            # Framework code
├── requirements.txt                # Python dependencies
├── pyproject.toml                  # Poetry configuration
└── README.md                       # This file
```

## Agent Descriptions

### Track A: User Modeling Agents

#### Agent 1 (MySimulationAgent / MCAgent)
- **Generative Memory**: Semantic similarity search to identify top 3 most relevant memory recalls
- **Chain-of-Thought Reasoning**: Structured step-by-step analysis for rating and review generation
- **Memory retrieval system**: Encodes user review histories as memory entries with user, rating, and review text
- **Scenario-based querying**: Uses task-business name and type to semantically search memory store

#### Agent 2 (AnalyzeAgent)
- **EfficientMemory**: Vector search-based memory retrieval without LLM calls for computational efficiency
- **Statistical User Profiling**: Analyzes user's historical patterns (rating distribution, consistency, user type classification)
- **Self-Consistent Reasoning**: Multiple response sampling (n=3) with median-based selection and variance reduction
- **Cost-efficient design**: Minimizes API calls while maintaining semantic relevance in memory retrieval

#### Agent 3 (ModelingAgent3)
- **EnhancedMemory**: Combines episodic memory retrieval, semantic retrieval, and comparative references
- **Persona Construction**: Forms detailed psychological portrait (rating behavior, linguistic style, value orientation)
- **UserItemFitReasoner**: Structured compatibility analysis between user traits and item attributes
- **MultiAspectGenerator**: Creates various review candidates considering writing style, tone, and emotions
- **Authenticity Validation**: Selects most realistic review based on rating prediction match, length consistency, and sentiment alignment

### Track B: Recommendation Agents

#### Agent 1 (MyRecommendationAgent)
- **Generative Memory**: Semantic similarity-based retrieval to identify top 3 most relevant past experiences
- **Chain-of-Thought Reasoning**: Structured analytical steps for ranking 20 candidate items
- **User profile and item information integration**: Token truncation for context management

#### Agent 2 (AdvancedRecommendationAgent)
- **EfficientMemory**: Vector search-based memory retrieval without LLM calls
- **Statistical User Profiling**: Rule-based behavioral analysis (rating patterns, category preferences, user type classification)
- **Preference Signal Extraction**: Identifies highest- and lowest-rated items as concrete preference signals
- **ConsistentReasoning**: Self-consistency reasoning with single-sample approach for ranking

#### Agent 3 (CategoryAwareRecommendationAgent)
- **Category Preference Extraction**: Organizes review history by categories and calculates average ratings
- **Two-Stage Ranking**: Stage 1 shortlists top 10 candidates, Stage 2 refines complete ranking of all 20 items
- **Category-based matching**: Scores candidates based on category overlap with user preferences
- **Hierarchical refinement**: Fallback strategies for robust parsing

#### Agent 4 (SimplifiedMemoryAgent)
- **MemoryDILU**: Simple similarity-based memory module
- **Chain-of-Thought Reasoning**: Proven COT approach adapted for ranking
- **Compact data formatting**: Efficient token management for user profiles, reviews, and candidate items
- **Robust parsing**: Multiple fallback strategies for handling malformed LLM outputs

## Results

### Evaluation Results

Evaluation results are saved in `./results/evaluation/` with the following naming convention:
- User Simulation: `evaluation_results_track1_{dataset}_{agent_name}.json`
- Recommendation: `evaluation_results_track2_{dataset}_{agent_name}.json`
- Ablation Studies: `evaluation_results_track2_{dataset}_agent2_{variant}.json`

Each result file contains:
- `type`: Task type ("simulation" or "recommendation")
- `metrics`: Performance metrics (hit rates, preference estimation, review generation quality, etc.)
- `data_info`: Information about evaluated tasks

### Generation Details

If enabled, detailed LLM outputs are saved in `./results/generation_detail/`:
- User Simulation: `sim_agent{1,2,3}.txt`
- Recommendation: `rec_agent{1,2,3,4}.txt`

## Evaluation Metrics

### User Simulation
- **Preference Estimation**: Accuracy of predicted star ratings
- **Review Generation**: Quality of generated review text
- **Overall Quality**: Combined metric

### Recommendation
- **HR@1, HR@3, HR@5**: Hit Rate at positions 1, 3, and 5
- **Average Hit Rate**: Average across all positions

