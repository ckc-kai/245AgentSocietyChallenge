# Simulation Agent Implementation Ideas

## Track A: User Modeling Agents

### Agent 1 (MySimulationAgent / MCAgent):
1. Generative Memory: Semantic similarity search to identify top 3 most relevant memory recalls
2. Chain-of-Thought Reasoning: Structured step-by-step analysis for rating and review generation
3. Memory retrieval system: Encodes user review histories as memory entries with user, rating, and review text
4. Scenario-based querying: Uses task-business name and type to semantically search memory store

### Agent 2 (AnalyzeAgent):
1. EfficientMemory: Vector search-based memory retrieval without LLM calls for computational efficiency
2. Statistical User Profiling: Analyzes user's historical patterns (rating distribution, consistency, user type classification)
3. Self-Consistent Reasoning: Multiple response sampling (n=3) with median-based selection and variance reduction
4. Cost-efficient design: Minimizes API calls while maintaining semantic relevance in memory retrieval

### Agent 3 (ModelingAgent3):
1. EnhancedMemory: Combines episodic memory retrieval, semantic retrieval, and comparative references
2. Persona Construction: Forms detailed psychological portrait (rating behavior, linguistic style, value orientation)
3. UserItemFitReasoner: Structured compatibility analysis between user traits and item attributes
4. MultiAspectGenerator: Creates various review candidates considering writing style, tone, and emotions
5. Authenticity Validation: Selects most realistic review based on rating prediction match, length consistency, and sentiment alignment

## Track B: Recommendation Agents

### Agent 1 (MyRecommendationAgent):
1. Generative Memory: Semantic similarity-based retrieval to identify top 3 most relevant past experiences
2. Chain-of-Thought Reasoning: Structured analytical steps for ranking 20 candidate items
3. User profile and item information integration with token truncation for context management

### Agent 2 (AdvancedRecommendationAgent):
1. EfficientMemory: Vector search-based memory retrieval without LLM calls
2. Statistical User Profiling: Rule-based behavioral analysis (rating patterns, category preferences, user type classification)
3. Preference Signal Extraction: Identifies highest- and lowest-rated items as concrete preference signals
4. ConsistentReasoning: Self-consistency reasoning with single-sample approach for ranking

### Agent 3 (CategoryAwareRecommendationAgent):
1. Category Preference Extraction: Organizes review history by categories and calculates average ratings
2. Two-Stage Ranking: Stage 1 shortlists top 10 candidates, Stage 2 refines complete ranking of all 20 items
3. Category-based matching: Scores candidates based on category overlap with user preferences
4. Hierarchical refinement with fallback strategies for robust parsing

### Agent 4 (SimplifiedMemoryAgent):
1. MemoryDILU: Simple similarity-based memory module
2. Chain-of-Thought Reasoning: Proven COT approach adapted for ranking
3. Compact data formatting: Efficient token management for user profiles, reviews, and candidate items
4. Robust parsing: Multiple fallback strategies for handling malformed LLM outputs

## For running the model, you should provide API keys for llm; then run python -m path/to/your/agent.
## The LLM outputs are in the ./results/generation_detial.
## The evaluation results are in the ./results/evaluation.
## Current format is not formal, to be worked on.