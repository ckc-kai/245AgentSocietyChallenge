# Simulation Agent Implementation Ideas
## MyModelingAgent:
1. Memory Module to find 3 most relavent memories.
2. Chain of Thoughts

## Another Agent:
1. User Profiling: Analyzes user's historical patterns (rating distribution, sentiment, preferences)
2. Context-Aware Retrieval: Uses semantic similarity to find relevant past reviews (No LLM involved in this step)
3. Persona-Based Generation: Creates reviews that match user's writing style and preferences
4. Self-Consistency: Uses multiple reasoning paths and selects most consistent output


## For running the model, you should provide API keys for llm; then run python -m path/to/your/agent.
## The LLM outputs are in the ./results/generation_detial.
## The evaluation results are in the ./results/evaluation.
## Current format is not formal, to be worked on.