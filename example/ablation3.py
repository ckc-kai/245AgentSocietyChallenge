import json
import logging
import numpy as np
from datetime import datetime
import os

from websocietysimulator import Simulator
from websocietysimulator.agent import SimulationAgent
from websocietysimulator.agent.modules.memory_modules import MemoryGenerative
from websocietysimulator.agent.modules.reasoning_modules import ReasoningCOT
from websocietysimulator.agent.modules import MemoryBase
from websocietysimulator.llm.llm import GeminiLLM
from langchain.docstore.document import Document
from websocietysimulator.agent.modules.reasoning_modules import ReasoningBase
from collections import Counter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("websocietysimulator")



class EfficientMemory(MemoryBase):
    '''
    Find the most similar memory without LLM;(Reudce the LLM calls)
    '''

    def __init__(self, llm):
        super().__init__(memory_type='efficient', llm=llm)

    def retriveMemory(self, query_scenario: str, k: int = 3):
        # Extract task name from query
        task_name = query_scenario
        
        # Return empty string if memory is empty
        if self.scenario_memory._collection.count() == 0:
            return ''
            
        # Find most similar memories
        similarity_results = self.scenario_memory.similarity_search_with_score(
            task_name, k=k)
            
        # Format memories with their similarity scores
        memories = []
        for i, (result, score) in enumerate(similarity_results, 1):
            trajectory = result.metadata['task_trajectory']
            memories.append(f"[Memory {i}]: {trajectory}")
        
        # Join trajectories with newlines and return
        return '\n\n'.join(memories)

    def addMemory(self, current_situation):
        memory_doc = Document(
            page_content=current_situation,
            metadata={
                "task_name": current_situation[:100],
                "task_trajectory": current_situation
            }
        )
        self.scenario_memory.add_documents([memory_doc])

class ConsistentReasoning(ReasoningBase):
    def __init__(self, profile_type_prompt, memory, llm, n_sample=3):
        super().__init__(profile_type_prompt, memory, llm)
        self.n_sample = n_sample
    
    def __call__(self, task_description: str):
        prompt = f'''Analyze this step-by-step and provide your final answer:
{task_description}
Think through this carefully, considering the user's profile, past behavior, and the business details.
Then provide your final answer in the exact format requested.
'''
        messages = [{"role": "user", "content": prompt}]
        responses = self.llm(
            messages=messages,
            temperature=0.20,
            max_tokens=400,
            n=self.n_sample
        )
        if isinstance(responses, str):
            logger.info("Single response received, skipping voting.")
            return responses

        # Parse and vote on star ratings
        parsed_responses = []
        star_ratings = []

        for response in responses:
            parsed_responses.append(response)
            try:
                # Extract star rating
                stars_line = [line for line in response.split('\n') if 'stars:' in line.lower()][0]
                stars = float(stars_line.split(':')[1].strip())
                star_ratings.append(stars)
            except:
                star_ratings.append(3.0) 

        # Select response with median star rating (most consistent)
        median_idx = np.argsort(star_ratings)[len(star_ratings) // 2]
        return parsed_responses[median_idx]

class MySimulationAgent(SimulationAgent):
    """Parent agent for all prompt ablation agents — A1/A2/A3"""

    def __init__(self, llm, use_memory: bool = True):
        super().__init__(llm=llm)
        self.memory = EfficientMemory(llm) if use_memory else None
        # 用 ConsistentReasoning
        self.reasoning = ConsistentReasoning(
            profile_type_prompt="", memory=self.memory, llm=self.llm, n_sample=3
        )

    # override in subclass
    def summarize_user_profile(self, user_reviews):
        """
        Analyze user's historical patterns without LLM calls.
        Returns a statistical summary of user behavior.
        """
        if not user_reviews:
            return "No historical data available."

        # Extract statistics
        ratings = [r['stars'] for r in user_reviews]
        review_lengths = [len(r['text'].split()) for r in user_reviews]

        avg_rating = np.mean(ratings)
        rating_std = np.std(ratings)
        rating_mode = Counter(ratings).most_common(1)[0][0]
        avg_length = int(np.mean(review_lengths))

        # Categorize user type
        if avg_rating >= 4.0:
            user_type = "generally positive and generous"
        elif avg_rating <= 2.0:
            user_type = "critical and discerning"
        else:
            user_type = "balanced and moderate"

        if rating_std < 0.5:
            consistency = "very consistent"
        elif rating_std < 1.0:
            consistency = "moderately consistent"
        else:
            consistency = "variable"

        profile = f"""User Behavior Profile Summary:
- Average Rating: {avg_rating:.1f} / 5.0 (Most common: {rating_mode})
- Rating Consistency: {consistency} (std: {rating_std:.2f})
- User Type: {user_type}
- Typical Review Length: {avg_length} words
- Total Reviews: {len(user_reviews)}"""

        return profile

    def learn_review_style(self, user_reviews):
        """
        Learn user's review style based on historical reviews.
        Returns a brief description of the user's writing style.
        """
        if not user_reviews:
            return "No historical data available."

        sorted_reviews = sorted(user_reviews, key=lambda x: x['stars'])

        learning_samples = []
        if len(sorted_reviews) >= 5:
            # learn from positive reviews
            learning_samples.append(f"Top Positive Review Sample {sorted_reviews[-1]['stars']} stars: {sorted_reviews[-1]['text'][:100]}")
            # learn from negative reviews
            learning_samples.append(f"Top Negative Review Sample {sorted_reviews[0]['stars']} stars: {sorted_reviews[0]['text'][:100]}")
            # learn from multiple neutral reviews
            neutral_reviews = [r for r in sorted_reviews if 2.0 < r['stars'] < 4.0]
            if len(neutral_reviews) > 1:
                neutral_sample = np.random.choice(neutral_reviews, 1, replace=False)
                learning_samples.append(f"Neutral Review Sample {neutral_sample[0]['stars']} stars: {neutral_sample[0]['text'][:100]}")
        else:
            learning_samples = [f"Review {r['stars']} stars: {r['text'][:100]}" for r in sorted_reviews]
        return '\n\n'.join(learning_samples)
    
    def build_prompt(self, *args, **kwargs):
        raise NotImplementedError()

    def _parse_out(self, llm_response):
        """Parse last stars: and review:"""
        try:
            last_stars = llm_response.rfind("stars:")
            last_review = llm_response.rfind("review:")
            sub = llm_response[min(last_stars, last_review):]

            stars = float([l for l in sub.split("\n") if "stars:" in l][0].split(":")[1])
            review = [l for l in sub.split("\n") if "review:" in l][0].split(":", 1)[1].strip()
            return {"stars": stars, "review": review[:512]}
        except:
            return {"stars": 0.0, "review": "PARSE_ERROR"}
    def workflow(self):
        try:
            user_id = self.task['user_id']
            item_id = self.task['item_id']  
            user_reviews = self.interaction_tool.get_reviews(user_id=user_id)
            user_profile = self.interaction_tool.get_user(user_id=user_id)

            for review in user_reviews:
                memory_entry = (
                    f"Reviewed business in category {review.get('categories', 'Unknown')}: "
                    f"Gave {review['stars']} stars. Review: {review['text'][:100]}..."
                )
                self.memory.addMemory(memory_entry)
            
            behavior_summary = self.summarize_user_profile(user_reviews)
            style_samples = self.learn_review_style(user_reviews)

            item_details = self.interaction_tool.get_item(item_id=item_id)

            query_scenario = (
                f"The user is now considering a business named "
                f"{item_details.get('name', 'N/A')} which is a"
                f"{item_details.get('category', 'N/A')}."
                )
            relevant_memory = self.memory.retriveMemory(query_scenario)

            # build prompt (A1 / A2 / A3)
            prompt = self.build_prompt(
                user_profile, behavior_summary, style_samples,
                     item_details, relevant_memory, user_reviews
            )

            llm_response = self.reasoning(task_description=prompt)
            return self._parse_out(llm_response)

        except Exception as e:
            logger.error(f"workflow error: {e}")
            return {"stars": 0.0, "review": "WORKFLOW_ERROR"}






class Agent_P1_FullPrompt(MySimulationAgent):
    """A1: Full Agent2 Prompt + Memory (真实 baseline)"""
    def build_prompt(self, user_profile, behavior_summary, style_samples,
                     item_details, relevant_memory, user_reviews):

        avg_rating = (
            np.mean([r["stars"] for r in user_reviews])
            if user_reviews else 3.0
        )

        return f"""
You are simulating a real user writing a review.

USER PROFILE:
{json.dumps(user_profile, indent=2)}

USER BEHAVIOR ANALYSIS:
{behavior_summary}

USER'S WRITING STYLE (from past reviews):
{style_samples}

BUSINESS TO REVIEW:
{json.dumps(item_details, indent=2)}

RELEVANT PAST EXPERIENCES:
{relevant_memory if relevant_memory else "No relevant past experiences."}

TASK:
Based on this user's profile, behavior patterns, and writing style, generate a review.

CRITICAL REQUIREMENTS:
1. Match user's typical rating pattern (avg: {avg_rating:.1f})
2. Match writing style and length
3. Be authentic
4. Consider business category & rating
5. Write 2–4 sentences

FORMAT (EXACT):
stars: [1.0–5.0]
review: [text]
        """
class Agent_P2_MediumPrompt(MySimulationAgent):
    """A2: Business + Memory only（删 Profile/Style）"""

    def build_prompt(
        self, user_profile, behavior_summary, style_samples,
                     item_details, relevant_memory, user_reviews
    ):

        return f"""
You are writing a Yelp review.

BUSINESS INFO:
{json.dumps(item_details, indent=2)}

RELEVANT EXPERIENCES FROM MEMORY:
{relevant_memory}

INSTRUCTIONS:
1. Analyze the business.
2. Compare with relevant past experiences.
3. Predict a reasonable star rating.
4. Write a natural 1–4 sentence review.

FORMAT (EXACT):
stars: [1.0–5.0]
review: [text]
        """
class Agent_P3_MinimalPrompt(MySimulationAgent):
    """A3: Minimal prompt — business only, NO MEMORY"""

    def build_prompt(
        self, user_profile, behavior_summary, style_samples,
                     item_details, relevant_memory, user_reviews
    ):

        return f"""
Write a Yelp review.

BUSINESS INFO:
{json.dumps(item_details, indent=2)}

INSTRUCTIONS:
- Guess a reasonable rating (1.0–5.0)
- Write 1–3 natural sentences that look like a Yelp review

FORMAT (EXACT):
stars: [number]
review: [text]
        """
if __name__ == "__main__":

    task_set = "yelp"     # or amazon / goodreads

    simulator = Simulator(
        data_dir="data/processed",
        device="cpu",
        cache=True,
    )
    simulator.set_task_and_groundtruth(
        task_dir=f"./example/track1/{task_set}/tasks",
        groundtruth_dir=f"./example/track1/{task_set}/groundtruth",
    )

    llm = GeminiLLM(
    api_key=os.getenv("GEMINI_API_KEY"),
    model = "models/gemini-2.0-flash",

    embedding_model="models/text-embedding-004",
)

    EXPERIMENTS = {
        "A1_FullPrompt": Agent_P1_FullPrompt,
        "A2_MediumPrompt": Agent_P2_MediumPrompt,
        "A3_MinimalPrompt": Agent_P3_MinimalPrompt,
    }

    for name, AgentClass in EXPERIMENTS.items():
        print(f"\n===== Running {name} =====\n")

        simulator.set_agent(AgentClass)
        simulator.set_llm(llm)

        simulator.run_simulation(number_of_tasks=50, enable_threading=True)
        results = simulator.evaluate()

        save_path = f"./results/evaluation/prompt_ablation_{name}.json"
        with open(save_path, "a") as f:
            json.dump({
                "ablation": name,
                "time": datetime.now().isoformat(),
                "results": results
            }, f, indent=4)
            f.write("\n")

        print(f"Saved → {save_path}")