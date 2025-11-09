from websocietysimulator.agent import SimulationAgent
from websocietysimulator.agent.modules import MemoryBase
from langchain.docstore.document import Document
from websocietysimulator.agent.modules.reasoning_modules import ReasoningBase
from websocietysimulator import Simulator
from websocietysimulator.llm.llm import ClaudeLLM
import numpy as np
from collections import Counter
import json
import logging
import os
from datetime import datetime
from dotenv import load_dotenv
load_dotenv(dotenv_path="./secrets.env")
logger = logging.getLogger("websocietysimulator")
logger.propagate = False


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

class AnalyzeAgent(SimulationAgent):
    '''
    1. User Profiling: Analyzes user's historical patterns (rating distribution, sentiment, preferences)
    2. Context-Aware Retrieval: Uses semantic similarity to find relevant past reviews
    3. Persona-Based Generation: Creates reviews that match user's writing style and preferences
    4. Self-Consistency: Uses multiple reasoning paths and selects most consistent output
    '''
    def __init__(self, llm):
        super().__init__(llm=llm)
        self.memory = EfficientMemory(llm=self.llm)
        self.reasoning = ConsistentReasoning(profile_type_prompt='', memory=self.memory, llm=self.llm, n_sample=3)
    
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
    def _parse_out(self, llm_response: str) -> dict:
        """
        A helper function to safely parse the LLM's response.
        There will be thoughts first then stars and review at the end.
        """
        try:
            # Find the *last* occurrence of "stars:" and "review:"
            last_stars_index = llm_response.rfind('stars:')
            last_review_index = llm_response.rfind('review:')
            
            if last_stars_index == -1 or last_review_index == -1:
                logger.error(f"Could not find 'stars:' or 'review:' in output. Response: {llm_response[:500]}")
                raise ValueError("Could not find 'stars:' or 'review:' in output.")

            # Extract the lines from the *last* part of the response
            response_subset = llm_response[min(last_stars_index, last_review_index):]
            stars_line = [line for line in response_subset.split('\n') if 'stars:' in line][0]
            review_line = [line for line in response_subset.split('\n') if 'review:' in line][0]

            stars = float(stars_line.split(':', 1)[1].strip())
            review = review_line.split(':', 1)[1].strip()

            if stars < 1.0: stars = 1.0
            if stars > 5.0: stars = 5.0
            if len(review) > 512: review = review[:512]
            
            return {"stars": stars, "review": review}
        except Exception as e:
            logger.error(f"Error parsing LLM response: {e}")
            return {"stars": 0.0, "review": "No review generated due to parsing error."}
    
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

            # Stage 4: Self-consistent reasoning
            task_description = f"""You are simulating a real user writing a review.

USER PROFILE:
{json.dumps(user_profile, indent=2)}

USER BEHAVIOR ANALYSIS:
{behavior_summary}

USER'S WRITING STYLE (from past reviews):
{style_samples}

BUSINESS TO REVIEW:
{json.dumps(item_details, indent=2)}

RELEVANT PAST EXPERIENCES:
{relevant_memory if relevant_memory else "No directly relevant past experiences."}

TASK:
Based on this user's profile, behavior patterns, and writing style, generate a review for this business.

CRITICAL REQUIREMENTS:
1. Match the user's typical rating pattern (consider their average: {np.mean([r['stars'] for r in user_reviews]) if user_reviews else 3.0:.1f})
2. Write in a similar style and length as their past reviews
3. Be authentic - don't be overly positive or negative unless that matches the user's pattern
4. Consider the business's existing rating and category
5. Write 2-4 sentences that sound natural

FORMAT (EXACTLY):
stars: [rating between 1.0 and 5.0]
review: [your review text]"""

            llm_response = self.reasoning(task_description=task_description)
            logger.info(f"Task {user_id}: LLM response: {llm_response}")

            result = self._parse_out(llm_response)
            with open(f'./results/generation_detail/analyzer_v1.txt', 'a') as f:
                f.write(f'\n {datetime.now()}')
                f.write(f'\n LLM Response: {json.dumps(result, indent=4)}')


            return result

        except Exception as e:
            logging.error(f"Error during workflow execution: {e}")
            import traceback
            traceback.print_exc()
            return {
                "stars": 3.0,
                "review": "No review generated due to workflow error."
            }

if __name__ == "__main__":
    claude_api = os.getenv("CLAUDE_API_KEY")
    openai_api = os.getenv("OPENAI_API_KEY")
    if not claude_api or not openai_api:
        raise ValueError("CLAUDE_API_KEY or OPENAI_API_KEY not found in environment variables.")
    else:
        logger.info("API keys successfully loaded from environment variables.")

    print("Starting simulation with MySimulationAgent and ClaudeLLM...")
    # Set the data
    task_set = "amazon" # "goodreads" or "yelp" or "amazon"
    simulator = Simulator(data_dir="/Users/ckc/Desktop/UCLA/2025fall/245/AgentSocietyChallenge/data/processed", device="auto", cache=True)
    simulator.set_task_and_groundtruth(task_dir=f"./example/track1/{task_set}/tasks", groundtruth_dir=f"./example/track1/{task_set}/groundtruth")

    # Set the agent and LLM
    llm = ClaudeLLM(api_key=claude_api, model="claude-sonnet-4-20250514",openai_api_key=openai_api)
    simulator.set_agent(AnalyzeAgent)
    simulator.set_llm(llm)

    # Run the simulation
    # If you don't set the number of tasks, the simulator will run all tasks.
    outputs = simulator.run_simulation(number_of_tasks=10, enable_threading=True, max_workers=10)
    
    # Evaluate the agent
    evaluation_results = simulator.evaluate()       
    with open(f'./results/evaluation/evaluation_results_track1_{task_set}_model2.json', 'a') as f:
        time = {"time": datetime.now().isoformat()}
        f.write(json.dump(time, f, indent=4)+"\n")
        f.write(json.dump(evaluation_results, f, indent=4) + '\n')

    # Get evaluation history
    evaluation_history = simulator.get_evaluation_history()