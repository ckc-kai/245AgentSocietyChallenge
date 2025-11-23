from websocietysimulator.agent import RecommendationAgent
from websocietysimulator.agent.modules import MemoryBase
from langchain.docstore.document import Document
from websocietysimulator.agent.modules.reasoning_modules import ReasoningBase
from websocietysimulator import Simulator
from websocietysimulator.llm.llm import OpenAILLM
import numpy as np
from collections import Counter
import json
import logging
import os
import re
import tiktoken
from datetime import datetime
from dotenv import load_dotenv
load_dotenv(dotenv_path="./secrets.env")
logger = logging.getLogger("websocietysimulator")
logger.propagate = False

def num_tokens_from_string(string: str) -> int:
    encoding = tiktoken.get_encoding("cl100k_base")
    try:
        return len(encoding.encode(string))
    except:
        return 0

class EfficientMemory(MemoryBase):
    '''
    Find the most similar memory without LLM (Reduce the LLM calls)
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
Think through this carefully, considering the user's profile, past behavior, item details, and location.
Then provide your final answer in the exact format requested.
'''
        messages = [{"role": "user", "content": prompt}]
        responses = self.llm(
            messages=messages,
            temperature=0.20,
            max_tokens=2000,
            n=self.n_sample
        )
        if isinstance(responses, str):
            logger.info("Single response received, skipping voting.")
            return responses

        # Parse all rankings
        parsed_rankings = []
        valid_responses = []
        for response in responses:
            try:
                match = re.search(r"\[.*\]", response, re.DOTALL)
                if match:
                    ranking = eval(match.group())
                    if isinstance(ranking, list) and len(ranking) > 0:
                        parsed_rankings.append(ranking)
                        valid_responses.append(response)
            except:
                continue
        
        if not parsed_rankings:
            # Fallback: return first response if parsing fails
            return responses[0] if responses else ""
        
        if len(parsed_rankings) == 1:
            return valid_responses[0]
        
        # Aggregate rankings using Borda count: sum positions across all samples
        # Lower position = higher score (position 0 gets max points)
        all_items = set()
        for ranking in parsed_rankings:
            all_items.update(ranking)
        
        item_scores = {item: 0.0 for item in all_items}
        for ranking in parsed_rankings:
            for position, item in enumerate(ranking):
                # Position 0 gets score=len(ranking), position 1 gets len-1, etc.
                score = len(ranking) - position
                if item in item_scores:
                    item_scores[item] += score
        
        # Create aggregated ranking: sort by total score (descending)
        aggregated_ranking = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)
        final_ranking = [item for item, score in aggregated_ranking]
        
        # Return the response whose ranking is closest to the aggregated ranking
        # Measure similarity by average position difference in top-10
        best_match_idx = 0
        best_similarity = float('inf')
        
        for i, ranking in enumerate(parsed_rankings):
            # Calculate average position difference for items in both rankings
            total_diff = 0
            count = 0
            for j, item in enumerate(final_ranking[:10]):  # Check top 10
                if item in ranking:
                    rank_pos = ranking.index(item)
                    total_diff += abs(j - rank_pos)
                    count += 1
            
            avg_diff = total_diff / count if count > 0 else float('inf')
            if avg_diff < best_similarity:
                best_similarity = avg_diff
                best_match_idx = i
        
        return valid_responses[best_match_idx]

class AdvancedRecommendationAgent(RecommendationAgent):
    '''
    1. User Profiling: Analyzes user's historical patterns (preferred categories, rating distribution, preferences)
    2. Context-Aware Retrieval: Uses semantic similarity to find relevant past reviews
    3. Item-User Matching: Compares candidate items with user preferences
    4. Self-Consistency: Uses multiple reasoning paths and selects most consistent output
    5. Location Awareness: Considers geographic proximity if location is provided
    '''
    def __init__(self, llm):
        super().__init__(llm=llm)
        self.memory = EfficientMemory(llm=self.llm)
        self.reasoning = ConsistentReasoning(profile_type_prompt='', memory=self.memory, llm=self.llm, n_sample=3)
    
    def analyze_user_preferences(self, user_reviews, user_profile):
        """
        Analyze user's historical patterns without LLM calls.
        Returns a statistical summary of user preferences.
        """
        if not user_reviews:
            return "No historical data available."

        # Extract statistics
        ratings = [r['stars'] for r in user_reviews]
        categories = []
        for review in user_reviews:
            item_id = review.get('item_id')
            if item_id:
                item = self.interaction_tool.get_item(item_id=item_id)
                if item:
                    cat = item.get('category') or item.get('categories', '')
                    if cat:
                        # Handle both string and list categories
                        if isinstance(cat, list):
                            categories.extend(cat)  # Add all items from list
                        else:
                            categories.append(cat)  # Add single string

        avg_rating = np.mean(ratings) if ratings else 3.0
        rating_std = np.std(ratings) if ratings else 0.0
        rating_mode = Counter(ratings).most_common(1)[0][0] if ratings else 3.0

        # Category preferences
        category_counts = Counter(categories)
        top_categories = category_counts.most_common(5)
        preferred_categories = ', '.join([cat[0] for cat in top_categories]) if top_categories else "None"

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

        profile = f"""User Preference Profile Summary:
- Average Rating: {avg_rating:.1f} / 5.0 (Most common: {rating_mode})
- Rating Consistency: {consistency} (std: {rating_std:.2f})
- User Type: {user_type}
- Preferred Categories: {preferred_categories}
- Total Reviews: {len(user_reviews)}"""

        return profile

    def extract_preference_signals(self, user_reviews):
        """
        Extract preference signals from user's review history.
        Returns examples of what user likes/dislikes.
        """
        if not user_reviews:
            return "No historical data available."

        sorted_reviews = sorted(user_reviews, key=lambda x: x['stars'])

        preference_signals = []
        if len(sorted_reviews) >= 5:
            # Learn from highly rated items
            high_rated = [r for r in sorted_reviews if r['stars'] >= 4.0]
            if high_rated:
                sample = high_rated[-1]
                item = self.interaction_tool.get_item(item_id=sample.get('item_id'))
                item_name = item.get('name', 'N/A') if item else 'N/A'
                preference_signals.append(
                    f"Liked ({sample['stars']} stars): {item_name} - {sample.get('text', '')[:80]}"
                )
            
            # Learn from low rated items
            low_rated = [r for r in sorted_reviews if r['stars'] <= 2.0]
            if low_rated:
                sample = low_rated[0]
                item = self.interaction_tool.get_item(item_id=sample.get('item_id'))
                item_name = item.get('name', 'N/A') if item else 'N/A'
                preference_signals.append(
                    f"Disliked ({sample['stars']} stars): {item_name} - {sample.get('text', '')[:80]}"
                )
        else:
            for review in sorted_reviews[:3]:
                item = self.interaction_tool.get_item(item_id=review.get('item_id'))
                item_name = item.get('name', 'N/A') if item else 'N/A'
                preference_signals.append(
                    f"Review ({review['stars']} stars): {item_name} - {review.get('text', '')[:80]}"
                )
        
        return '\n\n'.join(preference_signals) if preference_signals else "No preference signals available."

    def _parse_out(self, llm_response: str) -> list:
        """
        A helper function to safely parse the LLM's response.
        There will be thoughts first then a ranked list at the end.
        """
        try:
            # Find the last occurrence of a list pattern (non-greedy, find all lists and take the last one)
            # This handles cases where there might be multiple lists in the response
            matches = list(re.finditer(r"\[.*?\]", llm_response, re.DOTALL))
            if matches:
                # Get the last match (the final list)
                match = matches[-1]
                result_str = match.group()
                result = eval(result_str)
                if isinstance(result, list) and len(result) > 0:
                    return result
            logger.error(f"Could not find valid list in output. Response: {llm_response[:500]}")
            raise ValueError("Could not find valid list in output.")
        except Exception as e:
            logger.error(f"Error parsing LLM response: {e}")
            return []

    def workflow(self):
        try:
            user_id = self.task['user_id']
            candidate_list = self.task['candidate_list']
            candidate_category = self.task.get('candidate_category', 'N/A')
            loc = self.task.get('loc', [-1, -1])
            
            user_reviews = self.interaction_tool.get_reviews(user_id=user_id)
            user_profile = self.interaction_tool.get_user(user_id=user_id)

            # Store past reviews in memory
            for review in user_reviews:
                item = self.interaction_tool.get_item(item_id=review.get('item_id'))
                item_category = item.get('category', 'Unknown') if item else 'Unknown'
                memory_entry = (
                    f"User reviewed item in category {item_category}: "
                    f"Gave {review['stars']} stars. Review: {review.get('text', '')[:100]}..."
                )
                self.memory.addMemory(memory_entry)
            
            # Build user preference profile
            preference_profile = self.analyze_user_preferences(user_reviews, user_profile)
            preference_signals = self.extract_preference_signals(user_reviews)

            # Get candidate items
            item_list = []
            for item_id in candidate_list:
                item = self.interaction_tool.get_item(item_id=item_id)
                if item:
                    keys_to_extract = ['item_id', 'name', 'stars', 'review_count', 'attributes', 
                                     'title', 'average_rating', 'rating_number', 'description', 
                                     'ratings_count', 'title_without_series', 'category', 'categories']
                    filtered_item = {key: item.get(key, 'N/A') for key in keys_to_extract if key in item}
                    item_list.append(filtered_item)

            # Query memory for relevant past experiences
            query_scenario = (
                f"The user is looking for recommendations in category: {candidate_category}. "
                f"User has {len(user_reviews)} past reviews."
            )
            relevant_memory = self.memory.retriveMemory(query_scenario)

            # Truncate data if too long
            user_str = json.dumps(user_profile, indent=2)
            if num_tokens_from_string(user_str) > 3000:
                encoding = tiktoken.get_encoding("cl100k_base")
                user_str = encoding.decode(encoding.encode(user_str)[:3000])

            items_str = json.dumps(item_list, indent=2)
            if num_tokens_from_string(items_str) > 8000:
                encoding = tiktoken.get_encoding("cl100k_base")
                items_str = encoding.decode(encoding.encode(items_str)[:8000])

            # Location info
            location_info = ""
            if loc and loc != [-1, -1]:
                location_info = f"\nUser Location: Latitude {loc[0]}, Longitude {loc[1]}\nConsider geographic proximity when ranking items."

            # Self-consistent reasoning
            task_description = f"""You are an advanced recommendation system agent.

USER PROFILE:
{user_str}

USER PREFERENCE ANALYSIS:
{preference_profile}

USER'S PREFERENCE SIGNALS (from past reviews):
{preference_signals}

CANDIDATE CATEGORY: {candidate_category}
{location_info}

RELEVANT PAST EXPERIENCES:
{relevant_memory if relevant_memory else "No directly relevant past experiences."}

CANDIDATE ITEMS TO RANK (20 items):
{items_str}

TASK:
Based on this user's profile, preference patterns, and past experiences, rank these 20 items from most to least recommended.

CRITICAL REQUIREMENTS:
1. Match items to user's preferred categories and attributes
2. Consider item quality (ratings, review count) but prioritize user preference match
3. Use relevant past experiences to inform ranking decisions
4. If location is provided, consider geographic proximity
5. Rank all 20 items considering multiple factors simultaneously

RANKING FACTORS (in order of importance):
1. Category/attribute match with user preferences
2. Item quality (ratings, review count)
3. Relevance to user's past experiences
4. Location proximity (if provided)

FORMAT (EXACTLY):
After your analysis, your response must END with a Python list. Output format:

['item_id1', 'item_id2', 'item_id3', 'item_id4', 'item_id5', 'item_id6', 'item_id7', 'item_id8', 'item_id9', 'item_id10', 'item_id11', 'item_id12', 'item_id13', 'item_id14', 'item_id15', 'item_id16', 'item_id17', 'item_id18', 'item_id19', 'item_id20']

Replace item_id1 through item_id20 with actual IDs from: {candidate_list}
Rank from most recommended (first) to least recommended (last).
The list must be the LAST line with nothing after it."""

            llm_response = self.reasoning(task_description=task_description)
            logger.info(f"Task {user_id}: LLM response length: {len(llm_response)}")

            result = self._parse_out(llm_response)
            
            # Filter out invalid items (not in candidate list) - this is reasonable cleanup
            # But we do NOT add missing items - that would be "cheating" and mask LLM failures
            candidate_set = set(candidate_list)
            filtered_result = []
            for item in result:
                if item in candidate_set:
                    filtered_result.append(item)
                else:
                    logger.warning(f"Filtered out invalid item: {item} (not in candidate list)")
            
            result = filtered_result
            
            # Log if result is incomplete (but don't fix it - let evaluation handle it)
            if len(result) != len(candidate_list):
                missing_count = len(candidate_set - set(result))
                logger.warning(f"LLM output incomplete: Expected {len(candidate_list)} items, got {len(result)}. Missing {missing_count} items.")
                logger.debug(f"LLM response (first 1000 chars): {llm_response[:1000]}")
            
            # If parsing completely failed, return empty list (let evaluation handle it)
            if not result:
                logger.error("Failed to parse any valid items from LLM response. Returning empty list.")
                logger.error(f"LLM response (first 1000 chars): {llm_response[:1000]}")
                return []
            
            with open(f'./results/generation_detail/rec_agent2.txt', 'a', encoding='utf-8') as f:
                f.write(f'\n {datetime.now()}')
                f.write(f'\n User: {user_id}')
                f.write(f'\n Raw LLM Response: {llm_response}')
                f.write(f'\n Parsed Top 5 Recommendations: {result[:5]}')
                f.write(f'\n Full Ranking: {json.dumps(result, indent=2)}\n')

            return result

        except Exception as e:
            logging.error(f"Error during workflow execution: {e}")
            import traceback
            traceback.print_exc()
            return self.task.get('candidate_list', [])

if __name__ == "__main__":
    openai_api = os.getenv("OPENAI_API_KEY")
    if not openai_api:
        raise ValueError("OPENAI_API_KEY not found in environment variables.")
    else:
        logger.info("API key successfully loaded from environment variables.")

    print("Starting simulation with AdvancedRecommendationAgent and OpenAILLM...")
    # Set the data
    task_set = "amazon"  # "goodreads" or "yelp" or "amazon"
    simulator = Simulator(data_dir="./data/processed", device="auto", cache=True)
    simulator.set_task_and_groundtruth(
        task_dir=f"./example/track2/{task_set}/tasks", 
        groundtruth_dir=f"./example/track2/{task_set}/groundtruth"
    )

    # Set the agent and LLM
    llm = OpenAILLM(api_key=openai_api, model="gpt-4.1")
    simulator.set_agent(AdvancedRecommendationAgent)
    simulator.set_llm(llm)

    # Run the simulation
    # If you don't set the number of tasks, the simulator will run all tasks.
    outputs = simulator.run_simulation(number_of_tasks=10, enable_threading=False, max_workers=1)
    
    # Evaluate the agent
    evaluation_results = simulator.evaluate()       
    with open(f'./results/evaluation/evaluation_results_track2_{task_set}_agent2.json', 'w') as f:
        time_info = {"time": datetime.now().isoformat()}
        json.dump(time_info, f, indent=4)
        f.write('\n')
        json.dump(evaluation_results, f, indent=4)
        f.write('\n')

    # Get evaluation history
    evaluation_history = simulator.get_evaluation_history()
    print(f"Evaluation results: {evaluation_results}")

