import json
import logging
from websocietysimulator import Simulator
from websocietysimulator.agent import RecommendationAgent
from websocietysimulator.agent.modules.memory_modules import MemoryGenerative
from websocietysimulator.agent.modules.reasoning_modules import ReasoningCOT
from websocietysimulator.llm import OpenAILLM
from websocietysimulator.llm.llm import LLMBase
from datetime import datetime
import os
import re
import tiktoken
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

class MyRecommendationAgent(RecommendationAgent):
    def __init__(self, llm: LLMBase):
        super().__init__(llm=llm)
        self.memory = MemoryGenerative(llm=self.llm)
        self.reasoning = ReasoningCOT(profile_type_prompt='', memory=None, llm=self.llm)

    def _parse_out(self, llm_response: str) -> list:
        try:
            # Find the last occurrence of a list pattern
            match = re.search(r"\[.*\]", llm_response, re.DOTALL)
            if match:
                result_str = match.group()
                # Try to evaluate as Python list
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
            # Fetch user and task data
            user_id = self.task['user_id']
            candidate_list = self.task['candidate_list']
            candidate_category = self.task.get('candidate_category', 'N/A')
            loc = self.task.get('loc', [-1, -1])

            # Add user's past reviews to memory
            user_reviews = self.interaction_tool.get_reviews(user_id=user_id)
            for review in user_reviews:
                memory_entry = (
                    f"User {review['user_id']} previously reviewed item {review.get('item_id', 'N/A')}: "
                    f"(Stars: {review['stars']}, Review: {review.get('text', '')[:100]})"
                )
                self.memory.addMemory(memory_entry)

            # Gather current task info
            user_profile = self.interaction_tool.get_user(user_id=user_id)
            
            # Get candidate items information
            item_list = []
            for item_id in candidate_list:
                item = self.interaction_tool.get_item(item_id=item_id)
                if item:
                    # Extract relevant keys
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
            logger.info(f"Task {user_id}: Found relevant memory: {relevant_memory[:100] if relevant_memory else 'None'}...")

            # Truncate data if too long to avoid token limits
            user_str = json.dumps(user_profile, indent=2)
            if num_tokens_from_string(user_str) > 3000:
                encoding = tiktoken.get_encoding("cl100k_base")
                user_str = encoding.decode(encoding.encode(user_str)[:3000])

            items_str = json.dumps(item_list, indent=2)
            if num_tokens_from_string(items_str) > 8000:
                encoding = tiktoken.get_encoding("cl100k_base")
                items_str = encoding.decode(encoding.encode(items_str)[:8000])

            # COT Reasoning
            location_info = ""
            if loc and loc != [-1, -1]:
                location_info = f"\nUser Location: Latitude {loc[0]}, Longitude {loc[1]}"

            task_description = f"""
            You are a recommendation system agent.

            User Profile:
            {user_str}

            Candidate Category: {candidate_category}
            {location_info}

            User's Most Relevant Past Experiences (from memory):
            {relevant_memory if relevant_memory else "No directly relevant past experiences."}

            Candidate Items to Rank (20 items):
            {items_str}

            INSTRUCTIONS:
            Rank these 20 items from most recommended to least recommended for this user.

            First, think step-by-step (Chain of Thought) about your ranking decision:
            1. Analyze the user's profile and past review history. What are their preferences?
            2. Consider the candidate category and how it matches user's interests.
            3. For each candidate item, evaluate:
               - Category/attribute match with user preferences
               - Item quality (ratings, review count)
               - Relevance to user's past experiences
               - Location proximity (if location is provided)
            4. Rank all 20 items from most to least recommended.

            CRITICAL OUTPUT FORMAT:
            You MUST output ONLY a Python list, nothing else. No explanations, no reasoning, just the list.
            
            Output format (copy this exactly and fill in the item IDs):
            ['item_id1', 'item_id2', 'item_id3', 'item_id4', 'item_id5', 'item_id6', 'item_id7', 'item_id8', 'item_id9', 'item_id10', 'item_id11', 'item_id12', 'item_id13', 'item_id14', 'item_id15', 'item_id16', 'item_id17', 'item_id18', 'item_id19', 'item_id20']
            
            The list must contain EXACTLY these 20 item IDs in ranked order: {candidate_list}
            DO NOT add any text before or after the list. The list must be the ONLY output.
            """
            
            llm_response = self.reasoning(task_description=task_description)
            logger.info(f"Task {user_id}: LLM response length: {len(llm_response)}")

            result = self._parse_out(llm_response)
            
            candidate_set = set(candidate_list)
            filtered_result = []
            for item in result:
                if item in candidate_set:
                    filtered_result.append(item)
            
            result = filtered_result
            
            if not result:
                logger.error("Failed to parse any valid items from LLM response.")
                return []
            
            return result
        except Exception as e:
            logger.error(f"Error during workflow execution: {e}")
            import traceback
            traceback.print_exc()
            # Return empty list on error
            return []

if __name__ == "__main__":
    openai_api = os.getenv("OPENAI_API_KEY")
    if not openai_api:
        raise ValueError("OPENAI_API_KEY not found in environment variables.")
    else:
        logger.info("API key successfully loaded from environment variables.")

    task_set = "amazon"
    simulator = Simulator(data_dir="./data/processed", device="auto", cache=True)
    simulator.set_task_and_groundtruth(
        task_dir=f"./example/track2/{task_set}/tasks", 
        groundtruth_dir=f"./example/track2/{task_set}/groundtruth"
    )

    llm = OpenAILLM(api_key=openai_api, model="gpt-4.1")
    simulator.set_agent(MyRecommendationAgent)
    simulator.set_llm(llm)

    outputs = simulator.run_simulation(number_of_tasks=400, enable_threading=False, max_workers=1)
    evaluation_results = simulator.evaluate()       
    with open(f'./results/evaluation/evaluation_results_track2_{task_set}_agent1.json', 'w') as f:
        time_info = {"time": datetime.now().isoformat()}
        json.dump(time_info, f, indent=4)
        f.write('\n')
        json.dump(evaluation_results, f, indent=4)
        f.write('\n')

    print(f"Evaluation results: {evaluation_results}")

