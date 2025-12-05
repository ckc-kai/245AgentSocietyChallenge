import os
import re
import json
import logging
import tiktoken
from datetime import datetime
from dotenv import load_dotenv

load_dotenv(dotenv_path="./secrets.env")
logger = logging.getLogger("websocietysimulator")

from websocietysimulator.agent import RecommendationAgent
from websocietysimulator.agent.modules import MemoryDILU, ReasoningCOT
from websocietysimulator.llm import LLMBase, OpenAILLM
from websocietysimulator.simulator import Simulator


class SimplifiedMemoryAgent(RecommendationAgent):
    def __init__(self, llm: LLMBase):
        super().__init__(llm=llm)
        self.memory = MemoryDILU(llm=self.llm)
        self.reasoning = ReasoningCOT(
            profile_type_prompt='',
            memory=self.memory,
            llm=self.llm
        )
        self.encoding = tiktoken.get_encoding("cl100k_base")
    
    def truncate_text(self, text: str, max_tokens: int) -> str:
        tokens = self.encoding.encode(text)
        if len(tokens) > max_tokens:
            return self.encoding.decode(tokens[:max_tokens])
        return text
    
    def workflow(self):
        user_id = self.task['user_id']
        candidate_ids = self.task['candidate_list']
        
        user_profile = str(self.interaction_tool.get_user(user_id=user_id))
        user_profile = self.truncate_text(user_profile, 2500)
        
        reviews = self.interaction_tool.get_reviews(user_id=user_id)
        recent_reviews = reviews[-25:] if len(reviews) > 25 else reviews
        
        for review in recent_reviews:
            memory_text = f"User reviewed {review.get('item_id', '')} with {review.get('stars', 0)} stars"
            self.memory.addMemory(memory_text)
        
        review_summary = []
        for review in recent_reviews:
            item_id = review.get('item_id', '')
            rating = review.get('stars', 'N/A')
            text = review.get('text', '')[:80]
            review_summary.append(f"{item_id}: {rating}★ - {text}")
        
        review_text = "\n".join(review_summary)
        review_text = self.truncate_text(review_text, 3500)
        
        candidate_items = []
        for item_id in candidate_ids:
            try:
                item = self.interaction_tool.get_item(item_id=item_id)
                item_id_val = item.get('item_id', '')
                name = item.get('name', item.get('title', ''))[:50]
                rating = item.get('stars', item.get('average_rating', 'N/A'))
                categories = item.get('categories', item.get('category', ''))
                
                if isinstance(categories, list):
                    cat_str = ', '.join(categories[:2])
                else:
                    cat_str = str(categories)[:40]
                
                candidate_items.append(f"{item_id_val}: {name} | {rating}★ | {cat_str}")
            except Exception as e:
                logger.error(f"Error getting item {item_id}: {e}")
        
        candidates_text = "\n".join(candidate_items)
        candidates_text = self.truncate_text(candidates_text, 4000)
        
        memory_query = f"User rating patterns and preferences"
        retrieved_memory = self.memory.retriveMemory(memory_query)
        
        task_description = f'''You are ranking 20 items for a user based on their preferences.

USER PROFILE:
{user_profile}

PAST REVIEWS:
{review_text}

CANDIDATES (format: ID: Name | Rating | Categories):
{candidates_text}

INSTRUCTIONS:
1. Analyze which item categories/attributes the user prefers
2. Rank items matching those preferences higher
3. Consider item quality (ratings)

CRITICAL: Output ONLY a valid Python list. No explanations, no text before or after.
Format: ['ID1', 'ID2', 'ID3', 'ID4', 'ID5', 'ID6', 'ID7', 'ID8', 'ID9', 'ID10', 'ID11', 'ID12', 'ID13', 'ID14', 'ID15', 'ID16', 'ID17', 'ID18', 'ID19', 'ID20']

Your ranked list:'''
        
        result = self.reasoning(task_description)
        
        try:
            matches = list(re.finditer(r'\[[\s\S]*?\]', result))
            
            if matches:
                for match in reversed(matches):
                    try:
                        result_str = match.group()
                        result_str = result_str.replace('\n', '').replace('  ', ' ')
                        result_list = eval(result_str)
                        
                        if isinstance(result_list, list):
                            valid_items = [str(item_id).strip() for item_id in result_list 
                                         if str(item_id).strip() in candidate_ids]
                            
                            if valid_items:
                                missing_items = [item_id for item_id in candidate_ids 
                                               if item_id not in valid_items]
                                final_ranking = valid_items + missing_items
                                return final_ranking[:20]
                    except:
                        continue
        except Exception as e:
            logger.error(f"Parsing error: {e}")
        
        logger.warning("Failed to parse ranking, returning original candidate list")
        return candidate_ids


if __name__ == "__main__":
    openai_api = os.getenv("OPENAI_API_KEY")
    if not openai_api:
        raise ValueError("OPENAI_API_KEY not found in environment variables.")
    
    task_set = "amazon"
    simulator = Simulator(data_dir="./data/processed", device="auto", cache=True)
    simulator.set_task_and_groundtruth(
        task_dir=f"./example/track2/{task_set}/tasks",
        groundtruth_dir=f"./example/track2/{task_set}/groundtruth"
    )
    
    llm = OpenAILLM(api_key=openai_api, model="gpt-4o-mini")
    simulator.set_agent(SimplifiedMemoryAgent)
    simulator.set_llm(llm)
    
    outputs = simulator.run_simulation(number_of_tasks=400, enable_threading=False, max_workers=1)
    evaluation_results = simulator.evaluate()
    with open(f'./results/evaluation/evaluation_results_track2_{task_set}_agent4.json', 'w') as f:
        time_info = {"time": datetime.now().isoformat()}
        json.dump(time_info, f, indent=4)
        f.write('\n')
        json.dump(evaluation_results, f, indent=4)
        f.write('\n')
    
    print(f"Evaluation results: {evaluation_results}")

