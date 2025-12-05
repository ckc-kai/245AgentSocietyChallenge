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

class ScopedMemory(MemoryBase):
    """
    Memory that retrieves relevant past reviews based on semantic similarity
    BUT filters/prioritizes by the specific user to avoid pollution.
    """
    def __init__(self, llm):
        super().__init__(memory_type='scoped', llm=llm)

    def retriveMemory(self, query_scenario: str, user_id: str, k: int = 3):
        if self.scenario_memory._collection.count() == 0:
            return ''
            
        # We search for memories relevant to the item/category
        similarity_results = self.scenario_memory.similarity_search_with_score(query_scenario, k=k*2) # Fetch more to filter
        
        relevant_memories = []
        for result, score in similarity_results:
            # Only include memories from THIS user
            if result.metadata.get('user_id') == user_id:
                relevant_memories.append(result.metadata['task_trajectory'])
        
        # Take top k from the filtered list
        final_memories = relevant_memories[:k]
        
        if not final_memories:
            return "No relevant past reviews found for this user in this context."
            
        formatted = []
        for i, mem in enumerate(final_memories, 1):
            formatted.append(f"[Past Experience {i}]: {mem}")
            
        return '\n\n'.join(formatted)

    def addMemory(self, current_situation, user_id):
        memory_doc = Document(
            page_content=current_situation,
            metadata={
                "task_name": current_situation[:100],
                "task_trajectory": current_situation,
                "user_id": user_id # Tag with user_id
            }
        )
        self.scenario_memory.add_documents([memory_doc])

class RatingPredictor(ReasoningBase):
    """
    Specialized module to predict the star rating BEFORE generating text.
    """
    def __call__(self, user_profile, item_details, category_stats, global_stats):
        prompt = f"""Predict the star rating (1-5) this user would give to this business.

USER PROFILE:
{json.dumps(user_profile, indent=2)}

USER STATS:
- Global Average Rating: {global_stats['avg']:.2f}
- Category ({item_details.get('category', 'Unknown')}) Average: {category_stats['avg']:.2f} (Count: {category_stats['count']})

BUSINESS:
{json.dumps(item_details, indent=2)}

TASK:
Predict the star rating (1.0 to 5.0) based strictly on the user's history and the business type.
If the user is generally critical, predict lower. If generous, predict higher.
If the user hates this category, predict lower.

OUTPUT FORMAT:
Rating: [Number]
Reason: [Short explanation]"""

        messages = [{"role": "user", "content": prompt}]
        response = self.llm(messages=messages, temperature=0.1, max_tokens=100)
        
        try:
            if isinstance(response, list):
                response = response[0]
            
            # Simple parsing
            lines = response.split('\n')
            rating_line = [l for l in lines if 'Rating:' in l][0]
            rating = float(rating_line.split(':')[1].strip())
            return min(max(rating, 1.0), 5.0)
        except Exception as e:
            logger.error(f"Rating prediction failed: {e}")
            # Fallback to category average or global average
            return category_stats['avg'] if category_stats['count'] > 0 else global_stats['avg']

class StyleAwareGenerator(ReasoningBase):
    """
    Generates the review text using specific style samples that match the target rating.
    """
    def __call__(self, target_rating, user_profile, item_details, style_samples, memory_context):
        prompt = f"""Write a review for this business acting as the user.

TARGET RATING: {target_rating} Stars (You MUST give this rating)

USER PROFILE:
{json.dumps(user_profile, indent=2)}

STYLE SAMPLES (How this user writes {int(target_rating)}-star reviews):
{style_samples}

BUSINESS:
{json.dumps(item_details, indent=2)}

RELEVANT MEMORIES:
{memory_context}

INSTRUCTIONS:
1. Write a review that matches the USER'S VOICE from the style samples.
2. The length should be similar to the samples.
3. Focus on what this user typically cares about (price, service, food, etc.).
4. The sentiment must match the {target_rating} star rating.

FORMAT:
stars: {target_rating}
review: [Your review text]"""

        messages = [{"role": "user", "content": prompt}]
        return self.llm(messages=messages, temperature=0.7, max_tokens=300)

class AdvancedModelingAgent(SimulationAgent):
    def __init__(self, llm):
        super().__init__(llm=llm)
        self.memory = ScopedMemory(llm=self.llm)
        self.rating_predictor = RatingPredictor(profile_type_prompt='', memory=None, llm=self.llm)
        self.generator = StyleAwareGenerator(profile_type_prompt='', memory=None, llm=self.llm)
    
    def _get_user_stats(self, reviews):
        if not reviews:
            return {'avg': 3.0, 'std': 0.0, 'count': 0}, {}
            
        ratings = [r['stars'] for r in reviews]
        global_stats = {
            'avg': np.mean(ratings),
            'std': np.std(ratings),
            'count': len(ratings)
        }
        
        # Category stats
        cat_stats = {}
        for r in reviews:
            cat = r.get('categories', 'Unknown')
            # Handle list of categories or string
            if isinstance(cat, str):
                cats = [c.strip() for c in cat.split(',')]
            else:
                cats = ['Unknown']
                
            for c in cats:
                if c not in cat_stats:
                    cat_stats[c] = []
                cat_stats[c].append(r['stars'])
                
        final_cat_stats = {}
        for c, r_list in cat_stats.items():
            final_cat_stats[c] = {
                'avg': np.mean(r_list),
                'count': len(r_list)
            }
            
        return global_stats, final_cat_stats

    def _get_style_samples(self, reviews, target_rating):
        if not reviews:
            return "No past reviews."
            
        # Find reviews closest to target rating
        # Sort by distance to target rating
        sorted_reviews = sorted(reviews, key=lambda x: abs(x['stars'] - target_rating))
        
        # Take top 3 closest
        closest = sorted_reviews[:3]
        
        samples = []
        for r in closest:
            samples.append(f"[{r['stars']} Stars]: {r['text'][:200]}...")
            
        return "\n\n".join(samples)

    def _parse_out(self, llm_response: str) -> dict:
        try:
            # Find the *last* occurrence of "stars:" and "review:"
            last_stars_index = llm_response.rfind('stars:')
            last_review_index = llm_response.rfind('review:')
            
            if last_stars_index == -1 or last_review_index == -1:
                # Fallback for simple format
                if 'stars:' in llm_response and 'review:' in llm_response:
                     last_stars_index = llm_response.find('stars:')
                     last_review_index = llm_response.find('review:')
                else:
                    raise ValueError("Could not find 'stars:' or 'review:'")

            response_subset = llm_response[min(last_stars_index, last_review_index):]
            stars_line = [line for line in response_subset.split('\n') if 'stars:' in line][0]
            review_line = [line for line in response_subset.split('\n') if 'review:' in line][0]

            stars = float(stars_line.split(':', 1)[1].strip())
            review = review_line.split(':', 1)[1].strip()

            return {"stars": min(max(stars, 1.0), 5.0), "review": review[:512]}
        except Exception as e:
            logger.error(f"Parsing error: {e}")
            return {"stars": 3.0, "review": "Error parsing review."}

    def workflow(self):
        try:
            user_id = self.task['user_id']
            item_id = self.task['item_id']
            
            # 1. Fetch Data
            user_reviews = self.interaction_tool.get_reviews(user_id=user_id)
            user_profile = self.interaction_tool.get_user(user_id=user_id)
            item_details = self.interaction_tool.get_item(item_id=item_id)
            
            # 2. Populate Memory (Scoped)
            # In a real efficient system, we wouldn't re-add every time, but for this sim it's fine
            # or we check if already added. For safety in this framework, we just add current batch.
            # Ideally, the simulator persists memory, but here we might be re-initializing.
            # Let's assume we need to populate relevant history.
            for review in user_reviews:
                mem = f"Category: {review.get('categories', 'Unknown')}, Rating: {review['stars']}, Review: {review['text']}"
                self.memory.addMemory(mem, user_id)
            
            # 3. Analyze Stats
            global_stats, cat_stats = self._get_user_stats(user_reviews)
            
            # Get current item category stats
            item_cat = item_details.get('category', 'Unknown')
            # Try to find matching category in stats
            current_cat_stat = {'avg': global_stats['avg'], 'count': 0}
            
            # Simple fuzzy match for category
            if isinstance(item_cat, str):
                for c, s in cat_stats.items():
                    if c.lower() in item_cat.lower() or item_cat.lower() in c.lower():
                        current_cat_stat = s
                        break
            
            # 4. Predict Rating
            predicted_rating = self.rating_predictor(
                user_profile, item_details, current_cat_stat, global_stats
            )
            
            # 5. Retrieve Context
            query = f"{item_details.get('name')} {item_details.get('category')}"
            relevant_memory = self.memory.retriveMemory(query, user_id)
            
            # 6. Select Style Samples
            style_samples = self._get_style_samples(user_reviews, predicted_rating)
            
            # 7. Generate Review
            llm_response = self.generator(
                predicted_rating, user_profile, item_details, style_samples, relevant_memory
            )
            
            result = self._parse_out(llm_response)
            
            # Logging
            with open(f'./results/generation_detail/analyzer_v3.txt', 'a') as f:
                f.write(f'\n--- Task {user_id} ---\n')
                f.write(f'Predicted Rating: {predicted_rating}\n')
                f.write(f'Final Output: {json.dumps(result)}\n')
                
            return result

        except Exception as e:
            logger.error(f"Workflow error: {e}")
            import traceback
            traceback.print_exc()
            return {"stars": 3.0, "review": "Workflow error."}

if __name__ == "__main__":
    claude_api = os.getenv("CLAUDE_API_KEY")
    openai_api = os.getenv("OPENAI_API_KEY")
    if not claude_api or not openai_api:
        raise ValueError("API keys not found.")
        
    task_set = "amazon"
    simulator = Simulator(data_dir="/Users/ckc/Desktop/UCLA/2025fall/245/AgentSocietyChallenge/data/processed", device="auto", cache=True)
    simulator.set_task_and_groundtruth(task_dir=f"./example/track1/{task_set}/tasks", groundtruth_dir=f"./example/track1/{task_set}/groundtruth")
    
    llm = ClaudeLLM(api_key=claude_api, model="claude-sonnet-4-20250514", openai_api_key=openai_api)
    simulator.set_agent(AdvancedModelingAgent)
    simulator.set_llm(llm)
    
    outputs = simulator.run_simulation(number_of_tasks=10, enable_threading=True, max_workers=10)
    evaluation_results = simulator.evaluate()
    
    with open(f'./results/evaluation/evaluation_results_track1_{task_set}_model3.json', 'a') as f:
        entry = {
            "time": datetime.now().isoformat(),
            "results": evaluation_results
        }
        f.write(json.dumps(entry, indent=4) + "\n")
        
    print(f"Evaluation Results: {json.dumps(evaluation_results, indent=2)}")
