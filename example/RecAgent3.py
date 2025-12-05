import os
import re
import json
import logging
from typing import List, Dict, Any
from collections import Counter, defaultdict
from dotenv import load_dotenv
import tiktoken

load_dotenv(dotenv_path="./secrets.env")

from websocietysimulator.agent import RecommendationAgent
from websocietysimulator.llm import LLMBase, OpenAILLM
from websocietysimulator.simulator import Simulator

logger = logging.getLogger(__name__)


class CategoryAwareRecommendationAgent(RecommendationAgent):
    def __init__(self, llm: LLMBase):
        super().__init__(llm=llm)
        self.encoding = tiktoken.get_encoding("cl100k_base")
    
    def extract_category_preferences(self, reviews: List[Dict]) -> Dict[str, Any]:
        category_ratings = defaultdict(list)
        high_rated_items = []  # 4+ stars
        low_rated_items = []   # 2- stars
        
        for review in reviews:
            rating = review.get('stars', 0)
            item_id = review.get('item_id', '')
            
            # Get item details to extract category
            try:
                item = self.interaction_tool.get_item(item_id=item_id)
                categories = item.get('categories', item.get('category', ''))
                
                # Handle both string and list categories
                if isinstance(categories, list):
                    for cat in categories:
                        category_ratings[cat].append(rating)
                else:
                    category_ratings[categories].append(rating)
                
                # Track extreme preferences
                if rating >= 4.0:
                    high_rated_items.append({
                        'item_id': item_id,
                        'name': item.get('name', item.get('title', '')),
                        'rating': rating,
                        'review': review.get('text', '')[:150]
                    })
                elif rating <= 2.0:
                    low_rated_items.append({
                        'item_id': item_id,
                        'name': item.get('name', item.get('title', '')),
                        'rating': rating,
                        'review': review.get('text', '')[:150]
                    })
            except Exception as e:
                logger.debug(f"Error processing review: {e}")
                continue
        
        # Calculate category preferences
        category_preferences = {}
        for cat, ratings in category_ratings.items():
            if ratings:
                category_preferences[cat] = {
                    'avg_rating': sum(ratings) / len(ratings),
                    'count': len(ratings)
                }
        
        # Sort to get top categories
        top_categories = sorted(
            category_preferences.items(),
            key=lambda x: (x[1]['avg_rating'], x[1]['count']),
            reverse=True
        )[:5]
        
        return {
            'category_preferences': dict(top_categories),
            'high_rated_items': high_rated_items[:5],
            'low_rated_items': low_rated_items[:3]
        }
    
    def match_candidate_to_preferences(self, candidate_items: List[Dict], preferences: Dict) -> List[Dict]:
        scored_candidates = []
        category_prefs = preferences.get('category_preferences', {})
        
        for item in candidate_items:
            item_id = item.get('item_id', '')
            categories = item.get('categories', item.get('category', ''))
            
            # Calculate category match score
            match_score = 0
            matched_categories = []
            
            if isinstance(categories, list):
                for cat in categories:
                    if cat in category_prefs:
                        match_score += category_prefs[cat]['avg_rating']
                        matched_categories.append(cat)
            else:
                if categories in category_prefs:
                    match_score = category_prefs[categories]['avg_rating']
                    matched_categories.append(categories)
            
            scored_candidates.append({
                'item': item,
                'match_score': match_score,
                'matched_categories': matched_categories
            })
        
        # Sort by match score
        scored_candidates.sort(key=lambda x: x['match_score'], reverse=True)
        return scored_candidates
    
    def truncate_text(self, text: str, max_tokens: int) -> str:
        tokens = self.encoding.encode(text)
        if len(tokens) > max_tokens:
            return self.encoding.decode(tokens[:max_tokens])
        return text
    
    def stage1_shortlist(self, candidates: List[Dict], preferences: Dict, user_profile: str) -> List[str]:
        # Prepare preference summary
        high_rated = preferences.get('high_rated_items', [])
        low_rated = preferences.get('low_rated_items', [])
        
        likes_summary = "\n".join([
            f"- {item['name']} ({item['rating']} stars): {item['review']}"
            for item in high_rated
        ])
        
        dislikes_summary = "\n".join([
            f"- {item['name']} ({item['rating']} stars): {item['review']}"
            for item in low_rated
        ])
        
        candidate_summary = []
        for i, cand in enumerate(candidates[:20]):
            item = cand['item']
            item_id = item.get('item_id', '')
            name = item.get('name', item.get('title', ''))
            rating = item.get('stars', item.get('average_rating', 'N/A'))
            categories = item.get('categories', item.get('category', ''))
            matched = cand.get('matched_categories', [])
            
            cat_str = ', '.join(matched) if matched else str(categories)
            candidate_summary.append(f"{i+1}. {item_id} - {name} | Rating: {rating} | Categories: {cat_str}")
        
        candidate_text = "\n".join(candidate_summary)
        
        # Truncate to fit context
        user_profile = self.truncate_text(user_profile, 2000)
        candidate_text = self.truncate_text(candidate_text, 5000)
        
        prompt = f"""You are helping a user find items they'll love based on their preferences.

USER PROFILE:
{user_profile}

USER LIKES (High-rated items):
{likes_summary}

USER DISLIKES (Low-rated items):
{dislikes_summary}

CANDIDATES TO RANK:
{candidate_text}

TASK: Analyze the user's preferences and select the TOP 10 item IDs from the candidates that best match their interests.

Consider:
1. Category/type match with user's liked items
2. Avoid characteristics similar to disliked items
3. Item quality (ratings)
4. Specific preferences mentioned in reviews

Output ONLY a Python list of 10 item IDs in order of preference:
['item_id1', 'item_id2', 'item_id3', 'item_id4', 'item_id5', 'item_id6', 'item_id7', 'item_id8', 'item_id9', 'item_id10']
"""
        
        messages = [{"role": "user", "content": prompt}]
        
        try:
            response = self.llm(
                messages=messages,
                temperature=0.1,
                max_tokens=500
            )
            
            # Parse response
            match = re.search(r'\[.*?\]', response, re.DOTALL)
            if match:
                result_list = eval(match.group())
                # Validate all items are in candidate list
                valid_ids = [item['item']['item_id'] for item in candidates]
                filtered = [item_id for item_id in result_list if item_id in valid_ids]
                return filtered[:10]
        except Exception as e:
            logger.error(f"Stage 1 error: {e}")
        
        return [cand['item']['item_id'] for cand in candidates[:10]]
    
    def stage2_refine_ranking(self, shortlist: List[str], all_candidates: List[Dict], 
                             preferences: Dict, user_profile: str) -> List[str]:
        # Get all candidate IDs
        all_ids = [cand['item']['item_id'] for cand in all_candidates]
        remaining_ids = [item_id for item_id in all_ids if item_id not in shortlist]
        
        # Prepare focused prompt
        shortlist_items = []
        for item_id in shortlist:
            for cand in all_candidates:
                if cand['item']['item_id'] == item_id:
                    item = cand['item']
                    name = item.get('name', item.get('title', ''))
                    shortlist_items.append(f"{item_id} - {name}")
                    break
        
        remaining_items = []
        for item_id in remaining_ids:
            for cand in all_candidates:
                if cand['item']['item_id'] == item_id:
                    item = cand['item']
                    name = item.get('name', item.get('title', ''))
                    remaining_items.append(f"{item_id} - {name}")
                    break
        
        prompt = f"""You previously identified these TOP 10 items as best matches:
{chr(10).join(shortlist_items)}

And these 10 as lower priority:
{chr(10).join(remaining_items)}

Now create the FINAL ranking of all 20 items. The top 10 should generally come from your shortlist, but you can adjust order and include lower priority items if they're better fits.

Output ONLY a Python list of all 20 item IDs in ranked order:
['id1', 'id2', ..., 'id20']
"""
        
        messages = [{"role": "user", "content": prompt}]
        
        try:
            response = self.llm(
                messages=messages,
                temperature=0.1,
                max_tokens=500
            )
            
            # Parse response
            match = re.search(r'\[.*?\]', response, re.DOTALL)
            if match:
                result_list = eval(match.group())
                # Validate and filter
                filtered = [item_id for item_id in result_list if item_id in all_ids]
                
                # Add any missing items at the end
                missing = [item_id for item_id in all_ids if item_id not in filtered]
                final_ranking = filtered + missing
                
                return final_ranking[:20]
        except Exception as e:
            logger.error(f"Stage 2 error: {e}")
        
        return shortlist + remaining_ids
    
    def workflow(self):
        user_id = self.task['user_id']
        candidate_ids = self.task['candidate_list']
        
        # Get user profile
        user_profile = str(self.interaction_tool.get_user(user_id=user_id))
        
        all_reviews = self.interaction_tool.get_reviews(user_id=user_id)
        recent_reviews = all_reviews[-50:] if len(all_reviews) > 50 else all_reviews
        
        # Extract category-aware preferences
        preferences = self.extract_category_preferences(recent_reviews)
        
        # Get candidate items with full details
        candidate_items = []
        for item_id in candidate_ids:
            try:
                item = self.interaction_tool.get_item(item_id=item_id)
                candidate_items.append(item)
            except Exception as e:
                logger.error(f"Error getting item {item_id}: {e}")
        
        # Match candidates to preferences
        scored_candidates = self.match_candidate_to_preferences(candidate_items, preferences)
        
        # Stage 1: Shortlist top 10
        shortlist = self.stage1_shortlist(scored_candidates, preferences, user_profile)
        
        # Stage 2: Refine to full ranking
        final_ranking = self.stage2_refine_ranking(shortlist, scored_candidates, preferences, user_profile)
        
        logger.info(f"Final ranking (first 5): {final_ranking[:5]}")
        
        return final_ranking


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
    simulator.set_agent(CategoryAwareRecommendationAgent)
    simulator.set_llm(llm)
    
    outputs = simulator.run_simulation(number_of_tasks=400, enable_threading=False, max_workers=1)
    evaluation_results = simulator.evaluate()
    
    with open(f'./results/evaluation/evaluation_results_track2_{task_set}_agent3.json', 'w') as f:
        json.dump(evaluation_results, f, indent=4)
    
    print(f"The evaluation_results is: {evaluation_results}")

