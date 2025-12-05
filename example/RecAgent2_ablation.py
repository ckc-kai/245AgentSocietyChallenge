import os
import re
import json
import logging
import tiktoken
from datetime import datetime
from collections import Counter
from dotenv import load_dotenv

load_dotenv(dotenv_path="./secrets.env")
logger = logging.getLogger("websocietysimulator")
logger.propagate = False

from websocietysimulator.agent import RecommendationAgent
from websocietysimulator.agent.modules import MemoryBase, ReasoningCOT
from websocietysimulator.agent.modules.reasoning_modules import ReasoningBase
from langchain.docstore.document import Document
from websocietysimulator.llm import LLMBase, OpenAILLM
from websocietysimulator.simulator import Simulator

ABLATION_VARIANT = "full"


class EfficientMemory(MemoryBase):

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
    def __init__(self, profile_type_prompt, memory, llm, n_sample=1):
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
            return responses

        if self.n_sample == 1 or len(responses) == 1:
            return responses[0] if isinstance(responses, list) else responses
        
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
            return responses[0] if responses else ""
        
        if len(parsed_rankings) == 1:
            return valid_responses[0]
        
        return valid_responses[0]


class AdvancedRecommendationAgentAblation(RecommendationAgent):
    def __init__(self, llm: LLMBase, variant: str = "full"):
        super().__init__(llm=llm)
        self.variant = variant
        
        if variant != "no_memory":
            self.memory = EfficientMemory(llm=self.llm)
        else:
            self.memory = None
        
        if variant != "no_consistency":
            self.reasoning = ConsistentReasoning(
                profile_type_prompt='',
                memory=self.memory if variant != "no_memory" else None,
                llm=self.llm,
                n_sample=1
            )
        else:
            self.reasoning = ReasoningCOT(
                profile_type_prompt='',
                memory=self.memory if variant != "no_memory" else None,
                llm=self.llm
            )
        
        self.encoding = tiktoken.get_encoding("cl100k_base")
    
    def analyze_user_preferences(self, reviews):
        if self.variant == "no_profiling":
            return None
        
        if not reviews:
            return None
        
        ratings = []
        categories = []
        
        for review in reviews:
            rating = review.get('stars', 0)
            if rating > 0:
                ratings.append(rating)
            
            # Get item to extract category
            try:
                item_id = review.get('item_id', '')
                item = self.interaction_tool.get_item(item_id=item_id)
                item_categories = item.get('categories', item.get('category', ''))
                
                if isinstance(item_categories, list):
                    categories.extend(item_categories)
                else:
                    categories.append(item_categories)
            except:
                pass
        
        if not ratings:
            return None
        
        avg_rating = sum(ratings) / len(ratings)
        rating_std = (sum((r - avg_rating) ** 2 for r in ratings) / len(ratings)) ** 0.5
        rating_counts = Counter(ratings)
        most_common_rating = rating_counts.most_common(1)[0][0]
        
        # User type classification
        if avg_rating >= 4.0:
            user_type = "generally positive and generous"
        elif avg_rating <= 2.0:
            user_type = "critical and discerning"
        else:
            user_type = "balanced and moderate"
        
        # Consistency classification
        if rating_std < 0.5:
            consistency = "very consistent"
        elif rating_std < 1.0:
            consistency = "moderately consistent"
        else:
            consistency = "variable"
        
        # Category preferences
        category_counts = Counter(categories)
        top_categories = [cat for cat, _ in category_counts.most_common(5)]
        
        return {
            'avg_rating': avg_rating,
            'rating_std': rating_std,
            'most_common_rating': most_common_rating,
            'user_type': user_type,
            'consistency': consistency,
            'top_categories': top_categories,
            'total_reviews': len(reviews)
        }
    
    def extract_preference_signals(self, reviews):
        if self.variant == "no_profiling":
            return {'liked': [], 'disliked': []}
        
        liked_items = []
        disliked_items = []
        
        for review in reviews:
            rating = review.get('stars', 0)
            item_id = review.get('item_id', '')
            review_text = review.get('text', '')[:80]
            
            try:
                item = self.interaction_tool.get_item(item_id=item_id)
                item_name = item.get('name', item.get('title', ''))
                
                if rating >= 4.0:
                    liked_items.append(f"{item_name} ({rating}★): {review_text}")
                elif rating <= 2.0:
                    disliked_items.append(f"{item_name} ({rating}★): {review_text}")
            except:
                pass
        
        return {
            'liked': liked_items[:5],
            'disliked': disliked_items[:3]
        }
    
    def truncate_text(self, text: str, max_tokens: int) -> str:
        tokens = self.encoding.encode(text)
        if len(tokens) > max_tokens:
            return self.encoding.decode(tokens[:max_tokens])
        return text
    
    def workflow(self):
        user_id = self.task['user_id']
        candidate_ids = self.task['candidate_list']
        
        # Get user profile
        user_profile = str(self.interaction_tool.get_user(user_id=user_id))
        user_profile = self.truncate_text(user_profile, 2500)
        
        # Get user reviews (limit to last 20)
        all_reviews = self.interaction_tool.get_reviews(user_id=user_id)
        recent_reviews = all_reviews[-20:] if len(all_reviews) > 20 else all_reviews
        
        # Add to memory if memory is enabled
        memory_context = ""
        if self.memory is not None:
            for review in recent_reviews:
                memory_entry = f"User {user_id} reviewed item with {review.get('stars', 0)} stars"
                self.memory.addMemory(memory_entry)
            
            # Query memory
            memory_query = f"User {user_id} preferences and rating patterns"
            retrieved_memory = self.memory.retriveMemory(memory_query)
            if retrieved_memory:
                memory_context = f"\nPast Patterns: {self.truncate_text(retrieved_memory, 500)}"
        
        # Get user preferences if profiling is enabled
        profile_summary = ""
        preference_signals = {'liked': [], 'disliked': []}
        
        if self.variant != "no_profiling":
            user_prefs = self.analyze_user_preferences(recent_reviews)
            if user_prefs:
                profile_summary = f"""
User Behavior: {user_prefs['user_type']}, {user_prefs['consistency']} rater
Average Rating: {user_prefs['avg_rating']:.2f} (std: {user_prefs['rating_std']:.2f})
Preferred Categories: {', '.join(user_prefs['top_categories'][:3])}
"""
            preference_signals = self.extract_preference_signals(recent_reviews)
        
        # Format review history
        review_summary = []
        for review in recent_reviews:
            item_id = review.get('item_id', '')
            rating = review.get('stars', 'N/A')
            text = review.get('text', '')[:80]
            review_summary.append(f"{item_id}: {rating}★ - {text}")
        
        review_text = "\n".join(review_summary)
        review_text = self.truncate_text(review_text, 3500)
        
        # Get candidate items
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
        
        # Construct task description
        liked_text = "\n".join(preference_signals['liked']) if preference_signals['liked'] else "N/A"
        disliked_text = "\n".join(preference_signals['disliked']) if preference_signals['disliked'] else "N/A"
        
        task_description = f"""You are ranking 20 items for a user based on their preferences.

USER PROFILE:
{user_profile}
{profile_summary}
{memory_context}

PAST REVIEWS:
{review_text}

USER LIKES:
{liked_text}

USER DISLIKES:
{disliked_text}

CANDIDATES (ID: Name | Rating | Categories):
{candidates_text}

INSTRUCTIONS:
1. Match items to user's preferred categories and attributes
2. Consider item quality but prioritize user preference match
3. Use past experiences to inform ranking

OUTPUT: Provide ONLY a valid Python list of all 20 item IDs in ranked order.
Format: ['ID1', 'ID2', 'ID3', ..., 'ID20']

Your ranked list:"""
        
        # Use reasoning module
        result = self.reasoning(task_description)
        
        # Parse result
        try:
            matches = list(re.finditer(r'\[[\s\S]*?\]', result))
            
            if matches:
                for match in reversed(matches):
                    try:
                        result_str = match.group().replace('\n', '').replace('  ', ' ')
                        result_list = eval(result_str)
                        
                        if isinstance(result_list, list):
                            valid_items = [str(item_id).strip() for item_id in result_list 
                                         if str(item_id).strip() in candidate_ids]
                            
                            if valid_items:
                                missing_items = [item_id for item_id in candidate_ids 
                                               if item_id not in valid_items]
                                final_ranking = valid_items + missing_items
                                
                                logger.info(f"[{self.variant}] Successfully parsed. First 5: {final_ranking[:5]}")
                                return final_ranking[:20]
                    except:
                        continue
        except Exception as e:
            logger.error(f"[{self.variant}] Parsing error: {e}")
        
        logger.warning(f"[{self.variant}] Failed to parse, returning original candidate list")
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
    
    class AgentWithVariant(AdvancedRecommendationAgentAblation):
        def __init__(self, llm: LLMBase):
            super().__init__(llm=llm, variant=ABLATION_VARIANT)
    
    simulator.set_agent(AgentWithVariant)
    simulator.set_llm(llm)
    
    outputs = simulator.run_simulation(number_of_tasks=400, enable_threading=False, max_workers=1)
    evaluation_results = simulator.evaluate()
    
    output_file = f'./results/evaluation/evaluation_results_track2_{task_set}_agent2_{ABLATION_VARIANT}.json'
    with open(output_file, 'w') as f:
        combined_results = {
            "time": datetime.now().isoformat(),
            **evaluation_results
        }
        json.dump(combined_results, f, indent=4)
    
    print(f"Evaluation results: {evaluation_results}")

