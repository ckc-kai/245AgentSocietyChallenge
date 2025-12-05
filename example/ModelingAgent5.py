from websocietysimulator.agent import SimulationAgent
from websocietysimulator.agent.modules import MemoryBase
from langchain.docstore.document import Document
from websocietysimulator.agent.modules.reasoning_modules import ReasoningBase
from websocietysimulator import Simulator
from websocietysimulator.llm.llm import GeminiLLM
import numpy as np
from collections import Counter, defaultdict
import json
import logging
import os
import re
from datetime import datetime
from dotenv import load_dotenv
import time

load_dotenv(dotenv_path="./secrets.env")
logger = logging.getLogger("websocietysimulator")
logger.propagate = False

class EnhancedMemory(MemoryBase):
    '''Enhanced Memory with Episodic, Semantic, and Comparative capabilities'''
    def __init__(self, llm):
        super().__init__(memory_type='enhanced', llm=llm)
        self.comparative_memory = {}  # user_id -> {category: {best_item, worst_item, reference_points}}

    def retrieve_memory(self, query: str, user_id: str, k: int = 3):
        if self.scenario_memory._collection.count() == 0:
            return "No episodic memories."
        
        results = self.scenario_memory.similarity_search_with_score(query, k=k*4)
        relevant = []
        for res, score in results:
            if res.metadata.get('user_id') == user_id:
                relevant.append(res.metadata['task_trajectory'])
        
        if not relevant:
            return "No relevant past experiences found for this user."
            
        return '\n'.join([f"[Memory {i+1}]: {m}" for i, m in enumerate(relevant[:k])])

    def build_comparative_memory(self, user_id, reviews):
        """Build reference points for what user considers good/bad"""
        if not reviews:
            return
        
        category_items = defaultdict(list)
        for r in reviews:
            cat = r.get('categories', 'general')
            category_items[cat].append(r)
        
        self.comparative_memory[user_id] = {}
        for cat, items in category_items.items():
            if not items:
                continue
            sorted_items = sorted(items, key=lambda x: x['stars'])
            self.comparative_memory[user_id][cat] = {
                'worst': sorted_items[0] if sorted_items else None,
                'best': sorted_items[-1] if sorted_items else None,
                'avg_stars': np.mean([i['stars'] for i in items])
            }

    def get_comparative_context(self, user_id, category):
        """Get user's reference points for this category"""
        if user_id not in self.comparative_memory:
            return "No comparative reference available."
        
        comp = self.comparative_memory[user_id].get(category, {})
        if not comp:
            return f"User has no history in {category} for comparison."
        
        context = f"User's {category} reference points:\n"
        if comp.get('best'):
            context += f"- Best experience: {comp['best']['stars']}★ - {comp['best']['text'][:80]}...\n"
        if comp.get('worst'):
            context += f"- Worst experience: {comp['worst']['stars']}★ - {comp['worst']['text'][:80]}...\n"
        context += f"- Category average: {comp.get('avg_stars', 3.0):.1f}★"
        return context

    def add_memory(self, text, user_id):
        doc = Document(page_content=text, metadata={"task_trajectory": text, "user_id": user_id})
        self.scenario_memory.add_documents([doc])


class PersonaConstructor:
    """Builds rich psychological user profile"""
    
    @staticmethod
    def build_persona(reviews):
        if not reviews or len(reviews) < 2:
            return PersonaConstructor._default_persona()
        
        ratings = [r['stars'] for r in reviews]
        texts = [r['text'] for r in reviews]
        
        # 1. Rating distribution
        avg_rating = np.mean(ratings)
        std_rating = np.std(ratings)
        rating_dist = Counter(ratings)
        
        if avg_rating >= 4.0:
            rating_philosophy = "Optimistic supporter - tends to focus on positives"
        elif avg_rating <= 2.5:
            rating_philosophy = "Critical evaluator - high standards, focuses on flaws"
        else:
            rating_philosophy = "Balanced reviewer - evaluates fairly"
        
        # 2. Writing Style Analysis
        avg_length = np.mean([len(t.split()) for t in texts])
        exclamation_rate = np.mean([t.count('!') for t in texts])
        question_rate = np.mean([t.count('?') for t in texts])
        caps_ratio = np.mean([sum(1 for c in t if c.isupper()) / max(len(t), 1) for t in texts])
        
        style_traits = []
        if avg_length < 20:
            style_traits.append("Concise/terse writer")
            verbosity = "brief"
        elif avg_length > 100:
            style_traits.append("Detailed/verbose writer")
            verbosity = "detailed"
        else:
            style_traits.append("Moderate length writer")
            verbosity = "moderate"
        
        if exclamation_rate > 1.8:
            style_traits.append("Enthusiastic tone (many !)")
            emotion_level = "high"
        else:
            style_traits.append("Normal tone")
            emotion_level = "moderate"
        
        if caps_ratio > 0.15:
            style_traits.append("Uses emphasis (CAPS)")
        
        # 3. Value Sensitivity (Hybrid count/proportion)
        price_mentions = sum(1 for t in texts if any(word in t.lower() for word in ['bucks', 'cents', 'price', 'cost', 'expensive', 'cheap', 'value', 'worth', 'dollar', 'pricey', 'costly','affordable', 'inexpensive', 'budget', 'coupons', 'discount', 'deal', 'sale', 'promo', 'offer', 'underpriced', 'overpriced', 'cashback']))
        quality_mentions = sum(1 for t in texts if any(word in t.lower() for word in ['quality', 'well-made', 'durable', 'sturdy', 'premium', 'high-quality', 'high-end', 'premium', 'premium quality', 'premium quality', 'low quality', 'low-end', 'low quality', 'reliable', 'poor', 'broken', 'fragile', 'usable', 'performance', 'robust']))
        
        review_count = len(texts)
        price_ratio = price_mentions / review_count
        quality_ratio = quality_mentions / review_count
        print(f"Price ratio: {price_ratio}, Quality ratio: {quality_ratio}, length: {avg_length}")
        # Logic: For short reviews, any mention is significant. For long reviews, we need consistent mentions.
        is_price_focused = False
        if avg_length < 30:
            is_price_focused = price_ratio > 0.3 or price_mentions >= 2
        else:
            # For longer reviews, require higher frequency or count
            is_price_focused = price_ratio > 0.1 or price_mentions >= 4

        is_quality_focused = False
        if avg_length < 30:
            is_quality_focused = quality_ratio > 0.3 or quality_mentions >= 2
        else:
            is_quality_focused = quality_ratio > 0.1 or quality_mentions >= 4

        if is_price_focused:
            value_focus = "Price-conscious - frequently mentions cost/value"
        elif is_quality_focused:
            value_focus = "Quality-focused - prioritizes quality over price"
        else:
            value_focus = "Balanced value consideration"
        
        # 4. Common Vocabulary/Phrases
        all_text = " ".join(texts).lower()
        common_words = []
        for word in ['great', 'good', 'excellent', 'perfect', 'disappointed', 'expected', 'recommend', 'waste']:
            if all_text.count(word) >= 2:
                common_words.append(word)
        
        # 5. Comparison Behavior
        has_comparisons = any(word in all_text for word in ['better than', 'worse than', 'compared to', 'similar to', 'like my'])
        
        return {
            'rating_stats': {
                'avg': avg_rating,
                'std': std_rating,
                'distribution': dict(rating_dist),
                'philosophy': rating_philosophy
            },
            'writing_style': {
                'avg_length': int(avg_length),
                'verbosity': verbosity,
                'emotion_level': emotion_level,
                'traits': style_traits,
                'common_words': common_words
            },
            'value_orientation': value_focus,
            'uses_comparisons': has_comparisons,
            'review_count': len(reviews)
        }
    
    @staticmethod
    def _default_persona():
        return {
            'rating_stats': {'avg': 3.5, 'std': 1.0, 'distribution': {}, 'philosophy': 'Unknown'},
            'writing_style': {'avg_length': 50, 'verbosity': 'moderate', 'emotion_level': 'moderate', 'traits': [], 'common_words': []},
            'value_orientation': 'Balanced',
            'uses_comparisons': False,
            'review_count': 0
        }


class UserItemFitReasoner(ReasoningBase):
    """
    COT Reasoning about user-item fit
    Reason about whether THIS user would like THIS item
    """
    
    def __init__(self, llm):
        super().__init__(profile_type_prompt=None, memory=None, llm=llm)

    def __call__(self, persona, item, comparative_context, user_reviews_sample):
        """Chain-of-thought reasoning about user-item match"""
        
        prompt = f"""You are analyzing whether a specific user would like a specific item.

USER PROFILE:
- Rating tendency: {persona['rating_stats']['philosophy']} (avg: {persona['rating_stats']['avg']:.1f}★, std: {persona['rating_stats']['std']:.1f})
- Writing style: {persona['writing_style']['verbosity']}, {persona['writing_style']['emotion_level']} emotion
- Value focus: {persona['value_orientation']}
- Review history: {persona['review_count']} reviews

COMPARATIVE CONTEXT:
{comparative_context}

SAMPLE PAST REVIEWS (for reference):
{user_reviews_sample}

ITEM TO EVALUATE:
{json.dumps(item, indent=2)}

REASONING TASK:
Think step-by-step about whether THIS specific user would like THIS specific item:

1. MATCH ANALYSIS:
   - Does this item match user's quality expectations?
   - Does the price align with user's value sensitivity?
   - How does this compare to similar items user has reviewed?

2. ASPECT-LEVEL PREDICTIONS:
   For each aspect, predict user's likely reaction:
   - Quality/Build: [would user be satisfied? specific observation?]
   - Value for Money: [would user think it's worth it?]
   - Functionality: [meets user's likely needs?]
   - Category-specific: [any special considerations?]

3. RATING PREDICTION:
   - Based on above, what rating range? (e.g., 3.5-4.0)
   - What would push it higher? What would lower it?
   - Considering user's rating philosophy, final prediction?

4. KEY POINTS USER WOULD MENTION:
   - What 2-3 specific things would user praise?
   - What 1-2 things would user criticize or note?

Output your reasoning in this JSON format:
{{
    "match_analysis": "brief analysis",
    "aspect_predictions": {{
        "quality": {{"sentiment": "positive/neutral/negative", "note": "specific observation"}},
        "value": {{"sentiment": "positive/neutral/negative", "note": "specific observation"}},
        "functionality": {{"sentiment": "positive/neutral/negative", "note": "specific observation"}}
    }},
    "predicted_rating": float (1.0-5.0),
    "rating_confidence": "high/medium/low",
    "key_praise_points": ["point1", "point2"],
    "key_criticism_points": ["point1"]
}}"""

        messages = [{"role": "user", "content": prompt}]
        time.sleep(8)
        try:
            response = self.llm(messages=messages, temperature=0.3, max_tokens=650, n=1)
            if isinstance(response, list):
                response = response[0]
            
            # Parse JSON
            match = re.search(r'\{.*\}', response, re.DOTALL)
            if match:
                reasoning = json.loads(match.group(0))
                return reasoning
            else:
                logger.warning("Reasoning failed to parse, using fallback")
                return self._fallback_reasoning(persona, item)
        except Exception as e:
            logger.error(f"Reasoning error: {e}")
            return self._fallback_reasoning(persona, item)
    
    def _fallback_reasoning(self, persona, item):
        """Fallback if LLM reasoning fails"""
        predicted_rating = persona['rating_stats']['avg']
        return {
            'match_analysis': 'Using statistical prediction',
            'aspect_predictions': {
                'quality': {'sentiment': 'neutral', 'note': 'Average expectation'},
                'value': {'sentiment': 'neutral', 'note': 'Standard pricing'},
                'functionality': {'sentiment': 'neutral', 'note': 'Basic functionality'}
            },
            'predicted_rating': predicted_rating,
            'rating_confidence': 'low',
            'key_praise_points': ['Works as expected'],
            'key_criticism_points': []
        }


class MultiAspectGenerator:
    """
    Generates reviews aspect-by-aspect, then synthesizes
    """
    
    @staticmethod
    def generate(llm, persona, item, reasoning, n=3):
        """Generate review based on reasoning"""
        
        # Build style instructions
        style_instructions = f"""
WRITING STYLE GUIDELINES FOLLOW:
- Length: ~{persona['writing_style']['avg_length']} words ({persona['writing_style']['verbosity']})
- Emotion level: {persona['writing_style']['emotion_level']}
- Traits: {', '.join(persona['writing_style']['traits'])}
"""
        if persona['writing_style']['common_words']:
            style_instructions += f"- Try to use words like: {', '.join(persona['writing_style']['common_words'][:3])}\n"
        
        prompt = f"""You are writing a review AS this specific user (not about them).

USER'S RATING PHILOSOPHY: {persona['rating_stats']['philosophy']}
USER'S VALUE FOCUS: {persona['value_orientation']}

{style_instructions}

ITEM: {item.get('name')} - {item.get('category')}
Price: ${item.get('price', 'N/A')}

YOUR REASONING (Internal - use this to guide the review):
- Predicted satisfaction: {reasoning['predicted_rating']:.1f}★
- Quality reaction: {reasoning['aspect_predictions']['quality']['sentiment']} - {reasoning['aspect_predictions']['quality']['note']}
- Value reaction: {reasoning['aspect_predictions']['value']['sentiment']} - {reasoning['aspect_predictions']['value']['note']}
- Things to praise: {', '.join(reasoning['key_praise_points'])}
- Things to criticize: {', '.join(reasoning['key_criticism_points'])}

TASK: Write a review that:
1. Matches the predicted rating ({reasoning['predicted_rating']:.1f}★)
2. Mentions specific item features/details (not generic)
3. Reflects the aspect-level sentiments above
4. Uses the user's natural writing style
5. Sounds like a REAL person, not AI

Output ONLY valid JSON:
{{"stars": float, "review": "text"}}"""

        messages = [{"role": "user", "content": prompt}]
        
        time.sleep(12)
        try:
            responses = llm(messages=messages, temperature=0.5, max_tokens=350, n=n)
            if isinstance(responses, str):
                responses = [responses]
            return responses
        except Exception as e:
            logger.warning(f"Generation error: {e}. Retrying with simplified prompt.")
            time.sleep(60)
            
            # Fallback: Simplified prompt
            simple_prompt = f"""Write a review for {item.get('title', item.get('name', 'item'))}.
Rating: {reasoning['predicted_rating']} stars.
Style: {persona['rating_stats']['philosophy']}.
Output JSON: {{"stars": {reasoning['predicted_rating']}, "review": "your review here"}}"""
            
            try:
                messages = [{"role": "user", "content": simple_prompt}]
                response = llm(messages=messages, temperature=0.7, max_tokens=200, n=1)
                if isinstance(response, str):
                    return [response]
                return response
            except Exception as e2:
                logger.error(f"Fallback generation failed: {e2}")
                return [json.dumps({"stars": reasoning['predicted_rating'], "review": "Error in generation"})]


class AuthenticityValidator:
    """Enhanced validation with multiple checks"""
    
    @staticmethod
    def validate_and_select(responses, persona, reasoning, item):
        """Validate candidates and select best"""
        
        valid_candidates = []
        
        for response in responses:
            try:
                data = None
                clean_response = re.sub(r'```json\s*|\s*```', '', response).strip()
                match = re.search(r'\{.*\}', clean_response, re.DOTALL)
                if match:
                    try:
                        data = json.loads(match.group(0))
                    except:
                        pass
                
                if not data:
                    stars_match = re.search(r'"stars":\s*([\d\.]+)', response)
                    review_match = re.search(r'"review":\s*"([^"]+)"', response)
                    if stars_match and review_match:
                        data = {
                            'stars': float(stars_match.group(1)),
                            'review': review_match.group(1)
                        }
                
                if not data:
                    continue
                
                stars = float(data.get('stars', 3.0))
                stars = max(1.0, min(5.0, stars))
                review_text = data.get('review', '')
                
                if len(review_text) < 2: # Loosened length constraint
                    continue
                
                try:
                    score = AuthenticityValidator._calculate_authenticity_score(
                        stars, review_text, persona, reasoning, item
                    )
                except Exception as e:
                    logger.warning(f"Scoring error: {e}")
                    score = 10.0 # Default score on error
                
                valid_candidates.append({
                    'stars': stars,
                    'review': review_text,
                    'score': score
                })
                
            except Exception as e:
                logger.debug(f"Validation error: {e}")
                continue
        
        if not valid_candidates:
            return None
        
        # Return best scoring candidate
        valid_candidates.sort(key=lambda x: x['score'], reverse=True)
        best = valid_candidates[0]
        return {'stars': best['stars'], 'review': best['review']}
    
    @staticmethod
    def _calculate_authenticity_score(stars, review_text, persona, reasoning, item):
        """Multi-dimensional authenticity scoring with granular differentiation"""
        score = 100.0
        
        # 1. Rating consistency (Continuous penalty)
        predicted = reasoning.get('predicted_rating', 3.0)
        rating_diff = abs(stars - predicted)
        score -= rating_diff * 12.0  # Tuned penalty
        
        # 2. Length consistency (Continuous penalty)
        actual_length = len(review_text.split())
        expected_length = persona['writing_style'].get('avg_length', 50)
        length_ratio = actual_length / max(expected_length, 1)
        
        # Penalize deviation from 1.0 (perfect match)
        # e.g. ratio 1.5 -> diff 0.5 -> -10 pts
        dist_from_one = abs(length_ratio - 1.0)
        score -= dist_from_one * 20.0
        
        # 3. Sentiment-rating alignment
        positive_words = ['great', 'excellent', 'love', 'perfect', 'amazing', 'best', 'awesome', 'fantastic', 'wonderful', 'satisfied', 'happy', 'highly recommend', 'superb', 'impressed']
        negative_words = ['bad', 'terrible', 'worst', 'waste', 'disappointed', 'poor', 'awful', 'disappointing', 'hate', 'useless', 'cheap', 'frustrating', 'broke', 'unacceptable']
        
        review_lower = review_text.lower()
        pos_count = sum(1 for word in positive_words if word in review_lower)
        neg_count = sum(1 for word in negative_words if word in review_lower)
        
        if stars >= 4.0 and neg_count > pos_count:
            score -= 15.0
        elif stars <= 2.0 and pos_count > neg_count:
            score -= 15.0
            
        # 4. Bonuses for differentiation
        # Bonus for using persona's common words
        common_words = persona['writing_style'].get('common_words', [])
        hits = sum(1 for w in common_words if w in review_lower)
        score += min(10.0, hits * 2.0)
        
        # Bonus for specific item details (Specificity)
        item_name = item.get('name', '').lower()
        item_terms = [t for t in item_name.split() if len(t) > 3]
        detail_hits = sum(1 for t in item_terms if t in review_lower)
        score += min(10.0, detail_hits * 2.0)

        print(f"Score: {score:.1f} | Stars: {stars} | LenRatio: {length_ratio:.2f} | Diff: {rating_diff} | Dist: {dist_from_one:.2f} | Hits: {hits} | DetailHits: {detail_hits}")
        return max(0.0, score)


class ModelingAgent5(SimulationAgent):
    """Enhanced agent with persona-driven multi-aspect architecture"""
    
    def __init__(self, llm):
        super().__init__(llm=llm)
        self.memory = EnhancedMemory(llm=self.llm)
        self.reasoner = UserItemFitReasoner(llm=self.llm)
        
    def workflow(self):
        try:
            user_id = self.task['user_id']
            item_id = self.task['item_id']
            
            # Get data
            user_reviews = self.interaction_tool.get_reviews(user_id=user_id)
            item_details = self.interaction_tool.get_item(item_id=item_id)
            
            # Populate memory with rate limiting
            for r in user_reviews[:6]:  
                self.memory.add_memory(f"{r['stars']}*: {r['text']}", user_id)
                time.sleep(12)
            
            # Build comparative memory
            self.memory.build_comparative_memory(user_id, user_reviews)
            
            #  Build Rich Persona 
            logger.info(f"Building persona for user {user_id}")
            persona = PersonaConstructor.build_persona(user_reviews)
            
            # Get Context 
            category = item_details.get('category', 'general')
            comparative_context = self.memory.get_comparative_context(user_id, category)
            
            # Sample reviews for reasoning
            sample_reviews = "\n".join([f"- {r['stars']}★: {r['text'][:100]}..." 
                                       for r in user_reviews[:3]])
            
            # Reason About User-Item Fit ⭐
            logger.info("Reasoning about user-item fit")
            reasoning = self.reasoner(
                persona, item_details, comparative_context, sample_reviews
            )
            
            # Generate Multi-Aspect Reviews
            logger.info("Generating review candidates")
            responses = MultiAspectGenerator.generate(
                self.llm, persona, item_details, reasoning, n=3
            )
            
            # Validate & Select Best
            logger.info("Validating candidates")
            result = AuthenticityValidator.validate_and_select(
                responses, persona, reasoning, item_details
            )
            
            if not result:
                # Fallback
                logger.warning("All candidates failed validation, using reasoning-based fallback")
                item_name = item_details.get('title', item_details.get('name', 'item'))
                result = {
                    'stars': reasoning['predicted_rating'],
                    'review': f"This {item_name} is {ModelingAgent5._get_simple_descriptor(reasoning['predicted_rating'])}."
                }
            
            # Log details
            with open('./results/generation_detail/enhanced_agent_v5.txt', 'a') as f:
                f.write(f'\n{"="*60}\n')
                f.write(f'User: {user_id} | Item: {item_id}\n')
                f.write(f'Persona: {json.dumps(persona, indent=2)}\n')
                f.write(f'Reasoning: {json.dumps(reasoning, indent=2)}\n')
                f.write(f'Result: {json.dumps(result, indent=2)}\n')
            
            return result
            
        except Exception as e:
            logger.error(f"Workflow error: {e}")
            import traceback
            traceback.print_exc()
            return {"stars": 3.0, "review": "System error occurred."}
    
    @staticmethod
    def _get_simple_descriptor(rating):
        if rating >= 4.5: return "excellent"
        elif rating >= 3.5: return "good"
        elif rating >= 2.5: return "okay"
        else: return "disappointing"


if __name__ == "__main__":
    gemini_api = os.getenv("GEMINI_API_KEY")
    if not gemini_api:
        raise ValueError("GEMINI_API_KEY not found in environment")
    
    task_set = "amazon"
    simulator = Simulator(data_dir="./data/processed", device="auto", cache=True)
    simulator.set_task_and_groundtruth(
        task_dir=f"./example/track1/{task_set}/tasks",
        groundtruth_dir=f"./example/track1/{task_set}/groundtruth"
    )
    
    llm = GeminiLLM(api_key=gemini_api, model="gemini-2.0-flash")
    simulator.set_agent(ModelingAgent5)
    simulator.set_llm(llm)
    
    # Run simulation
    outputs = simulator.run_simulation(
        number_of_tasks=10,
        enable_threading=False,
        max_workers=1
    )
    
    evaluation_results = simulator.evaluate()
    
    # Save results
    os.makedirs('./results/evaluation', exist_ok=True)
    with open(f'./results/evaluation/evaluation_results_track1_{task_set}_agent5.json', 'a') as f:
        entry = {
            "timestamp": datetime.now().isoformat(),
            "model": "ModelingAgent5_Enhanced",
            "results": evaluation_results
        }
        f.write(json.dumps(entry, indent=4) + "\n")
    
    print(f"\n{'='*60}")
    print(f"EVALUATION RESULTS - Enhanced Agent v5")
    print(f"{'='*60}")
    print(json.dumps(evaluation_results, indent=2))