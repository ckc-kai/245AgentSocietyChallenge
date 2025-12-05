from websocietysimulator.agent import SimulationAgent
from websocietysimulator.agent.modules import MemoryBase
from langchain.docstore.document import Document
from websocietysimulator.agent.modules.reasoning_modules import ReasoningBase
from websocietysimulator import Simulator
from websocietysimulator.llm.llm import GeminiLLM
import numpy as np
from collections import Counter
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

class ScopedMemory(MemoryBase):
    '''
    Layer 1:Retrieve relavent top 3 information based on this particular user
    '''
    def __init__(self, llm):
        super().__init__(memory_type='scoped', llm=llm)

    def retrieve_memory(self, query: str, user_id: str, k: int = 3):
        if self.scenario_memory._collection.count() == 0:
            return "No episodic memories."
        
        # Fetch more to filter by user_id
        results = self.scenario_memory.similarity_search_with_score(query, k=k*4)
        
        relevant = []
        for res, score in results:
            if res.metadata.get('user_id') == user_id:
                relevant.append(res.metadata['task_trajectory'])
        
        if not relevant:
            return "No relevant past experiences found for this user."
            
        return '\n'.join([f"[Episodic {i+1}]: {m}" for i, m in enumerate(relevant[:k])])

    def retrieve_semantic(self, reviews, category):
        """Clustered memory for specific category behavior."""
        if not reviews: return "No semantic memory."
        
        cat_reviews = [r for r in reviews if category.lower() in str(r.get('categories', '')).lower()]
        
        if not cat_reviews:
            return "User has no history in this specific category."
            
        # Simple clustering: Best, Worst, Average
        sorted_r = sorted(cat_reviews, key=lambda x: x['stars'])
        cluster = []
        if sorted_r: cluster.append(f"[Best {category}]: {sorted_r[-1]['stars']}* - {sorted_r[-1]['text'][:100]}...")
        if len(sorted_r) > 1: cluster.append(f"[Worst {category}]: {sorted_r[0]['stars']}* - {sorted_r[0]['text'][:100]}...")
        
        avg_r = np.mean([r['stars'] for r in cat_reviews])
        return f"User avg in '{category}': {avg_r:.2f}\n" + "\n".join(cluster)

    def add_memory(self, text, user_id):
        doc = Document(page_content=text, metadata={"task_trajectory": text, "user_id": user_id})
        self.scenario_memory.add_documents([doc])


class ModelingAgent4(SimulationAgent):
    CURRENT_CONFIG = {}  # class-level config storage

    def __init__(self, llm):
        super().__init__(llm=llm)

        cfg = ModelingAgent4.CURRENT_CONFIG

        self.disable_episodic = cfg.get("disable_episodic", False)
        self.disable_semantic = cfg.get("disable_semantic", False)
        self.disable_reflection = cfg.get("disable_reflection", False)
        self.reduced_prompt = cfg.get("reduced_prompt", False)

        self.memory = ScopedMemory(llm=self.llm)

    def _layer1_profile_modeling(self, reviews):
        """Layer 1: Profile Modeling (Stats, Style, Intent)"""
        if not reviews:
            return {"avg": 3.0, "std": 0.4, "style": "Neutral", "intent": "Unknown"}
            
        ratings = [r['stars'] for r in reviews]
        lengths = [len(r['text'].split()) for r in reviews]
        
        avg = np.mean(ratings)
        std = np.std(ratings)
        
        # Style heuristics
        avg_len = np.mean(lengths)
        exclamations = sum(r['text'].count('!') for r in reviews) / len(reviews)
        caps_ratio = sum(sum(1 for c in r['text'] if c.isupper())/len(r['text']) for r in reviews) / len(reviews)
        
        style_desc = f"Length ~{int(avg_len)} words. "
        if exclamations > 2.0: style_desc += "Excitable (many !). "
        if caps_ratio > 0.4: style_desc += "Intense (caps). "
        if avg_len < 20: style_desc += "Terse/Short. "
        elif avg_len > 120: style_desc += "Detailed/Verbose. "
        
        return {
            "avg": avg,
            "std": std,
            "style": style_desc,
            "dist": dict(Counter(ratings))
        }

    def _layer3_4_5_generate(self, profile, episodic, semantic, item, feedback=None, n=3):
        """
        Merged Layers:
        - Planning & Reasoning
        - Generation
        - Reflection
        """
        
        feedback_section = ""
        if feedback:
            feedback_section = f"\nPREVIOUS FAILED ATTEMPT FEEDBACK: {feedback}\n"

        # --- MERGED PROMPT: Includes implicit reflection instructions ---
        prompt = f"""You are a User Simulator. 

1. USER PROFILE
- Target Rating: Based on user avg ({profile['avg']:.2f}) and category history.
- Tone/Style: {profile['style']}
- Intent: React to {item.get('name')} ({item.get('category')}).

2. MEMORY CONTEXT
Episodic:
{episodic}
Semantic:
{semantic}
{feedback_section}

3. ITEM DETAILS
{json.dumps(item, indent=2)}

4. TASK & SELF-REFLECTION
Generate a review. Before outputting, internally verify:
- Does the rating match the user's historical distribution?
- Is the text style (length, caps, punctuation) strictly aligned with '{profile['style']}'?
- Is the content specific to the item features?

If the review is generic, REWRITE it to be specific.

Output Format: A single valid JSON object with 'stars' (float) and 'review' (string)."""

        messages = [{"role": "user", "content": prompt}]
        
        time.sleep(5) 
        
        try:
            # Layer 4: Multi-path Reasoning (Sampling)
            responses = self.llm(
                messages=messages,
                temperature=0.8, # High temp for diversity
                max_tokens=400,
                n=n
            )
        except Exception as e:
            logger.warning(f"API Error (likely rate limit): {e}. Sleeping 60s...")
            time.sleep(60)
            return ["Error"]

        if isinstance(responses, str): responses = [responses]
        return responses

    def _layer6_consistency(self, responses, profile):
        """Layer 6: Consistency & Calibration
            - Validates JSON
            - Checks limits
            - Returns best match
        """
        valid_results = []
        
        for r in responses:
            try:
                # Robust JSON parsing
                match = re.search(r'\{.*\}', r, re.DOTALL)
                if match:
                    data = json.loads(match.group(0))
                else:
                    # Fallback regex
                    s = re.search(r'stars"?\s*:\s*([\d\.]+)', r)
                    t = re.search(r'review"?\s*:\s*"(.*)"', r, re.DOTALL)
                    if s and t:
                        data = {"stars": float(s.group(1)), "review": t.group(1)}
                    else:
                        continue
                
                # Calibration check
                stars = float(data.get('stars', 3.0))
                stars = min(max(stars, 1.0), 5.0)
                data['stars'] = stars
                
                # Score based on profile alignment
                dist = abs(stars - profile['avg'])
                penalty = dist if profile['std'] > 1.0 else dist * 2
                
                valid_results.append((penalty, data))
            except:
                continue
        
        if not valid_results:
            return None
            
        # Return best (lowest penalty)
        valid_results.sort(key=lambda x: x[0])
        return valid_results[0][1]

    # --- Layer 7 REMOVED (Merged into Layer 3/4/5 prompt) ---

    def workflow(self):
        try:
            user_id = self.task['user_id']
            item_id = self.task['item_id']
            
            # Data
            user_reviews = self.interaction_tool.get_reviews(user_id=user_id)
            item_details = self.interaction_tool.get_item(item_id=item_id)
            
            # Populate Memory with Rate Limiting
            for r in user_reviews:
                self.memory.add_memory(f"{r['stars']}*: {r['text']}", user_id)
                # Sleep briefly here because embeddings might also count toward quota
                time.sleep(2) 
            
            # Layer 1 Profile Building
            profile = self._layer1_profile_modeling(user_reviews)
            
            # Layer 2
            query = f"{item_details.get('name')} {item_details.get('category')}"
            episodic = self.memory.retrieve_memory(query, user_id)
            semantic = self.memory.retrieve_semantic(user_reviews, item_details.get('category', ''))
            
            # Loop
            max_retries = 2
            best_result = {"stars": 3.0, "review": "Error in generation."}
            
            # We removed current_feedback because we aren't calling an external reflector anymore
            # If the JSON parsing fails, we just retry generation
            
            for attempt in range(max_retries + 1):
                # Layers 3, 4, 5 (Now includes Reflection)
                responses = self._layer3_4_5_generate(
                    profile, episodic, semantic, item_details, feedback=None
                )
                
                # Layer 6 (Validation)
                candidate = self._layer6_consistency(responses, profile)
                
                if candidate:
                    best_result = candidate
                    # If we got a valid JSON that matches the profile stats (Layer 6), we trust it.
                    break
                else:
                    logger.info(f"Attempt {attempt} failed JSON validation or consistency.")
                    # Retry logic continues
            
            # Log
            with open(f'./results/generation_detail/super_strong_analyzer_v1.0.txt', 'a') as f:
                f.write(f'\n--- Task {user_id} ---\n')
                f.write(f'Profile: {profile}\n')
                f.write(f'Result: {json.dumps(best_result)}\n')
                
            return best_result

        except Exception as e:
            logger.error(f"Workflow Error: {e}")
            import traceback
            traceback.print_exc()
            return {"stars": 3.0, "review": "System Error."}


ABLATION_SETTINGS = {
    "full_model": {},
    "no_episodic": {"disable_episodic": True},
    "no_semantic": {"disable_semantic": True},
    "no_reflection": {"disable_reflection": True},
    "reduced_prompt": {"reduced_prompt": True}
}


if __name__ == "__main__":

    task_set = "amazon"
    llm = GeminiLLM(
        api_key=os.getenv("GEMINI_API_KEY"),
        model="models/gemini-2.0-flash",
        embedding_model="models/text-embedding-004"
    )
    simulator = Simulator(
        data_dir="./data/processed",
        device="auto",
        cache=True
    )

    simulator.set_task_and_groundtruth(
        f"./example/track1/{task_set}/tasks",
        f"./example/track1/{task_set}/groundtruth"
    )
    for name, cfg in ABLATION_SETTINGS.items():

        print(f"\n==== RUNNING {name} ====")

        ModelingAgent4.CURRENT_CONFIG = cfg 

        simulator.set_agent(ModelingAgent4)
        simulator.set_llm(llm)

        outputs = simulator.run_simulation(
            number_of_tasks=50,
            enable_threading=False,
            max_workers=1
        )

        results = simulator.evaluate()
        save_path = f"./results/evaluation/result_{name}.json"

        with open(save_path, "w") as f:
            json.dump({
                "time": datetime.now().isoformat(),
                "config": cfg,
                "results": results
            }, f, indent=4)

        print(f"DONE {name} — saved at {save_path}")