import json
import logging
import os
from datetime import datetime
from collections import Counter

import numpy as np
from dotenv import load_dotenv

from websocietysimulator.agent import SimulationAgent
from websocietysimulator.agent.modules import MemoryBase
from websocietysimulator.agent.modules.reasoning_modules import ReasoningBase

from websocietysimulator.llm.local_llm import LocalLLM
from websocietysimulator import Simulator
from langchain.docstore.document import Document




logger = logging.getLogger("websocietysimulator")
logger.propagate = False


# ==========================================
# EfficientMemory（完全和原版一样）
# ==========================================
class EfficientMemory(MemoryBase):

    def __init__(self, llm):
        super().__init__(memory_type="efficient", llm=llm)

    def retriveMemory(self, query_scenario: str, k: int = 3):
        if self.scenario_memory._collection.count() == 0:
            return ""

        similarity_results = (
            self.scenario_memory.similarity_search_with_score(
                query_scenario, k=k
            )
        )

        memories = []
        for i, (result, score) in enumerate(similarity_results, 1):
            trajectory = result.metadata["task_trajectory"]
            memories.append(f"[Memory {i}]: {trajectory}")

        return "\n\n".join(memories)

    def addMemory(self, current_situation):
        memory_doc = Document(
            page_content=current_situation,
            metadata={
                "task_name": current_situation[:100],
                "task_trajectory": current_situation,
            },
        )
        self.scenario_memory.add_documents([memory_doc])


# ==========================================
# ConsistentReasoning（完全和原版一样）
# ==========================================
class ConsistentReasoning(ReasoningBase):

    def __init__(self, profile_type_prompt, memory, llm, n_sample=3):
        super().__init__(profile_type_prompt, memory, llm)
        self.n_sample = n_sample

    def __call__(self, task_description: str):
        prompt = f"""Analyze this step-by-step and provide your final answer:
{task_description}
Think through this carefully, considering the user's profile, past behavior, and the business details.
Then provide your final answer in the exact format requested.
"""
        messages = [{"role": "user", "content": prompt}]
        responses = self.llm(
            messages=messages,
            temperature=0.20,
            max_tokens=400,
            n=self.n_sample,
        )

        if isinstance(responses, str):
            logger.info("Single response received, skipping voting.")
            return responses

        parsed_responses = []
        star_ratings = []

        for response in responses:
            parsed_responses.append(response)
            try:
                stars_line = [
                    line for line in response.split("\n")
                    if "stars:" in line.lower()
                ][0]
                stars = float(stars_line.split(":", 1)[1].strip())
                star_ratings.append(stars)
            except:
                star_ratings.append(3.0)

        median_idx = np.argsort(star_ratings)[len(star_ratings) // 2]
        return parsed_responses[median_idx]


# AnalyzeAgent（原版结构 + 加入 ablation 最小改动）

class AnalyzeAgent(SimulationAgent):

    def __init__(self, llm, ablation):
        super().__init__(llm=llm)
        self.ablation = ablation 
        # memory（off → None）
        if self.ablation["use_memory"]:
            self.memory = EfficientMemory(llm=self.llm)
        else:
            self.memory = None

        # reasoning n_sample
        n_sample = 3 if self.ablation["use_consistency"] else 1
        self.reasoning = ConsistentReasoning(
            profile_type_prompt="",
            memory=self.memory,
            llm=self.llm,
            n_sample=n_sample,
        )

    # ==========================================
    # 以下所有函数完全保留原版（未改）
    # ==========================================

    def summarize_user_profile(self, user_reviews):
        if not user_reviews:
            return "No historical data available."

        ratings = [r["stars"] for r in user_reviews]
        review_lengths = [len(r["text"].split()) for r in user_reviews]

        avg_rating = np.mean(ratings)
        rating_std = np.std(ratings)
        rating_mode = Counter(ratings).most_common(1)[0][0]
        avg_length = int(np.mean(review_lengths))

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

        return f"""User Behavior Profile Summary:
- Average Rating: {avg_rating:.1f} / 5.0 (Most common: {rating_mode})
- Rating Consistency: {consistency} (std: {rating_std:.2f})
- User Type: {user_type}
- Typical Review Length: {avg_length} words
- Total Reviews: {len(user_reviews)}"""

    def learn_review_style(self, user_reviews):
        if not user_reviews:
            return "No historical data available."

        sorted_reviews = sorted(user_reviews, key=lambda x: x["stars"])

        learning_samples = []
        if len(sorted_reviews) >= 5:
            learning_samples.append(
                f"Top Positive Review Sample {sorted_reviews[-1]['stars']} stars: "
                f"{sorted_reviews[-1]['text'][:100]}"
            )
            learning_samples.append(
                f"Top Negative Review Sample {sorted_reviews[0]['stars']} stars: "
                f"{sorted_reviews[0]['text'][:100]}"
            )
            neutral = [r for r in sorted_reviews if 2.0 < r["stars"] < 4.0]
            if len(neutral) > 1:
                s = np.random.choice(neutral)
                learning_samples.append(
                    f"Neutral Review Sample {s['stars']} stars: {s['text'][:100]}"
                )
        else:
            learning_samples = [
                f"Review {r['stars']} stars: {r['text'][:100]}"
                for r in sorted_reviews
            ]

        return "\n\n".join(learning_samples)

    def _parse_out(self, llm_response):
        import re
        try:
            clean = llm_response.replace("**","").replace("*","").replace("`","")
            stars_match = re.search(r"stars\s*[:=]\s*([0-9.]+)", clean, re.I)
            stars = float(stars_match.group(1)) if stars_match else 3.0

            review_match = re.search(r"review\s*[:=]\s*(.*)", clean, re.I)
            review = (
                review_match.group(1).strip() if review_match else "No review."
            )[:512]

            return {"stars": min(max(stars,1.0),5.0), "review": review}

        except Exception as e:
            logger.error(f"Parsing error: {e}")
            return {"stars": 3.0, "review": "Parsing error."}

    # =====================================================
    # workflow（只加入 ablation 条件，其他完全保持原版）
    # =====================================================
    def workflow(self):
        try:
            user_id = self.task["user_id"]
            item_id = self.task["item_id"]

            user_reviews = self.interaction_tool.get_reviews(user_id=user_id)
            user_profile = self.interaction_tool.get_user(user_id=user_id)
            item_details = self.interaction_tool.get_item(item_id=item_id)

            # memory add（关闭则不写入）
            if self.ablation["use_memory"] and self.memory:
                for review in user_reviews:
                    entry = (
                        f"Reviewed business in category {review.get('categories','?')}: "
                        f"Gave {review['stars']} stars. Review: {review['text'][:100]}..."
                    )
                    self.memory.addMemory(entry)

            # profile ablation
            behavior_summary = (
                self.summarize_user_profile(user_reviews)
                if self.ablation["use_profile"]
                else "User profiling disabled."
            )

            style_samples = self.learn_review_style(user_reviews)

            # memory retrieval
            if self.ablation["use_memory"] and self.memory:
                relevant_memory = self.memory.retriveMemory(
                    f"The user is now considering {item_details.get('name','N/A')}."
                )
            else:
                relevant_memory = "Memory disabled."

            avg_rating = (
                np.mean([r["stars"] for r in user_reviews]) if user_reviews else 3.0
            )

            PROMPT_VERSION_1 = """
You are simulating a real user writing a review.

USER PROFILE:
{user_profile}

USER BEHAVIOR ANALYSIS:
{behavior_summary}

USER'S WRITING STYLE:
{style_samples}

BUSINESS TO REVIEW:
{item_details}

RELEVANT PAST EXPERIENCES:
{relevant_memory}

TASK:
Based on this user's profile, behavior patterns, and writing style, generate a review for this business.

CRITICAL REQUIREMENTS:
1. Match the user's typical rating pattern (consider their average: {avg_rating:.1f})
2. Write in a similar style and length as their past reviews
3. Be authentic - don't be overly positive or negative unless that matches the user's pattern
4. Consider the business's existing rating and category
5. Write 2-4 sentences that sound natural

FORMAT (EXACTLY):
stars: [rating between 1.0 and 5.0]
review: [your review text]
"""

            PROMPT_VERSION_2 = """


You are a real Yelp user expressing your feelings about a business.
Do NOT write analytically. 
Do NOT try to reason step-by-step.
Write naturally, emotionally, and personally — as if you were telling a friend what happened.

Here is your profile:
{user_profile}

Your general behavior patterns:
{behavior_summary}

Your writing style examples:
{style_samples}

Business you are reviewing:
{item_details}

Your relevant past memories:
{relevant_memory}

TASK:
Write a short, emotional review that reflects how this user would FEEL about this business.
Focus on:
- what immediately stood out (good or bad)
- how the experience made the user feel
- small details that matter to this user
- any comparison to previous similar experiences

Do NOT try to be formal or structured.
Be casual, expressive, and realistic.

END YOUR RESPONSE WITH EXACTLY:
stars: [a rating between 1.0 and 5.0]
review: [a natural 2–4 sentence emotional review]

"""

            if self.ablation["prompt_version"] == 1:
                task_description = PROMPT_VERSION_1.format(
                    user_profile=json.dumps(user_profile, indent=2),
                    behavior_summary=behavior_summary,
                    style_samples=style_samples,
                    item_details=json.dumps(item_details, indent=2),
                    relevant_memory=relevant_memory,
                    avg_rating=avg_rating,
    )
            else:
                task_description = PROMPT_VERSION_2.format(
                    user_profile=json.dumps(user_profile, indent=2),
                    behavior_summary=behavior_summary,
                    style_samples=style_samples,
                    item_details=json.dumps(item_details, indent=2),
                    relevant_memory=relevant_memory,
                    avg_rating=avg_rating,
    )

            task_description = f"""
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
{relevant_memory}

TASK:
Based on this user's profile, behavior patterns, and writing style, generate a review for this business.

CRITICAL REQUIREMENTS:
1. Match the user's typical rating pattern (consider their average: {avg_rating:.1f})
2. Write in a similar style and length as their past reviews
3. Be authentic - don't be overly positive or negative unless that matches the user's pattern
4. Consider the business's existing rating and category
5. Write 2-4 sentences that sound natural

FORMAT (EXACTLY):
stars: [rating between 1.0 and 5.0]
review: [your review text]
"""

            llm_response = self.reasoning(task_description=task_description)

            return self._parse_out(llm_response)

        except Exception as e:
            logger.error(f"Workflow error: {e}")
            return {"stars": 3.0, "review": "Workflow error."}
# =======================
#         MAIN
# =======================
if __name__ == "__main__":
    # claude_api = os.getenv("CLAUDE_API_KEY")
    # openai_api = os.getenv("OPENAI_API_KEY")
    # if not claude_api or not openai_api:
    #     raise ValueError("CLAUDE_API_KEY or OPENAI_API_KEY not found in environment variables.")
    # else:
    #     logger.info("API keys successfully loaded from environment variables.")

    print("Starting simulation with localllm.")
   

    task_set = "yelp"

    # ablation_config = {
    #     "use_memory": True,
    #     "use_profile": True,
    #     "use_consistency": True,
    #     "prompt_version": 1,
    # }

    # 示例：关闭 memory
    # ablation_config = {
    #     "use_memory": False,
    #     "use_profile": True,
    #     "use_consistency": True,
    #     "prompt_version": 1,
    # }

    # # 示例：关闭 self-consistency
    # ablation_config = {
    #     "use_memory": True,
    #     "use_profile": True,
    #     "use_consistency": False,
    #     "prompt_version": 1,

    # }

            # 示例：关闭 use-profile
    # ablation_config = {
    #     "use_memory": True,
    #     "use_profile": False,
    #     "use_consistency": True,
    #     "prompt_version": 1,

    # }
    #             示例：换prompt
    ablation_config = {
        "use_memory": True,
        "use_profile": True,
        "use_consistency": True,
        "prompt_version": 2

    }


    # ===================
    #     Simulator
    # ===================
    simulator = Simulator(
        data_dir="data/processed",
        device="auto",
        cache=True,
    )

    simulator.set_task_and_groundtruth(
        task_dir=f"./example/track1/{task_set}/tasks",
        groundtruth_dir=f"./example/track1/{task_set}/groundtruth",
    )

    # ===================
    #       LLM
    # ===================
    
    llm = LocalLLM(
        model="llama3.1:8b",
        embedding_model_name="nomic-embed-text",
    )

    simulator.set_agent(AnalyzeAgent)
    simulator.set_llm(llm)

    # ===============================
    #     关键：将 ablation 注入 agent
    # ===============================
    class AblationAnalyzeAgent(AnalyzeAgent):
        def __init__(self, llm):
            super().__init__(llm, ablation=ablation_config)
    simulator.set_agent(AblationAnalyzeAgent)



    # ===================
    #     Run tasks
    # ===================
    outputs = simulator.run_simulation(
        number_of_tasks=50,
        enable_threading=False,
        max_workers=1,
    )

    # ===================
    #     Evaluation
    # ===================
    evaluation_results = simulator.evaluate()

    os.makedirs("./results/evaluation", exist_ok=True)

    # 文件名包含 ablation 名称方便区分
    ablation_name = (
        f"{'mem' if ablation_config['use_memory'] else 'noMem'}_"
        f"{'profile' if ablation_config['use_profile'] else 'noProfile'}_"
        f"{'cons' if ablation_config['use_consistency'] else 'noCons'}"
        f"{'promp' if ablation_config['prompt_version'] else 'promp1'}"
    )

    out_path = (
        f"./results/evaluation/"
        f"evaluation_track1_{task_set}_model_local_{ablation_name}.json"
    )

    meta = {
        "time": datetime.now().isoformat(),
        "task_set": task_set,
        "ablation": ablation_config,
        "results": evaluation_results,
    }

    with open(out_path, "a") as f:
        f.write(json.dumps(meta, indent=4) + "\n")

    print(f"\nEvaluation saved to: {out_path}")
