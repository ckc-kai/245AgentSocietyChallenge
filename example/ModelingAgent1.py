import json
import logging
from websocietysimulator import Simulator
from websocietysimulator.agent import SimulationAgent
from websocietysimulator.agent.modules.memory_modules import MemoryGenerative
from websocietysimulator.agent.modules.reasoning_modules import ReasoningCOT
from websocietysimulator.llm import ClaudeLLM
from websocietysimulator.llm.llm import LLMBase
from datetime import datetime
import os
from dotenv import load_dotenv
load_dotenv(dotenv_path="./secrets.env")
logger = logging.getLogger("websocietysimulator")
logger.propagate = False
class MySimulationAgent(SimulationAgent):
    """
    A custom simulation agent for the Web Society Simulator.
    This agent uses a generative memory module and a chain-of-thought reasoning module.
    Generative memory allows the top 3 most relavent memories to be retrieved for reasoning.
    Chain-of-thought reasoning enables step-by-step problem solving.
    """
    def __init__(self, llm: LLMBase):
        super().__init__(llm=llm)
        self.memory = MemoryGenerative(llm=self.llm)
        self.reasoning = ReasoningCOT(profile_type_prompt='', memory=None, llm=self.llm)

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
        """
        Simulate user behavior
        Returns:
            tuple: (star (float), useful (float), funny (float), cool (float), review_text (str))
        """
        try:
            # Fetch user and item data
            user_id=self.task['user_id']
            item_id=self.task['item_id']

            # add the user past to memory
            user_reviews = self.interaction_tool.get_reviews(user_id=user_id)
            for review in user_reviews:
                memory_entry = f"User {review['user_id']} previously reviewed a business: (Stars: {review['stars']}, Review: {review['text']})"
                self.memory.addMemory(memory_entry)

            # gather current task info
            user_profile = self.interaction_tool.get_user(user_id=user_id)
            item_details = self.interaction_tool.get_item(item_id=item_id)

            # Task Promot 
            query_scenario = (
                f"The user is now considering a business named "
                f"{item_details.get('name', 'N/A')} which is a"
                f"{item_details.get('category', 'N/A')}."
                )
            relevant_memory = self.memory.retriveMemory(query_scenario)
            logger.info(f"Task {user_id}: Found relevant memory: {relevant_memory[:50]}...")

            # COT Reasoning
            task_description = f"""
            You are a Yelp reviewer.

            Your Profile:
            {json.dumps(user_profile, indent=2)}

            Business to Review:
            {json.dumps(item_details, indent=2)}

            Your Most Relevant Past Experience (from your memory):
            {relevant_memory}

            INSTRUCTIONS:
            Write a new review for the business.

            First, think step-by-step (Chain of Thought) about your decision.
            1.  Analyze your user profile. What do you like or dislike?
            2.  Analyze the business. What are its key features?
            3.  Compare the business to your relevant past experience. Is it better or worse?
            4.  Based on this, decide on a final star rating (1.0-5.0).
            5.  Write a 1-4 sentence review that justifies your rating and reflects your profile. The length and writting style should match your profile, but DO NOT write more than 4 sentences.

            CRITICAL: After all your thinking, you MUST end your response with EXACTLY these two lines (nothing after them):
            stars: [Your star rating as a number between 1.0 and 5.0]
            review: [Your review text in 2-4 sentences]

            Make sure these two lines are the LAST lines of your response with nothing following them.
            """
            llm_response = self.reasoning(task_description=task_description)
            logger.info(f"Task {user_id}: LLM response: {llm_response}")

            result = self._parse_out(llm_response)
            
            with open(f'./results/generation_detail/reason_v1.txt', 'a') as f:
                f.write(f'\n {datetime.now()}')
                f.write(f'\n LLM Response: {json.dumps(result, indent=4)}')
            
            return result
        except Exception as e:
            logger.error(f"Error during workflow execution: {e}")  
            return {
                "stars": 0.0,
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
    task_set = "yelp" # "goodreads" or "yelp" or "amazon"
    simulator = Simulator(data_dir="/Users/ckc/Desktop/UCLA/2025fall/245/AgentSocietyChallenge/data/processed", device="auto", cache=True)
    simulator.set_task_and_groundtruth(task_dir=f"./example/track1/{task_set}/tasks", groundtruth_dir=f"./example/track1/{task_set}/groundtruth")

    # Set the agent and LLM
    llm = ClaudeLLM(api_key=claude_api, model="claude-sonnet-4-20250514",openai_api_key=openai_api)
    simulator.set_agent(MySimulationAgent)
    simulator.set_llm(llm)

    # Run the simulation
    # If you don't set the number of tasks, the simulator will run all tasks.
    outputs = simulator.run_simulation(number_of_tasks=10, enable_threading=True, max_workers=10)
    
    # Evaluate the agent
    evaluation_results = simulator.evaluate()       
    with open(f'./results/evaluation/evaluation_results_track1_{task_set}.json_model1', 'a') as f:
        time = {"time": datetime.now().isoformat()}
        f.write(json.dump(time, f, indent=4)+"\n")
        f.write(json.dump(evaluation_results, f, indent=4) + '\n')

    # Get evaluation history
    evaluation_history = simulator.get_evaluation_history()