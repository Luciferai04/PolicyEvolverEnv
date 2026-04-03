import asyncio
import os
import sys

# Add current directory to path so we can import everything correctly
sys.path.insert(0, os.getcwd())

from server.environment import PolicyEvolverEnvironment
from models import Action

async def verify_repetition():
    env = PolicyEvolverEnvironment()
    # Reset to start fresh
    obs1 = env.reset(task_id="task_easy")
    
    same_action = {
        "action_type": "propose_clarification",
        "ambiguous_term": "appropriate",
        "suggested_definition": "Behavior is defined as appropriate when it specifically follows the community guidelines, meaning it does not include excessive slurs and meets the 5% threshold for verified user reports.",
        "justification": "The current policy leads to inconsistent and subjective moderation because it is unclear and varies between interpreters.",
        "think": ""
    }
    
    # First step
    print("\n--- Step 1 ---")
    res1 = env.step(same_action)
    print(f"Step 1 Reward: {res1.reward}")
    
    # Second step with identical action
    print("\n--- Step 2 (Repeat) ---")
    res2 = env.step(same_action)
    print(f"Step 2 Reward: {res2.reward}")
    
    # Assert
    assert res2.reward < res1.reward, f"Repeated action should score lower! res2={res2.reward}, res1={res1.reward}"
    print("\n✅ Anti-repetition test passed! Reward was penalized as expected.")

if __name__ == "__main__":
    asyncio.run(verify_repetition())
