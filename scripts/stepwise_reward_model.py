

import os
import time
from typing import List, Tuple

import openai

sk
openai.api_key = ""

def evaluate_stepwise(prompt: str, steps: List[str]) -> List[float]:
    """
    Given a prompt and stepwise reasoning, returns stepwise reward scores from GPT-4.

    Args:
        prompt (str): The original task/question.
        steps (List[str]): Step-by-step reasoning.

    Returns:
        List[float]: Reward scores ∈ [0, 1] per step.
    """

    step_text = "\n".join([f"Step {i+1}: {step}" for i, step in enumerate(steps)])
    
    full_prompt = f"""
You are an expert tutor evaluating a student's step-by-step reasoning.

Task: {prompt}

Student's Reasoning:
{step_text}

Rate the correctness of each step on a scale from 0.0 (completely incorrect) to 1.0 (fully correct).
Only output a list of floats separated by commas.
    """

    for attempt in range(3):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert grader."},
                    {"role": "user", "content": full_prompt.strip()}
                ],
                temperature=0,
            )
            output = response.choices[0].message.content.strip()
            scores = [float(s) for s in output.split(",")]
            return scores
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(2)

    return [0.5 for _ in steps]

