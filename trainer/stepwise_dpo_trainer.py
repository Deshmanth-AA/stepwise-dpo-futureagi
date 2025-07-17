

from typing import Any, Dict

import torch
from transformers import Trainer


class StepwiseDPOTrainer(Trainer):
    """
    Custom Trainer for Stepwise DPO using step-level rewards.
    This extends Hugging Face's Trainer for fine-tuning with LLM-generated stepwise feedback.
    """

    def __init__(self, reward_model_fn, *args, **kwargs):
        """
        Args:
            reward_model_fn: Callable that takes (prompt, steps) and returns stepwise rewards.
        """
        super().__init__(*args, **kwargs)
        self.reward_model_fn = reward_model_fn

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Compute the loss using stepwise reward aggregation.
        """
        # Forward pass
        outputs = model(**inputs)
        logits = outputs.logits
        labels = inputs.get("labels")

        # Step 1: Decode inputs into reasoning steps
        input_ids = inputs["input_ids"]
        decoded_inputs = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)

        # Step 2: Extract prompt and steps
        prompts, step_lists = [], []
        for full_text in decoded_inputs:
            # Assumes format: "Prompt:\n...\nSteps:\n1. ...\n2. ...\n"
            if "Steps:" not in full_text:
                return torch.tensor(0.0, requires_grad=True, device=logits.device)
            parts = full_text.split("Steps:")
            prompt = parts[0].strip()
            steps = parts[1].strip().split("\n")
            steps = [s.strip().lstrip("1234567890. ") for s in steps if s.strip()]
            parts = full_text.split("Steps:")
            prompt = parts[0].strip()
            steps = parts[1].strip().split("\n")
            steps = [s.strip().lstrip("1234567890. ") for s in steps if s.strip()]
            prompts.append(prompt)
            step_lists.append(steps)

        # Step 3: Get stepwise rewards from LLM
        batch_rewards = [
            self.reward_model_fn(prompt, steps)
            for prompt, steps in zip(prompts, step_lists)
        ]

        # Step 4: Aggregate reward to a scalar (e.g., average)
        scalar_rewards = torch.tensor(
            [sum(rewards) / len(rewards) for rewards in batch_rewards],
            device=logits.device
        )

        # Step 5: Compute loss (Negative Log Likelihood scaled by reward)
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        per_token_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        per_token_loss = per_token_loss.view(shift_labels.size())

        input_mask = (shift_labels != -100).float()
        loss_per_sample = (per_token_loss * input_mask).sum(dim=1) / input_mask.sum(dim=1)

        weighted_loss = loss_per_sample * scalar_rewards
        loss = weighted_loss.mean()

        return (loss, outputs) if return_outputs else loss
