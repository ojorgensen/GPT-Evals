from logging import getLogger
from typing import List, Optional, Callable

from src.evals.config import EvaluationConfig

from src.models.openai_model import (
    OpenAIChatModels,
    OpenAITextModels,
    generate_chat_completion,
    generate_text_completion,
)

def evaluation(
    config: EvaluationConfig,
):
    """
    Given a dataset and a model, apply a prompt template to each item in the dataset.
    Feed each of these to the model, and store the results. Further, apply an evaluation function
    to each of the results, and store the results of that. 
    """
    
    with open(config.dataset_location, "r") as dataset:
        for line in dataset:
            prompt = config.prompt_template(line)
            model_response = generate_chat_completion(
                prompt_turns=prompt,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                model=config.model,
            )

        




if __name__ == "__main__":
    logger = getLogger(__name__)
    logger.info("Hello, World!")