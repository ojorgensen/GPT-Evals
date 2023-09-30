import logging
from typing import List, Optional, Callable
import os
import time
import pandas as pd

import hydra
from omegaconf import DictConfig


from src.evals.config import EvaluationConfig

from src.models.openai_model import (
    OpenAIChatModels,
    generate_chat_completion,
)


def logging_setup(config: EvaluationConfig):
    # Set up a logger
    logger = logging.getLogger('my_logger')

    # Set the logging level
    logger.setLevel(config.logging_level)

    # Create a file handler that writes log messages to a file
    log_file = config.output_location + 'logfile.log'
    file_handler = logging.FileHandler(log_file)

    # Create a formatter and set it for the handler
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(file_handler)



def evaluation(
    config: EvaluationConfig,
):
    """
    Given a dataset and a model, apply a prompt template to each item in the dataset.
    Feed each of these to the model, and store the results. Further, apply an evaluation function
    to each of the results, and store the results of that. 
    """
    # config.output_location = config.output_location + str(time.time()) + "/"
    # os.mkdir(config.output_location)

    

    results = pd.DataFrame(columns=["prompt", "model_response", "evaluation_response"])
    with open(config.dataset_location, "r") as dataset:

        for line in dataset:
            prompt = config.prompt_template(line)
            model_response = generate_chat_completion(
                prompt_turns=prompt,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                model=config.model,
            )
            evaluation_response = config.evaluation_function(model_response)
            results.append(
                {
                    "prompt": prompt,
                    "model_response": model_response,
                    "evaluation_response": evaluation_response,
                },
                ignore_index=True,
            )

    results.to_csv(config.output_location + "results.csv")
    return results


@hydra.main(config_path="conf", config_name="main")
def main(cfg: DictConfig):
    # Convert the DictConfig to a config object
    config: EvaluationConfig = EvaluationConfig.from_dict(cfg)
    # Setup the logger
    logging_setup(config)
    # Run the evaluation
    evaluation(config)


if __name__ == "__main__":
    config = EvaluationConfig.from_dict(
        {
            "model": OpenAIChatModels.DA_VINCI,
            "dataset_location": "data/convai2/valid_self_original.txt",
            "prompt_template": lambda x: x,
            "evaluation_function": lambda x: x,
            "output_location": "results/",
            "seed": 0,
            "temperature": 0.0,
            "max_tokens": 256,
            "logging_level": logging.INFO,
        }
    )