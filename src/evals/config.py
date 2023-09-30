from dataclasses import dataclass, fields
from typing import Optional, Union, Callable

from omegaconf import DictConfig, OmegaConf

from src.models.base_model import BaseModel
from src.models.utils import get_model_from_string

@dataclass
class BaseEvalConfig:
    model: BaseModel

    def __post_init__(self):
        if isinstance(self.model, str):
            self.model = get_model_from_string(self.model)

    @classmethod
    def from_dict(cls, params: Union[dict, DictConfig]):
        if isinstance(params, DictConfig):
            params = OmegaConf.to_container(params)

        mandatory_fields = [field.name for field in fields(cls) if field.init]
        return cls(**{k: v for k, v in params.items() if k in mandatory_fields})
    
@dataclass
class EvaluationConfig(BaseEvalConfig):
    dataset_location: str
    prompt_template: Callable
    evaluation_function: Callable
    seed: int = 0
    temperature: float = 0.0
    max_tokens: int = 256
    