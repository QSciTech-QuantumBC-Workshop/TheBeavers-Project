from typing import Dict, Any, Optional
import json
import os


class PipelineRunOutput:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def to_json(self, filename: Optional[str] = None):
        data = self.__dict__
        for key, value in data.items():
            if hasattr(value, "__json__"):
                data[key] = value.__json__()
            elif hasattr(value, "to_json"):
                data[key] = value.to_json()
        if filename is not None:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, "w") as f:
                json.dump(data, f, indent=4, default=vars)
        return data


class BasePipeline:
    def __init__(self, **config):
        self.config = config

    def run(self, **kwargs) -> PipelineRunOutput:
        raise NotImplementedError

    def __json__(self):
        return self.config

