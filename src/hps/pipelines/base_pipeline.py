from typing import Dict, Any


class PipelineRunOutput:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class BasePipeline:
    def __init__(self, **config):
        self.config = config

    def run(self, **kwargs) -> PipelineRunOutput:
        raise NotImplementedError

