import json
from abc import ABC, abstractmethod
from pathlib import Path

from utils.aggregator import save_result


class PipelineStage(ABC):
    @property
    @abstractmethod
    def output_path(self) -> Path:
        pass

    @abstractmethod
    def run(self) -> None:
        pass

    def save(self, result: dict) -> None:
        save_result(result, self.output_path)

    def load_output(self) -> dict:
        with open(self.output_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {int(k) if k.isdigit() else k: v for k, v in data.items()}
