import pytest
from deepeval.models import DeepEvalBaseLLM

from deepteam.guardrails.guards import ToxicityGuard
from deepteam.guardrails.guards.schema import SafetyLevelSchema


class StructuredModel(DeepEvalBaseLLM):
    """Returns a parsed SafetyLevelSchema, like a model with schema support."""

    def __init__(self, safety_level: str):
        self.safety_level = safety_level

    def load_model(self):
        return self

    def generate(self, prompt: str, schema=None):
        return SafetyLevelSchema(safety_level=self.safety_level, reason="r")

    async def a_generate(self, prompt: str, schema=None):
        return self.generate(prompt, schema)

    def get_model_name(self):
        return "structured-model"


class JsonModel(StructuredModel):
    """Rejects `schema` and returns raw JSON text, exercising the fallback."""

    def generate(self, prompt: str, schema=None):
        if schema is not None:
            raise TypeError("schema not supported")
        return f'{{"safety_level": "{self.safety_level}", "reason": "r"}}'


SCORES = [("safe", 1.0), ("borderline", 0.0), ("unsafe", 0.0)]


@pytest.mark.parametrize("model_cls", [StructuredModel, JsonModel])
@pytest.mark.parametrize("safety_level,expected", SCORES)
class TestGuardScore:
    def test_score_sync(self, model_cls, safety_level, expected):
        guard = ToxicityGuard(model=model_cls(safety_level))
        guard.guard_input("hello")
        assert guard.safety_level == safety_level
        assert guard.score == expected

    @pytest.mark.asyncio
    async def test_score_async(self, model_cls, safety_level, expected):
        guard = ToxicityGuard(model=model_cls(safety_level))
        await guard.a_guard_input("hello")
        assert guard.safety_level == safety_level
        assert guard.score == expected
