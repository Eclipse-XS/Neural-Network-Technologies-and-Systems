"""Fixed, portable experiment configuration (no model imports)."""
from dataclasses import asdict, dataclass
from pathlib import Path
import math

LAB_ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = 'llava-hf/llava-interleave-qwen-0.5b-hf'
MODEL_REVISION = '1090956dd1c79bc93ae98dcf395590369435ec91'
INPUT_IMAGE_PATH = LAB_ROOT / 'assets/input/primary_image.png'
QUESTIONS_PATH = LAB_ROOT / 'questions.json'
OUTPUT_ROOT = LAB_ROOT / 'outputs'
BASE_MAX_NEW_TOKENS = 64
BASE_DO_SAMPLE = False
TEMPERATURE_VALUES = (0.2, 0.7, 1.0)
MAX_TOKEN_VALUES = (24, 64, 128)
SAMPLING_SEED = 42
GROUNDING_INSTRUCTION = (
    'Answer using only information visible in the image. '
    'If the answer cannot be determined from the image, explicitly say so. '
    'Answer concisely and do not invent details.'
)


@dataclass(frozen=True)
class GenerationConfig:
    do_sample: bool = BASE_DO_SAMPLE
    max_new_tokens: int = BASE_MAX_NEW_TOKENS
    temperature: float | None = None
    top_p: float | None = None
    seed: int | None = None

    def __post_init__(self):
        if type(self.max_new_tokens) is not int or self.max_new_tokens <= 0:
            raise ValueError('max_new_tokens must be a positive integer')
        if self.do_sample:
            if self.temperature is None or not math.isfinite(self.temperature) or self.temperature <= 0:
                raise ValueError('Sampling requires a positive finite temperature')
            if self.top_p is None or not 0 < self.top_p <= 1:
                raise ValueError('Sampling requires top_p in (0, 1]')
            if type(self.seed) is not int or self.seed < 0:
                raise ValueError('Sampling requires a nonnegative seed')
        elif any(v is not None for v in (self.temperature, self.top_p, self.seed)):
            raise ValueError('Greedy decoding must omit sampling parameters')

    def kwargs(self):
        return {k: v for k, v in asdict(self).items() if v is not None and k != 'seed'}
