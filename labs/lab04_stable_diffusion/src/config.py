from dataclasses import asdict, dataclass
from pathlib import Path
import math

LAB_ROOT = Path(__file__).resolve().parents[1]
BASE_PROMPT = ('a futuristic city street at sunset, neon signs, wet pavement reflections, '
               'detailed architecture, cinematic lighting, pedestrians in the distance')
STYLE_SCENE = 'a small mountain cabin beside a calm lake, surrounded by pine trees, soft morning light'
NEGATIVE_PROMPT = 'blurry, low quality, watermark, text, distorted architecture, deformed objects, artifacts'
STYLES = {
    'watercolor': 'watercolor painting, soft washes, paper texture',
    'isometric': 'isometric illustration, clean geometric forms',
    'photorealistic': 'photorealistic landscape photography, natural realistic textures',
}


@dataclass(frozen=True)
class ModelConfig:
    requested_model_id: str = 'runwayml/stable-diffusion-v1-5'
    model_id: str = 'stable-diffusion-v1-5/stable-diffusion-v1-5'
    revision: str = '451f4fe16113bff5a5d2269ed5ad43b0592e9a14'
    device_strategy: str = 'model_cpu_offload'


@dataclass(frozen=True)
class GenerationSpec:
    experiment: str = 'baseline'
    case_id: str = 'baseline'
    prompt: str = BASE_PROMPT
    negative_prompt: str | None = None
    seed: int = 42
    num_inference_steps: int = 30
    guidance_scale: float = 7.5
    width: int = 512
    height: int = 512
    style: str | None = None

    def __post_init__(self):
        if not self.prompt.strip():
            raise ValueError('Prompt must not be empty')
        if type(self.seed) is not int or not 0 <= self.seed < 2**63:
            raise ValueError('Seed must be a nonnegative integer below 2**63')
        if type(self.num_inference_steps) is not int or self.num_inference_steps <= 0:
            raise ValueError('Steps must be a positive integer')
        if not math.isfinite(self.guidance_scale) or self.guidance_scale <= 1:
            raise ValueError('This CFG experiment requires finite guidance > 1')
        if any(type(v) is not int or v < 64 or v % 8 for v in (self.width, self.height)):
            raise ValueError('Dimensions must be positive multiples of eight, at least 64')

    def controls(self):
        return {k: v for k, v in asdict(self).items() if k not in ('experiment', 'case_id', 'style')}
