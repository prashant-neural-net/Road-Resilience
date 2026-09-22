"""Runtime configuration for the Road Resilience API.

Values are intentionally read from environment variables so the repository can
remain free of large model and generated-output files. See ``.env.example`` for
the available options.
"""

from dataclasses import dataclass
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    """Application paths and inference settings."""

    model_checkpoint: Path
    output_dir: Path
    device: str
    prediction_threshold: float


def get_settings() -> Settings:
    """Build settings from the environment, with repository-friendly defaults."""

    return Settings(
        model_checkpoint=Path(
            os.getenv("ROAD_RESILIENCE_MODEL_PATH", PROJECT_ROOT / "deeplabv3plus_road.pth")
        ).expanduser(),
        output_dir=Path(
            os.getenv("ROAD_RESILIENCE_OUTPUT_DIR", PROJECT_ROOT / "outputs")
        ).expanduser(),
        device=os.getenv("ROAD_RESILIENCE_DEVICE", "auto").lower(),
        prediction_threshold=float(os.getenv("ROAD_RESILIENCE_THRESHOLD", "0.5")),
    )


settings = get_settings()
