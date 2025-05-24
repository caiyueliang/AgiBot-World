from lightning.pytorch.cli import LightningCLI

from genie.dataset import LightningPlatformer2D
from genie.dataset_openx import LightningOpenX, LightningOpenX_MultiView

from genie.model import Genie, DINO_LAM

cli = LightningCLI(
    Genie,
    LightningOpenX,
    seed_everything_default=42,
)
