"""Version 1 contracts. Serialized identifiers are always strings."""

from dataclasses import dataclass
from typing import Literal, TypedDict

SCHEMA_VERSION = "1.0"


class MaskRLE(TypedDict):
    size: list[int]
    counts: str


def exact_id(value: int | str) -> str:
    """Never accept floats (or booleans) as identifiers, even if integral."""
    if type(value) not in (int, str) or str(value) == "":
        raise ValueError("Identifier must be a nonempty string or exact integer")
    return str(value)


@dataclass(frozen=True)
class FilamentInstance:
    sequence_id: str
    frame_index: int
    instance_id: str
    bbox_xywh: tuple[float, float, float, float]
    mask_rle: MaskRLE | None = None
    score: float | None = None
    track_id: str | None = None
    center_xy: tuple[float, float] | None = None
    length_px: float | None = None

    def __post_init__(self) -> None:
        for field in ("sequence_id", "instance_id", "track_id"):
            value = getattr(self, field)
            if value is not None:
                object.__setattr__(self, field, exact_id(value))
        if self.frame_index < 0:
            raise ValueError("frame_index must be nonnegative")


@dataclass(frozen=True)
class SequenceMetadata:
    sequence_id: str
    source: Literal["synthetic", "real"]
    fps: float | None = None
    pixel_size_um: float | None = None
    calibration_provenance: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "sequence_id", exact_id(self.sequence_id))
        if self.source not in ("synthetic", "real"):
            raise ValueError("Unknown source")
        for value in (self.fps, self.pixel_size_um):
            if value is not None and not (0 < value < float("inf")):
                raise ValueError("Calibration must be finite and positive")
        if (self.fps is not None or self.pixel_size_um is not None) and not (
            self.calibration_provenance
        ):
            raise ValueError("Calibration requires provenance")
