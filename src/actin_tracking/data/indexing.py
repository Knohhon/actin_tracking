"""Read audited annotations without rescanning source JSON."""

import json
import sqlite3
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from actin_tracking.data.schemas import exact_id


class DatasetIndex:
    def __init__(self, path: Path) -> None:
        self.connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
        try:
            summary = self.connection.execute(
                "SELECT value FROM metadata WHERE key='summary'"
            ).fetchone()
            if summary is None:
                raise ValueError("Incomplete audit index: summary is absent")
        except (ValueError, sqlite3.DatabaseError):
            self.connection.close()
            raise

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "DatasetIndex":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def get_frame(self, image_id: str | int) -> dict[str, Any]:
        row = self.connection.execute(
            "SELECT payload FROM images WHERE id=?", (exact_id(image_id),)
        ).fetchone()
        if row is None:
            raise KeyError(image_id)
        return {
            "image": json.loads(row[0]),
            "annotations": [
                json.loads(payload)
                for (payload,) in self.connection.execute(
                    "SELECT payload FROM annotations WHERE image_id=? ORDER BY row_id",
                    (exact_id(image_id),),
                )
            ],
        }

    def get_sequence(self, sequence_id: str | int) -> Iterator[dict[str, Any]]:
        for (identifier,) in self.connection.execute(
            "SELECT id FROM images WHERE sequence_id=? ORDER BY frame_index,id",
            (exact_id(sequence_id),),
        ):
            yield self.get_frame(identifier)

    def iter_unique_frames(self) -> Iterator[dict[str, Any]]:
        """Only unconflicted, readable physical frames, one record per PNG."""
        for (identifier,) in self.connection.execute(
            "SELECT canonical_image_id FROM physical_frames "
            "WHERE conflicted=0 AND valid=1 AND sha256 IS NOT NULL ORDER BY file_name"
        ):
            yield self.get_frame(identifier)
