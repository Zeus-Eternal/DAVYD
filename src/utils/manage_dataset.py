# src/utils/manage_dataset.py

import shutil
import json
import logging
import pandas as pd

from typing import List, Optional
from pathlib import Path
from uuid import uuid4
from datetime import datetime, timezone

# Configure root logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatasetManager:
    """
    Enhanced dataset manager with:
      - temp/archive/merged directories
      - metadata tracking (datasets_meta.json)
      - validation on load/save/merge
      - automatic cleanup of old temp files
    """

    def __init__(
        self,
        temp_dir: str,
        archive_dir: str,
        merged_dir: str = "data_bin/merged_datasets"
    ):
        self.temp_dir    = Path(temp_dir)
        self.archive_dir = Path(archive_dir)
        self.merged_dir  = Path(merged_dir)

        self._ensure_directories_exist()
        self._init_metadata()

    def _ensure_directories_exist(self) -> None:
        """Create all required directories or abort."""
        for d in (self.temp_dir, self.archive_dir, self.merged_dir):
            try:
                d.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                logger.critical(f"Cannot create directory '{d}': {e}")
                raise

    def _init_metadata(self) -> None:
        """Initialize JSON metadata file beside temp_dir."""
        self.metadata_path = self.temp_dir.parent / "datasets_meta.json"
        if not self.metadata_path.exists():
            try:
                self.metadata_path.write_text(json.dumps({}, indent=2))
            except Exception as e:
                logger.warning(f"Could not initialize metadata file: {e}")

    def _update_metadata(self, filename: str, action: str, **extras) -> None:
        """
        Record last action, timestamp, and any extras for a given filename.
        Stored in datasets_meta.json.
        """
        try:
            meta = json.loads(self.metadata_path.read_text())
        except Exception:
            meta = {}

        entry = meta.get(filename, {})
        entry.update({
            "last_action": action,
            "timestamp"  : datetime.now(timezone.utc).isoformat(),
            **extras
        })
        meta[filename] = entry

        try:
            self.metadata_path.write_text(json.dumps(meta, indent=2))
        except Exception as e:
            logger.warning(f"Failed to write metadata for '{filename}': {e}")

    def _validate_file(self, path: Path) -> bool:
        """Basic file sanity checks."""
        if not path.exists():
            logger.error(f"File not found: {path}")
            return False
        if not path.is_file():
            logger.error(f"Not a file: {path}")
            return False
        if path.stat().st_size == 0:
            logger.error(f"Empty file: {path}")
            return False
        return True

    def list_datasets(self, which: str = "temp") -> List[str]:
        """
        List non‐empty .csv/.json files in one of: temp, archive, merged.
        """
        dirs = {
            "temp"   : self.temp_dir,
            "archive": self.archive_dir,
            "merged" : self.merged_dir
        }
        if which not in dirs:
            raise ValueError(f"Unknown directory type '{which}'")
        result = []
        for p in dirs[which].iterdir():
            if p.is_file() and p.suffix in (".csv", ".json") and p.stat().st_size > 0:
                result.append(p.name)
        return sorted(result)

    def load_dataset(self, filename: str, which: str = "temp") -> Optional[pd.DataFrame]:
        """
        Load a CSV or JSON dataset; return None on failure.
        """
        dirs = {
            "temp"   : self.temp_dir,
            "archive": self.archive_dir,
            "merged" : self.merged_dir
        }
        if which not in dirs:
            raise ValueError(f"Unknown directory type '{which}'")

        path = dirs[which] / filename
        if not self._validate_file(path):
            return None

        try:
            if path.suffix == ".csv":
                df = pd.read_csv(path)
            elif path.suffix == ".json":
                df = pd.read_json(path)
            else:
                logger.error(f"Unsupported extension '{path.suffix}' for {path}")
                return None

            self._update_metadata(filename, "load", directory=which)
            return df

        except Exception as e:
            logger.error(f"Error loading '{path}': {e}")
            return None

    def save_dataset(
        self,
        df: pd.DataFrame,
        filename: str,
        which: str = "temp",
        overwrite: bool = False
    ) -> bool:
        """
        Save DataFrame to CSV or JSON. Returns True on success.
        """
        if df.empty:
            logger.error("Cannot save empty DataFrame")
            return False

        dirs = {
            "temp"   : self.temp_dir,
            "archive": self.archive_dir,
            "merged" : self.merged_dir
        }
        if which not in dirs:
            raise ValueError(f"Unknown directory type '{which}'")

        path = dirs[which] / filename
        if path.exists() and not overwrite:
            logger.error(f"File exists and overwrite=False: {path}")
            return False

        try:
            if path.suffix == ".csv":
                df.to_csv(path, index=False)
            elif path.suffix == ".json":
                df.to_json(path, orient="records", indent=2)
            else:
                logger.error(f"Unsupported extension '{path.suffix}' for {path}")
                return False

            self._update_metadata(filename, "save", directory=which)
            return True

        except Exception as e:
            logger.error(f"Error saving '{path}': {e}")
            return False

    def archive_dataset(self, filename: str) -> bool:
        """Move a file from temp → archive."""
        src = self.temp_dir    / filename
        dst = self.archive_dir / filename

        if not self._validate_file(src):
            return False

        try:
            shutil.move(str(src), str(dst))
            self._update_metadata(filename, "archive")
            return True
        except Exception as e:
            logger.error(f"Archive failed for '{filename}': {e}")
            return False

    def restore_dataset(self, filename: str) -> bool:
        """Move a file from archive → temp."""
        src = self.archive_dir / filename
        dst = self.temp_dir    / filename

        if not self._validate_file(src):
            return False

        try:
            shutil.move(str(src), str(dst))
            self._update_metadata(filename, "restore")
            return True
        except Exception as e:
            logger.error(f"Restore failed for '{filename}': {e}")
            return False

    def delete_dataset(self, filename: str) -> bool:
        """
        Delete a dataset from whichever directory it lives in.
        Returns True if any file was removed.
        """
        removed = False
        for directory in (self.temp_dir, self.archive_dir, self.merged_dir):
            path = directory / filename
            if path.exists():
                try:
                    path.unlink()
                    removed = True
                    self._update_metadata(filename, "delete", directory=str(directory))
                except Exception as e:
                    logger.error(f"Deletion failed for '{path}': {e}")

        if not removed:
            logger.error(f"No such file to delete: '{filename}'")
        return removed

    def merge_datasets(
        self,
        filenames: List[str],
        output_name: str,
        validate: bool = True
    ) -> bool:
        """
        Merge multiple temp/archive/merged files into one in merged_dir.
        Returns True on success.
        """
        if len(filenames) < 2:
            logger.error("Need at least two files to merge")
            return False

        dfs = []
        for fn in filenames:
            # Find the first directory containing it
            for directory in (self.archive_dir, self.merged_dir, self.temp_dir):
                candidate = directory / fn
                if candidate.exists():
                    if not self._validate_file(candidate):
                        return False
                    try:
                        if candidate.suffix == ".csv":
                            dfs.append(pd.read_csv(candidate))
                        elif candidate.suffix == ".json":
                            dfs.append(pd.read_json(candidate))
                        else:
                            logger.error(f"Unsupported extension '{candidate.suffix}' for {candidate}")
                            return False
                    except Exception as e:
                        logger.error(f"Failed to load '{candidate}': {e}")
                        return False
                    break
            else:
                logger.error(f"File not found in any directory: '{fn}'")
                return False

        try:
            merged = pd.concat(dfs, ignore_index=True)
            out_path = self.merged_dir / output_name

            if out_path.suffix == ".csv":
                merged.to_csv(out_path, index=False)
            elif out_path.suffix == ".json":
                merged.to_json(out_path, orient="records", indent=2)
            else:
                logger.error(f"Unsupported merge output extension '{out_path.suffix}'")
                return False

            self._update_metadata(output_name, "merge", sources=filenames)
            return True

        except Exception as e:
            logger.error(f"Merge failed for '{output_name}': {e}")
            return False

    def get_temp_filename(self, prefix: str = "dataset") -> str:
        """Generate a unique temp‐directory filename (CSV)."""
        return f"{prefix}_{uuid4().hex}.csv"

    def cleanup_temp(self, max_age_days: int = 7) -> int:
        """
        Delete any temp files older than max_age_days.
        Returns the count of deleted files.
        """
        cutoff = datetime.now(timezone.utc).timestamp() - (max_age_days * 86400)
        deleted = 0

        for f in self.temp_dir.iterdir():
            if f.is_file() and f.stat().st_mtime < cutoff:
                try:
                    f.unlink()
                    deleted += 1
                    logger.info(f"Cleaned up old file: {f.name}")
                except Exception as e:
                    logger.warning(f"Failed to delete '{f}': {e}")

        return deleted
