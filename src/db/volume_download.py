"""Download files from Unity Catalog Volumes via Databricks SDK.

Databricks Apps don't have the /Volumes/ FUSE mount available in notebooks.
This module uses the Databricks SDK Files API to download Volume contents
to a local cache directory for use at runtime.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def download_volume_dir(volume_path: str, local_dir: str) -> Path:
    """Download all files from a UC Volume directory to a local directory.

    Args:
        volume_path: Path like ``/Volumes/catalog/schema/volume/folder``
        local_dir: Local directory to cache files into

    Returns:
        Path to the local directory with downloaded files.
    """
    local = Path(local_dir)

    # If volume path is already accessible (notebook environment), just return it
    vol = Path(volume_path)
    if vol.is_dir() and any(vol.iterdir()):
        logger.info("Volume path %s is directly accessible", volume_path)
        return vol

    # If local cache already has files, reuse it
    if local.is_dir() and any(local.iterdir()):
        logger.info("Using cached files from %s", local)
        return local

    # Download via Databricks SDK
    try:
        from databricks.sdk import WorkspaceClient

        w = WorkspaceClient()
        local.mkdir(parents=True, exist_ok=True)

        count = 0
        for f in w.files.list_directory_contents(volume_path):
            if f.is_directory:
                continue
            file_path = f.path
            dest = local / f.name
            logger.info("Downloading %s → %s", file_path, dest)
            with w.files.download(file_path).contents as src:
                dest.write_bytes(src.read())
            count += 1

        logger.info("Downloaded %d files from %s to %s", count, volume_path, local)
        return local

    except Exception:
        logger.exception("Failed to download from Volume %s", volume_path)
        raise
