"""
JSON serialization for RF System Tool scenes.

Saves and loads the complete scene state:
  - All blocks (with their parameters and canvas positions)
  - All connections between blocks
  - Annotation items (free-text labels)
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Any, Optional

from rf_tool.blocks.components import block_from_dict, BLOCK_REGISTRY
from rf_tool.models.rf_block import RFBlock


def save_scene(
    blocks: List[RFBlock],
    connections: List[Dict],
    annotations: Optional[List[Dict]] = None,
    filepath: str = "scene.json",
    metadata: Optional[Dict] = None,
) -> None:
    """
    Serialize the full scene to a JSON file.

    Parameters
    ----------
    blocks : list of RFBlock
    connections : list of dict
        Each entry: {
            "src_block_id": str,
            "src_port": str,
            "dst_block_id": str,
            "dst_port": str,
        }
    annotations : list of dict, optional
        Each entry: {"text": str, "x": float, "y": float,
                     "font": str, "color": str, "font_size": int}
    filepath : str
    metadata : dict, optional
        Extra key/value pairs saved alongside the scene.
    """
    scene_data: Dict[str, Any] = {
        "version": "1",
        "metadata": metadata or {},
        "blocks": [b.to_dict() for b in blocks],
        "connections": connections,
        "annotations": annotations or [],
    }
    with open(filepath, "w", encoding="utf-8") as fh:
        json.dump(scene_data, fh, indent=2)


def load_scene(filepath: str) -> Dict[str, Any]:
    """
    Load a scene from a JSON file.

    Returns
    -------
    dict with keys:
        "blocks"      : list of RFBlock
        "connections" : list of dict
        "annotations" : list of dict
        "metadata"    : dict
    """
    with open(filepath, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    blocks = [block_from_dict(b) for b in data.get("blocks", [])]
    return {
        "blocks": blocks,
        "connections": data.get("connections", []),
        "annotations": data.get("annotations", []),
        "metadata": data.get("metadata", {}),
    }
