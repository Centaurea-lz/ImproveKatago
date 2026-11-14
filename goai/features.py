"""Board -> neural-net input tensor (AlphaZero-style 17 planes)."""
from __future__ import annotations

import numpy as np

from .board import BLACK, GoBoard

NUM_PLANES = 17


def encode(board: GoBoard) -> np.ndarray:
    """Return a float32 tensor of shape (17, size, size).

    8 pairs of planes = own/opponent stones for the last 8 positions
    (re-oriented to the current player), plus one color-to-move plane.
    """
    size = board.size
    planes = np.zeros((NUM_PLANES, size, size), dtype=np.float32)
    cur = board.current
    opp = 3 - cur
    hist = board.recent
    for i in range(min(8, len(hist))):
        snap = hist[-1 - i].reshape(size, size)  # i moves ago
        planes[2 * i] = (snap == cur)
        planes[2 * i + 1] = (snap == opp)
    if cur == BLACK:
        planes[16] = 1.0
    return planes
