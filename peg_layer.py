"""Native-resolution peg grid overlay, shown while the editor places a hook (used by editor.py and main_opengl.py).

The world is drawn on a surface scaled down by scaleDownFactor, so a 1 px dot there shows up as a 2x2 block on
screen. The pegs are instead drawn on this layer at native screen (window) resolution: a one-channel mask, 255 on a
dot pixel and 0 elsewhere, `size` px wide square dots centred on the pixel each peg falls in. It's built once and
cached; `version` changes whenever it's rebuilt, so main_opengl.py uploads it to a texture only then and draws it
each frame as a blended quad (shaders/pegs.frag, colour and opacity from editor.PEG_COL / PEG_ALPHA).
"""
import numpy as np
import pygame


class PegLayer:
    def __init__(self, screen_size, size=1):
        """screen_size: native screen (window) px; size: dot width in screen px (odd, so a dot centres on its pixel)."""
        self.w, self.h = screen_size
        self.size = size
        self.mask = None  # (h, w) uint8, row 0 = the top row
        self.key = None  # what the mask was built for
        self.version = 0

    def update(self, key, columns_rows):
        """Rebuild the mask if `key` changed; columns_rows() gives the dots' pixel columns and rows (one dot at
        every column x row crossing)."""
        if key == self.key:
            return
        cols, rows = columns_rows()
        half = self.size // 2
        mask = np.zeros((self.h, self.w), np.uint8)
        offsets = np.arange(-half, half + 1)
        cols = np.unique((np.asarray(cols)[:, None] + offsets).ravel())
        rows = np.unique((np.asarray(rows)[:, None] + offsets).ravel())
        cols, rows = cols[(cols >= 0) & (cols < self.w)], rows[(rows >= 0) & (rows < self.h)]
        mask[np.ix_(rows, cols)] = 255
        self.mask, self.key = mask, key
        self.version += 1

    def to_surface(self, col, alpha):
        """The overlay as a per-pixel-alpha pygame surface (col at `alpha` on the dots) for CPU presents / tests."""
        surface = pygame.Surface((self.w, self.h), pygame.SRCALPHA)
        surface.fill((*col, 0))
        a = pygame.surfarray.pixels_alpha(surface)  # indexed [x, y]
        a[...] = (self.mask.T > 0) * alpha
        del a
        return surface
