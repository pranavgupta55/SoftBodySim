"""Editing existing objects while paused with the spawn editor closed: select, move, rotate, delete, copy / paste and
duplicate. Mixed into editor.Editor (it uses the editor's picking, containment tree, peg grid, messages and outlines).

Objects ("units"): a body (not a wall), a rope, a hook, or a whole chain (its boxes and ropes, by their chain_id).
- Click picks one: a hook, else a rope right under the cursor, else the innermost body, else a rope nearby. Shift+click
  adds / removes it. Dragging on empty space selects every object fully inside the box (Shift: adds to the selection);
  a click on empty space clears it.
- Dragging a selected object moves the selection; Q / E rotate it ROTATE_STEP degrees (counter-clockwise / clockwise
  on screen) about its centre. Both are validated like placement: every point inside the world box and at least
  contact_radius from other bodies' edges, no edges crossing (containment is fine: a body may end up inside another).
  A rejected move snaps back with a message, a rejected rotation doesn't happen.
- A container carries its contents (bodies inside it, at any depth) when moved, rotated, copied or duplicated.
  Deleting a container keeps its contents where they are.
- Hooks stay on the peg grid: a group with hooks moves in whole peg steps, and hooks snap to the nearest peg after a
  rotation. Ropes attached to moved bodies / hooks follow them; a rope whose length changes keeps its slack but never
  starts stretched (rest length = new length + old slack), so nothing gets yanked on resume.
- Delete / Backspace removes the selection plus every rope attached to a removed body or hook (a chain goes as a whole).
- Ctrl/Cmd+C copies the bodies (type, springs, stiffness, rest shape, colours), their contents and the hooks, plus
  every rope whose ends are all in the copy or free anchors, if the rope was selected (chain ropes are) or joins two
  copied objects. Ropes to anything not in the copy are dropped.
- Ctrl/Cmd+V shows the copy as a ghost following the mouse (Q / E rotate it); a click on a valid spot pastes it and
  selects the pasted objects, Esc cancels. Ctrl/Cmd+D duplicates the selection next to the original at the first valid
  spot nearby (right, below, left, above, then diagonals and further out), or as a paste ghost if there is no room.
Every change calls before_world_change(label) first (world-level undo).
"""
import copy
import math

import numpy as np
import pygame

from entities import EnvConfig, FixedBody, Hook, RigidBody, Rope, SoftBody, new_chain_id

ROTATE_STEP = 15.0  # degrees per Q / E press
BAD_MOVE = "you cannot put it there"
NOTHING_COPIED = "nothing copied yet"
NO_ROOM = "no room nearby: click to place the copy"
BAND_UI_PX = 3  # a press on empty space that moves less than this (UI px) is a click (clears the selection)


def _unit(obj):
    """The selectable object a piece belongs to: a chain piece's chain ("chain", id), else the piece itself."""
    cid = getattr(obj, "chain_id", None)
    return ("chain", cid) if cid is not None else obj


def _index_in(obj, items):
    return next((k for k, x in enumerate(items) if x is obj), None)


def _rotation(deg):
    t = math.radians(deg)
    return np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]])


class _Pose:
    """What a move / rotation / paste changes: body points, hook positions and the free anchors of the ropes involved,
    with their positions when the pose was taken."""

    def __init__(self, bodies, ropes, hooks, other_hooks=()):
        """other_hooks: hooks that aren't moving (e.g. the world's): a rope end on one isn't a free anchor either."""
        self.bodies, self.ropes, self.hooks = bodies, ropes, hooks
        hook_refs = {id(h.pos) for h in list(hooks) + list(other_hooks)}
        # free (x, y) anchors of these ropes (a hook end shares the hook's array: it moves only with its hook)
        self.anchors = [ref for r in ropes for body, ref in r.ends if body is None and id(ref) not in hook_refs]
        self.body_pts = [b.points.copy() for b in bodies]
        self.hook_pts = [h.pos.copy() for h in hooks]
        self.anchor_pts = [a.copy() for a in self.anchors]

    def all_points(self):
        parts = self.body_pts + [p.reshape(2, 1) for p in self.hook_pts + self.anchor_pts]
        return np.concatenate(parts, axis=1) if parts else np.zeros((2, 0))

    def centre(self):
        pts = self.all_points()
        return (pts.min(axis=1) + pts.max(axis=1)) / 2 if pts.size else np.zeros(2)

    def size(self):
        pts = self.all_points()
        return pts.max(axis=1) - pts.min(axis=1) if pts.size else np.zeros(2)

    def write(self, new):
        """Put the (body points, hook positions, anchors) `new` in place (the World's arrays and the ropes see it)."""
        for b, p in zip(self.bodies, new[0]):
            b.points[:] = p
        for h, p in zip(self.hooks, new[1]):
            h.pos[:] = p
        for a, p in zip(self.anchors, new[2]):
            a[:] = p

    def original(self):
        return self.body_pts, self.hook_pts, self.anchor_pts


class SelectionTools:
    """Editor mixin: see the module docstring. State: picked (selected units), clipboard (deep copies), paste (the
    ghost being placed), edit_drag (a move in progress), band (a box selection in progress)."""

    def _init_selection(self):
        self.picked = []
        self.clipboard = None  # (bodies, ropes, hooks): deep copies, independent of the world
        self.paste = None  # {"pose": _Pose of the ghost's own copies, "label": "paste" / "duplicate"}
        self.edit_drag = None  # {"pose", "start", "new", "valid", "moved"}
        self.band = None  # [start wp, current wp, additive]

    def before_world_change(self, label):
        """Called right before any edit that changes the world (spawn, delete, move, paste, rotate, load...), so a
        world-level undo can snapshot it. A no-op until that undo exists."""

    def _editing(self):
        """Selection editing is on while paused with the spawn editor closed."""
        return self.paused and not self.active

    def _end_edit(self):
        """Leave selection editing (resume / editor opened): a move in progress snaps back, a ghost is dropped and the
        selection cleared."""
        if self.edit_drag is not None:
            self.edit_drag["pose"].write(self.edit_drag["pose"].original())
        self.edit_drag = self.band = self.paste = None
        self.picked = []

    # ---------------- Units and what they contain
    def _unit_parts(self, unit):
        """(bodies, ropes, hooks) of a unit that are in the world now."""
        w = self.world
        if isinstance(unit, tuple):
            cid = unit[1]
            return ([b for b in w.bodies if getattr(b, "chain_id", None) == cid],
                    [r for r in w.ropes if getattr(r, "chain_id", None) == cid], [])
        if isinstance(unit, Hook):
            return [], [], [unit] if _index_in(unit, w.hooks) is not None else []
        if isinstance(unit, Rope):
            return [], [unit] if _index_in(unit, w.ropes) is not None else [], []
        return [unit] if _index_in(unit, w.bodies) is not None else [], [], []

    def _live_picked(self):
        """The selection without units that left the world (e.g. after a world undo)."""
        self.picked = [u for u in self.picked if any(self._unit_parts(u))]
        return self.picked

    def _gather(self, units, carry=True):
        """(bodies, ropes, hooks) of the units, each in world order. carry: also everything inside a container (any
        depth), and the whole chain of any chain box that comes along that way."""
        w = self.world
        bodies, ropes, hooks = {}, {}, {}
        for u in units:
            b, r, h = self._unit_parts(u)
            bodies.update((id(x), x) for x in b)
            ropes.update((id(x), x) for x in r)
            hooks.update((id(x), x) for x in h)
        while carry:
            more = {id(x): x for x in self._with_contents(bodies.values())}
            chains = {x.chain_id for x in more.values() if getattr(x, "chain_id", None) is not None}
            for cid in chains:
                b, r, _ = self._unit_parts(("chain", cid))
                more.update((id(x), x) for x in b)
                ropes.update((id(x), x) for x in r)
            if more.keys() <= bodies.keys():
                break
            bodies.update(more)
        return ([b for b in w.bodies if id(b) in bodies], [r for r in w.ropes if id(r) in ropes],
                [h for h in w.hooks if id(h) in hooks])

    def _selection_summary(self):
        units = self._live_picked()
        if not units:
            return "none"
        if len(units) > 1:
            return f"{len(units)} items"
        u = units[0]
        if isinstance(u, tuple):
            return "chain"
        if isinstance(u, (Hook, Rope)):
            return type(u).__name__.lower()
        return {SoftBody: "soft", RigidBody: "rigid", FixedBody: "fixed"}.get(type(u), "body")

    # ---------------- Picking
    def _pick_unit(self, wp):
        """The object under wp: a hook, else a rope within 2 UI px, else the innermost body (see _pick), else a rope
        within PICK_UI_PX. None if nothing is there."""
        from editor import PICK_UI_PX
        p, z, w = np.asarray(wp, dtype=float), self.zoom, self.world
        best, best_d = None, math.inf
        for h in w.hooks:
            d = float(np.hypot(*(h.pos - p)))
            if d <= (h.radius + 2) / z and d < best_d:
                best, best_d = h, d
        if best is not None:
            return best

        def rope_near(limit):
            from editor import point_seg_dist
            found, found_d = None, limit
            for r in w.ropes:
                d = float(point_seg_dist(p.reshape(2, 1), r.end_pos(0).reshape(2, 1), r.end_pos(1).reshape(2, 1))[0])
                if d <= found_d:
                    found, found_d = r, d
            return found

        rope = rope_near(2 / z)
        if rope is not None:
            return _unit(rope)
        body = self._pick(p)
        if body is not None:
            return _unit(body)
        rope = rope_near(PICK_UI_PX / z)
        return _unit(rope) if rope is not None else None

    def _units_in_rect(self, a, b):
        """Units lying fully inside the world rect with corners a, b."""
        lo, hi = np.minimum(a, b).reshape(2, 1), np.maximum(a, b).reshape(2, 1)

        def inside(pts):
            pts = np.asarray(pts, dtype=float).reshape(2, -1)
            return bool(((pts >= lo) & (pts <= hi)).all())

        w, units = self.world, []
        for body in w.bodies:
            if not self._is_wall(body) and _unit(body) not in units and inside(body.points):
                u = _unit(body)
                if not isinstance(u, tuple) or all(inside(x.points) for x in self._unit_parts(u)[0]):
                    units.append(u)
        for r in w.ropes:
            if _unit(r) not in units and inside(np.array([r.end_pos(0), r.end_pos(1)]).T):
                if not isinstance(_unit(r), tuple):  # a chain comes in only with all its boxes
                    units.append(r)
        units += [h for h in w.hooks if inside(h.pos)]
        return units

    # ---------------- Transforms and validation (world units)
    def _snap_hooks(self, hook_pts):
        """Each hook position on its nearest peg (left as is outside the world box, where validation rejects it)."""
        out = []
        for p in hook_pts:
            spot = self._hook_spot(p)
            out.append(spot if spot is not None else p)
        return out

    def _moved(self, pose, f, hooks_rigid=False):
        """New (body points, hooks, anchors) with every original position mapped by f ((2, k) -> (2, k)). Hooks then
        snap to pegs, unless hooks_rigid (a translation by whole peg steps keeps them on their pegs exactly)."""
        bodies = [f(p) for p in pose.body_pts]
        hooks = [f(p.reshape(2, 1))[:, 0] for p in pose.hook_pts]
        anchors = [f(p.reshape(2, 1))[:, 0] for p in pose.anchor_pts]
        return bodies, (hooks if hooks_rigid else self._snap_hooks(hooks)), anchors

    def _translated(self, pose, offset):
        """pose moved by offset; with hooks, the offset is rounded so the first hook lands on a peg (the others then
        do too: hooks sit on pegs)."""
        offset = np.asarray(offset, dtype=float)
        if pose.hook_pts:
            h0 = pose.hook_pts[0]
            spot = self._hook_spot(h0 + offset)
            if spot is not None:
                offset = spot - h0
        return self._moved(pose, lambda p: p + offset.reshape(2, 1), hooks_rigid=True), offset

    def _rotated(self, pose, deg):
        c = pose.centre().reshape(2, 1)
        rot = _rotation(deg)
        return self._moved(pose, lambda p: rot @ (p - c) + c)

    def _pose_ok(self, pose, new, exclude=()):
        """Can the pose's objects be at `new`? Bodies: inside the world box (by contact_radius) and no point or edge
        within contact_radius of another body's edge (crossing included); a body inside another is fine. The bodies in
        `exclude` (the ones being moved) aren't obstacles. Hooks: inside the box, never two on one peg. Free rope anchors:
        inside the box and not inside a body. The objects among themselves keep their (valid) layout."""
        from editor import closed, point_in_poly, segs_dist
        r, w = EnvConfig.contact_radius, self.world
        bodies_new, hooks_new, anchors_new = new
        lo, hi_x, hi_y = self.inset + r, self.world_w - self.inset - r, self.world_h - self.inset - r
        if bodies_new:
            pts = np.concatenate(bodies_new, axis=1)
            if (pts[0] < lo).any() or (pts[0] > hi_x).any() or (pts[1] < lo).any() or (pts[1] > hi_y).any():
                return False
            gone = [k for k, b in enumerate(w.bodies) if any(b is x for x in exclude)]
            keep = ~np.isin(w.edge_body, gone)
            a, b = w.P[:, w.edges[0, keep]], w.P[:, w.edges[1, keep]]
            e_lo, e_hi = np.minimum(a, b) - r, np.maximum(a, b) + r
            for body, p in zip(pose.bodies, bodies_new):
                p_lo, p_hi = p.min(axis=1, keepdims=True), p.max(axis=1, keepdims=True)
                near = ((e_lo <= p_hi) & (e_hi >= p_lo)).all(axis=0)
                if near.any() and segs_dist(p[:, body.edges[0]], p[:, body.edges[1]], a[:, near], b[:, near]).min() < r:
                    return False
        half = EnvConfig.peg_spacing / 2
        others = [h.pos for h in w.hooks if not any(h is x for x in pose.hooks)]
        for k, p in enumerate(hooks_new):
            if not self._in_box(p):
                return False
            if any(np.hypot(*(q - p)) < half for q in others + hooks_new[:k]):
                return False
        if anchors_new:
            polys = [b.points for b in w.bodies if closed(b) and not any(b is x for x in exclude)]
            polys += [p for body, p in zip(pose.bodies, bodies_new) if closed(body)]
            for p in anchors_new:
                if not self._in_box(p) or any(point_in_poly(p, poly) for poly in polys):
                    return False
        return True

    # ---------------- Committing changes
    def _rope_lengths(self):
        return [(r, r.length()) for r in self.world.ropes]

    @staticmethod
    def _retension(before):
        """Ropes whose length changed keep their slack but never start stretched: rest = new length + old slack."""
        for r, old in before:
            new = r.length()
            if abs(new - old) > 1e-9:
                r.rest_length = new + max(r.rest_length - old, 0.0)

    def _fix_order(self):
        """Containers draw before what they hold: if an edit put a body before its container, reorder by depth."""
        w, parents = self.world, self._parents()
        order = {id(b): k for k, b in enumerate(w.bodies)}
        if all(parents.get(b) is None or order[id(parents[b])] < order[id(b)] for b in w.bodies):
            return
        depth = {id(b): len(self._ancestors(b)) for b in w.bodies}
        w.reorder(sorted(w.bodies, key=lambda b: depth[id(b)]))

    def _commit_pose(self, pose, new, label):
        """Move the world's objects in `pose` to `new` as one undoable edit (they start at rest)."""
        pose.write(pose.original())  # a live drag already moved them: the snapshot must see the old state
        before = self._rope_lengths()
        self.before_world_change(label)
        pose.write(new)
        for b in pose.bodies:
            b.vels[:] = 0.0
            b.prev[:] = b.points
        self._retension(before)
        self._fix_order()

    # ---------------- Actions
    def delete_selection(self):
        units = self._live_picked()
        if not units:
            return
        bodies, ropes, hooks = self._gather(units, carry=False)
        self.before_world_change("delete")
        self.world.remove(bodies, ropes, hooks)
        self.picked = []

    def _copy_set(self, units):
        """Deep copies (bodies, ropes, hooks) of the units, their contents, and the ropes that come along (see the
        module docstring)."""
        bodies, ropes, hooks = self._gather(units)
        body_ids, hook_refs = {id(b) for b in bodies}, {id(h.pos) for h in hooks}
        world_hook_refs = {id(h.pos) for h in self.world.hooks}
        chosen = {id(r) for r in ropes}
        keep = []
        for r in self.world.ropes:
            ok, joined = True, 0
            for body, ref in r.ends:
                if body is not None:
                    ok, joined = ok and id(body) in body_ids, joined + 1
                elif id(ref) in world_hook_refs:
                    ok, joined = ok and id(ref) in hook_refs, joined + 1
            if ok and (id(r) in chosen or joined == 2):
                keep.append(r)
        if not (bodies or keep or hooks):
            return None
        return copy.deepcopy((bodies, keep, hooks))  # one memo: copied ropes point at the copied bodies / hooks

    def copy_selection(self):
        units = self._live_picked()
        if units:
            self.clipboard = self._copy_set(units)

    def _clip_count(self):
        if self.clipboard is None:
            return 0
        bodies, ropes, hooks = self.clipboard
        return len({_unit(x) if isinstance(_unit(x), tuple) else id(x) for x in bodies + ropes + hooks})

    def _start_ghost(self, clip, label):
        bodies, ropes, hooks = copy.deepcopy(clip)
        for b in bodies:
            b.vels[:] = 0.0
        self.paste = {"pose": _Pose(bodies, ropes, hooks), "label": label}
        self.edit_drag = self.band = None

    def start_paste(self):
        if self.clipboard is None:
            self._say(NOTHING_COPIED)
            return
        self._start_ghost(self.clipboard, "paste")

    def _ghost_new(self, wp):
        """(new positions, valid) of the paste ghost centred on wp."""
        pose = self.paste["pose"]
        new, _ = self._translated(pose, np.asarray(wp, dtype=float) - pose.centre())
        return new, self._pose_ok(pose, new)

    def _place_ghost(self, new):
        """Add the ghost's objects at `new` to the world, as fresh objects (chains get new ids), and select them."""
        pose, label = self.paste["pose"], self.paste["label"]
        before = [(r, r.length()) for r in pose.ropes]
        pose.write(new)
        for b in pose.bodies:
            b.prev[:] = b.points
        self._retension(before)
        ids = {}
        for piece in pose.bodies + pose.ropes:
            cid = getattr(piece, "chain_id", None)
            if cid is not None:
                piece.chain_id = ids.setdefault(cid, new_chain_id())
        self.before_world_change(label)
        self.world.add_many(pose.bodies, pose.ropes, pose.hooks)
        units = []
        for x in pose.bodies + pose.ropes + pose.hooks:
            if _unit(x) not in units:
                units.append(_unit(x))
        self.picked, self.paste = units, None
        self._fix_order()

    def duplicate_selection(self):
        """A copy next to the original at the first valid spot (see the module docstring), else a paste ghost."""
        units = self._live_picked()
        clip = self._copy_set(units) if units else None
        if clip is None:
            return
        self._start_ghost(clip, "duplicate")
        pose = self.paste["pose"]
        w, h = pose.size()
        gap = 2 * EnvConfig.contact_radius + 4
        for k in (1, 2, 3):
            for dx, dy in ((1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (-1, 1), (1, -1), (-1, -1)):
                new, _ = self._translated(pose, (k * dx * (w + gap), k * dy * (h + gap)))
                if self._pose_ok(pose, new):
                    self._place_ghost(new)
                    return
        self._say(NO_ROOM)

    def rotate_selection(self, sign):
        """Q (sign -1, counter-clockwise on screen) / E (+1): rotate the selection, or the paste ghost, by ROTATE_STEP."""
        if self.paste is not None:
            pose = self.paste["pose"]
            pose.write(self._rotated(pose, sign * ROTATE_STEP))
            self.paste["pose"] = _Pose(pose.bodies, pose.ropes, pose.hooks)  # the rotated copies are the new original
            return
        units = self._live_picked()
        if not units or self.edit_drag is not None:
            return
        pose = _Pose(*self._gather(units), self.world.hooks)
        new = self._rotated(pose, sign * ROTATE_STEP)
        if self._pose_ok(pose, new, exclude=pose.bodies):
            self._commit_pose(pose, new, "rotate")
        else:
            self._say(BAD_MOVE)

    def _esc_cancels(self):
        """Would Esc cancel something here (a ghost, a move, a box, the selection) instead of quitting?"""
        return self._editing() and (self.paste is not None or self.edit_drag is not None or self.band is not None
                                    or bool(self._live_picked()))

    # ---------------- Input (world coordinates)
    def _edit_key(self, event):
        """Keys while editing; True if used."""
        cmd = event.mod & (pygame.KMOD_CTRL | pygame.KMOD_META)
        k = event.key
        if k == pygame.K_ESCAPE:  # cancel a ghost / move, then clear the selection; with nothing to cancel Esc quits
            if self.paste is not None or self.edit_drag is not None or self.band is not None:
                picked = self.picked
                self._end_edit()
                self.picked = picked
                return True
            if self._live_picked():
                self.picked = []
                return True
            return False
        if k in (pygame.K_DELETE, pygame.K_BACKSPACE) and not cmd:
            self.delete_selection()
            return True
        if cmd and k == pygame.K_c:
            self.copy_selection()
            return True
        if cmd and k == pygame.K_v:
            self.start_paste()
            return True
        if cmd and k == pygame.K_d:
            self.duplicate_selection()
            return True
        if k in (pygame.K_q, pygame.K_e) and not cmd and (self.paste is not None or self._live_picked()):
            self.rotate_selection(-1 if k == pygame.K_q else 1)
            return True
        return False

    def _edit_down(self, wp):
        wp = np.asarray(wp, dtype=float)
        if self.paste is not None:
            new, ok = self._ghost_new(wp)
            if ok:
                self._place_ghost(new)
            else:
                self._say(BAD_MOVE)
            return
        shift = bool(pygame.key.get_mods() & pygame.KMOD_SHIFT)
        unit = self._pick_unit(wp)
        picked = self._live_picked()
        if unit is None:
            self.band = [wp, wp, shift]
            return
        if shift:
            if unit in picked:
                picked.remove(unit)
            else:
                picked.append(unit)
            return
        if unit not in picked:
            self.picked = [unit]
        pose = _Pose(*self._gather(self.picked), self.world.hooks)
        self.edit_drag = {"pose": pose, "start": wp, "new": pose.original(), "valid": True, "moved": False}

    def _edit_motion(self, wp):
        wp = np.asarray(wp, dtype=float)
        if self.band is not None:
            self.band[1] = wp
        elif self.edit_drag is not None:
            d = self.edit_drag
            new, offset = self._translated(d["pose"], wp - d["start"])
            d["moved"] = d["moved"] or float(np.hypot(*offset)) > 1e-9
            d["new"], d["valid"] = new, self._pose_ok(d["pose"], new, exclude=d["pose"].bodies)
            d["pose"].write(new)  # live: the world draws them where they'd go (red outline if invalid)

    def _edit_up(self, wp):
        if self.band is not None:
            a, _, additive = self.band
            b = np.asarray(wp, dtype=float)
            self.band = None
            if np.abs(b - a).max() * self.zoom < BAND_UI_PX:  # a click on empty space
                if not additive:
                    self.picked = []
                return
            found = self._units_in_rect(a, b)
            if additive:
                self.picked = self._live_picked() + [u for u in found if u not in self.picked]
            else:
                self.picked = found
            return
        if self.edit_drag is None:
            return
        self._edit_motion(wp)
        d, self.edit_drag = self.edit_drag, None
        if not d["moved"]:
            d["pose"].write(d["pose"].original())
        elif d["valid"]:
            self._commit_pose(d["pose"], d["new"], "move")
        else:
            d["pose"].write(d["pose"].original())
            self._say(BAD_MOVE)

    # ---------------- Hotkeys on the info card
    def _edit_hotkeys(self):
        """The Controls card's "Select" section while editing: what's selected and what can be done with it."""
        has = bool(self._live_picked())
        n = self._clip_count()
        return [("click", "select", self._selection_summary(), True),
                ("shift", "add / remove", None, True),
                ("drag", "move" if has else "box select", None, True),
                ("Q E", "rotate", None, has or self.paste is not None),
                ("DEL", "delete", None, has),
                ("^C", "copy", None, has),
                ("^V", "paste", f"{n} copied" if n else None, n > 0),
                ("^D", "duplicate", None, has)]

    def _edit_hint(self):
        """Is the paste hint box up (a paste ghost follows the mouse)?"""
        return self._editing() and self.paste is not None

    def _edit_hint_lines(self):
        pose = self.paste["pose"]
        what = "copy" if len(pose.bodies) + len(pose.hooks) == 1 else "copies"
        return [f"Click to place the {what}", "Q / E rotate  ·  Esc cancel"]

    # ---------------- Drawing
    def _draw_edit(self, surface, ghost_surface, wp, over_ui):
        """Selection outlines (yellow; red while a move would be rejected), the object under the mouse (white), the
        box being dragged out, and the paste ghost (grey, red where a click would be rejected)."""
        pal, z = self.palette, self.zoom
        if self.paste is not None:
            new, ok = self._ghost_new(wp)
            self._draw_ghost(surface, ghost_surface, self.paste["pose"], new, (200, 200, 200) if ok else tuple(pal.debug_red))
            return
        col = pal.yellow
        if self.edit_drag is not None and self.edit_drag["moved"] and not self.edit_drag["valid"]:
            col = pal.neon_red  # not debug_red: that's also a body colour (the demo square)
        if self.edit_drag is not None:  # what's being moved (the live positions would change what counts as contents)
            pose = self.edit_drag["pose"]
            bodies, ropes, hooks = pose.bodies, pose.ropes, pose.hooks
        else:
            bodies, ropes, hooks = self._gather(self._live_picked())
        for b in bodies:
            self._outline(surface, b, col)
        for r in ropes:
            pygame.draw.line(surface, col, r.end_pos(0) * z, r.end_pos(1) * z, 3)
        for h in hooks:
            pygame.draw.circle(surface, col, self._peg_px(h.pos), 6, 1)
        if self.band is not None:
            a, b = self.band[0] * z, self.band[1] * z
            rect = pygame.Rect(int(min(a[0], b[0])), int(min(a[1], b[1])), int(abs(b[0] - a[0])) + 1, int(abs(b[1] - a[1])) + 1)
            ghost_surface.fill((200, 200, 200), rect)
            pygame.draw.rect(surface, pal.white, rect, 1)
        elif self.edit_drag is None and not over_ui:
            unit = self._pick_unit(wp)
            if unit is not None and unit not in self.picked:
                hb, hr, hh = self._unit_parts(unit)
                for b in hb:
                    self._outline(surface, b, pal.white)
                for r in hr:
                    pygame.draw.line(surface, pal.white, r.end_pos(0) * z, r.end_pos(1) * z, 3)
                for h in hh:
                    pygame.draw.circle(surface, pal.white, self._peg_px(h.pos), 6, 1)

    def _draw_ghost(self, surface, ghost_surface, pose, new, col):
        z = self.zoom
        bodies_new, hooks_new, _ = new
        for body, p in zip(pose.bodies, bodies_new):
            loop = (p[:, body.edges[0]] * z).T.tolist()
            if len(loop) >= 3:
                pygame.draw.polygon(ghost_surface, col, loop)
                pygame.draw.lines(surface, col, True, loop, 1)
            else:
                pygame.draw.line(ghost_surface, col, (p[:, 0] * z).tolist(), (p[:, 1] * z).tolist(), 3)
        # ropes: their ends at the ghost positions (a body end at its copy's point, a hook / anchor where it'd go)
        where = {id(b): p for b, p in zip(pose.bodies, bodies_new)}
        moved = {id(a): p for a, p in zip(pose.anchors, new[2])}
        moved.update({id(h.pos): p for h, p in zip(pose.hooks, hooks_new)})
        for r in pose.ropes:
            ends = [where[id(body)][:, ref] if body is not None else moved.get(id(ref), ref) for body, ref in r.ends]
            pygame.draw.line(ghost_surface, col, (ends[0] * z).tolist(), (ends[1] * z).tolist(), max(1, r.width))
        for p in hooks_new:
            pygame.draw.circle(ghost_surface, col, self._peg_px(p), 4)
            pygame.draw.circle(surface, col, self._peg_px(p), 6, 1)
