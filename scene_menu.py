"""The editor's save and open dialogs (Ctrl/Cmd+S, Ctrl/Cmd+O), drawn on the UI surface in the shared pixel style.

Both are modal: while one is open it takes every key and click. Save captures the world the moment it opens (what you
saw when you pressed Ctrl+S), suggests the first free scene_N name, and writes scenes/<name>.json on Enter / Save.
Open lists the presets (scene.PRESETS) and the saved scenes; clicking one validates it and replaces the world (an
undoable step), or shows why it can't be loaded. Positions are UI-surface pixels, like the editor's other panels.
"""
import os

import pygame

import scene_io
import ui_style as ui

PANEL_W = 240
ROW_H = 18
ROW_GAP = 3
HEADER_H = 30
PAD = 8
BTN_H = 18
NAME_MAX = 24
NAME_CHARS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")


class SceneMenus:
    def __init__(self, editor):
        self.ed = editor
        self.state = None  # None, "save" or "load"
        self.name = ""  # the save dialog's name field
        self.pending = None  # scene_io.scene_data() captured when the save dialog opened
        self.scroll = 0  # first visible row of the open dialog
        self.folder = None  # where scenes are saved / listed (None: scene_io.SCENES_DIR)

    # ---------------- Actions
    def box(self):
        ed = self.ed
        return ed.world_w, ed.world_h, ed.inset

    def open_save(self):
        self.ed.world.release()
        self.pending = scene_io.scene_data(self.ed.world, self.box(), self.ed.view)
        self.name, self.state = scene_io.auto_name(self.folder), "save"

    def open_load(self):
        ed = self.ed
        if ed.active and ed.stage == "place" and ed.points:
            ed._say("finish or undo the new object first")
            return
        ed.world.release()
        self.scroll, self.state = 0, "load"

    def cancel(self):
        self.state, self.pending = None, None

    def save(self):
        name = scene_io.safe_name(self.name) if self.name.strip() else scene_io.auto_name(self.folder)
        try:
            path = scene_io.write_data(self.pending, name, self.folder)
        except OSError as e:
            self.ed._say(f"can't save ({type(e).__name__})")
            return
        self.cancel()
        self.ed._say(f"saved {os.path.basename(path)}")

    def entries(self):
        """(kind, name) rows of the open dialog: the presets, then the saved scenes."""
        from scene import PRESETS
        return [("preset", n) for n in PRESETS] + [("file", n) for n in scene_io.list_scenes(self.folder)]

    def load(self, kind, name):
        """Validate and load a preset or saved scene. On success the editor closes (the world it was working on is
        gone) and the load is one undo step; on failure the world is untouched and the reason is shown."""
        ed = self.ed
        try:
            if kind == "preset":
                data = scene_io.preset_data(name, self.box())
            else:
                data = scene_io.read_file(scene_io.scene_path(name, self.folder))
            parsed = scene_io.parse(data, self.box())
        except scene_io.SceneError as e:
            ed._say(f"can't load {name}: {e}")
            return False
        except Exception as e:  # a preset builder that fails
            ed._say(f"can't load {name} ({type(e).__name__})")
            return False
        ed.before_world_change("load", env=True)
        scene_io.install(ed.world, *parsed)
        env = data.get("env", scene_io.DEFAULT_ENV)
        try:
            scene_io.apply_env(env, ed.world)
        except (KeyError, TypeError, ValueError):
            scene_io.apply_env(scene_io.DEFAULT_ENV, ed.world)
        if ed.view is not None and isinstance(data.get("view"), dict):
            ed.view.filled = bool(data["view"].get("filled", ed.view.filled))
            ed.view.show_grid = bool(data["view"].get("show_grid", ed.view.show_grid))
        self.cancel()
        if ed.active:
            ed.close()
        ed.world_replaced()
        ed._say(f"loaded {name}")
        return True

    # ---------------- Layout (UI px)
    def _panel(self, h):
        ed = self.ed
        area = ed._area()
        w = min(PANEL_W, area.w - 2 * ui.CARD_MARGIN)
        h = min(h, ed.H - 2 * ui.CARD_MARGIN)
        return pygame.Rect(max(ui.CARD_MARGIN, area.centerx - w // 2), max(ui.CARD_MARGIN, area.centery - h // 2), w, h)

    def _visible_rows(self):
        room = self.ed.H - 2 * ui.CARD_MARGIN - HEADER_H - 2 * PAD - ui.font().get_height()
        return max(1, (room + ROW_GAP) // (ROW_H + ROW_GAP))

    def layout(self):
        """{"panel", "close", ...}: the save dialog's "field", "cancel", "save"; the open dialog's "rows"
        [(rect, kind, name)], "more" (rows hidden below / above)."""
        lay = {}
        if self.state == "save":
            h = HEADER_H + ui.font().get_height() + 2 + BTN_H + 4 + ui.font().get_height() + PAD + BTN_H + PAD
            panel = lay["panel"] = self._panel(h)
            y = panel.y + HEADER_H
            lay["label"] = (panel.x + PAD, y)
            y += ui.font().get_height() + 2
            lay["field"] = pygame.Rect(panel.x + PAD, y, panel.w - 2 * PAD, BTN_H)
            y += BTN_H + 4
            lay["path"] = (panel.x + PAD, y)
            bw = (panel.w - 3 * PAD) // 2
            by = panel.bottom - PAD - BTN_H
            lay["cancel"] = pygame.Rect(panel.x + PAD, by, bw, BTN_H)
            lay["save"] = pygame.Rect(panel.right - PAD - bw, by, bw, BTN_H)
        elif self.state == "load":
            entries = self.entries()
            shown = min(len(entries), self._visible_rows())
            self.scroll = max(0, min(self.scroll, len(entries) - shown))
            hint_h = ui.font().get_height()
            h = HEADER_H + shown * (ROW_H + ROW_GAP) - ROW_GAP + PAD + hint_h + PAD
            panel = lay["panel"] = self._panel(h)
            y = panel.y + HEADER_H
            lay["rows"] = [(pygame.Rect(panel.x + PAD, y + i * (ROW_H + ROW_GAP), panel.w - 2 * PAD, ROW_H), kind, name)
                           for i, (kind, name) in enumerate(entries[self.scroll:self.scroll + shown])]
            lay["hint"] = (panel.centerx, panel.bottom - PAD - hint_h // 2)
            lay["more"] = (self.scroll > 0, self.scroll + shown < len(entries))
        if lay:
            p = lay["panel"]
            lay["close"] = pygame.Rect(p.right - 6 - BTN_H, p.y + 6, BTN_H, BTN_H)
        return lay

    def widgets(self):
        """(rect, action, draw spec) in UI px, like Editor._panel_widgets."""
        lay = self.layout()
        if not lay:
            return []
        ws = [(lay["close"], self.cancel, ("button", "x", False, False))]
        if self.state == "save":
            ws += [(lay["cancel"], self.cancel, ("button", "Cancel", False, False)),
                   (lay["save"], self.save, ("button", "Save", True, False))]
        else:
            ws += [(rect, lambda k=kind, n=name: self.load(k, n), ("scene_row", kind, name)) for rect, kind, name in lay["rows"]]
        return ws

    # ---------------- Events (mouse positions in real screen px)
    def handle_event(self, event):
        """While a dialog is open: use the event (always True: the dialogs are modal)."""
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                self.cancel()
            elif self.state == "save":
                if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                    self.save()
                elif event.key == pygame.K_BACKSPACE:
                    self.name = self.name[:-1]
                else:
                    ch = getattr(event, "unicode", "") or ""
                    ch = "_" if ch == " " else ch
                    if len(ch) == 1 and ch in NAME_CHARS and len(self.name) < NAME_MAX:
                        self.name += ch
            elif self.state == "load":
                if event.key in (pygame.K_UP, pygame.K_PAGEUP):
                    self.scroll -= 1 if event.key == pygame.K_UP else self._visible_rows()
                elif event.key in (pygame.K_DOWN, pygame.K_PAGEDOWN):
                    self.scroll += 1 if event.key == pygame.K_DOWN else self._visible_rows()
                self.scroll = max(0, self.scroll)
            return True
        if event.type == pygame.MOUSEWHEEL and self.state == "load":
            self.scroll = max(0, self.scroll - event.y)
            return True
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            pos = self.ed.to_ui(event.pos)
            for rect, action, _ in self.widgets():
                if rect.collidepoint(pos):
                    action()
                    break
        return True

    # ---------------- Drawing (UI surface)
    def draw(self, surface, mouse_ui):
        lay = self.layout()
        if not lay:
            return
        ed = self.ed
        surface.blit(ed._dim, (0, 0))
        panel = lay["panel"]
        ui.draw_frame(surface, panel, ui.PANEL_BG)
        title = "Save scene" if self.state == "save" else "Open scene"
        ui.text(surface, title, (panel.centerx, panel.y + 6 + BTN_H // 2), ui.WHITE, "bold", ui.TITLE_SIZE, "center", "middle")
        if self.state == "save":
            ui.text(surface, "Name", lay["label"], ui.TEXT)
            field = lay["field"]
            ui.draw_frame(surface, field, ui.CARD_BG)
            text = self.name + "_"  # the caret
            while len(text) > 1 and ui.font().size(text)[0] > field.w - 8:
                text = text[1:]  # long names scroll: keep the end (where you type) in view
            ui.text(surface, text, (field.x + 4, field.centery), ui.WHITE, valign="middle")
            name = scene_io.safe_name(self.name) if self.name.strip() else "?"
            exists = name in scene_io.list_scenes(self.folder)
            note = f"scenes/{name}.json" + ("  (replaces it)" if exists else "")
            while ui.font().size(note)[0] > panel.w - 2 * PAD and len(note) > 4:
                note = note[:-4] + "..."
            ui.text(surface, note, lay["path"], ui.ACCENT if exists else ui.TEXT_DIM)
        else:
            if not lay["rows"]:
                ui.text(surface, "no scenes", (panel.centerx, panel.y + HEADER_H), ui.TEXT_DIM, align="center")
            up, down = lay["more"]
            hint = "click to load  ·  Esc cancel"
            if up or down:  # where the list is, instead of the click hint (the wheel / arrow keys scroll)
                n = len(self.entries())
                hint = f"{self.scroll + 1}-{self.scroll + len(lay['rows'])} of {n}  ·  wheel scrolls  ·  Esc cancel"
            while ui.font().size(hint)[0] > panel.w - 2 * PAD and "  ·  " in hint:
                hint = hint.rsplit("  ·  ", 1)[0]
            ui.text(surface, hint, lay["hint"], ui.TEXT_DIM, align="center", valign="middle")
        for rect, _, spec in self.widgets():
            hovered = rect.collidepoint(mouse_ui)
            if spec[0] == "scene_row":
                self._draw_row(surface, rect, spec[1], spec[2], hovered)
            else:
                ed._draw_widget(surface, rect, spec, hovered)

    @staticmethod
    def _draw_row(surface, rect, kind, name, hovered):
        ui.draw_button(surface, rect, None, hovered)
        tag = "preset" if kind == "preset" else "saved"
        tag_w = ui.font().size(tag)[0]
        label = name
        while len(label) > 1 and ui.font("bold").size(label)[0] > rect.w - tag_w - 18:
            label = label[:-2] + "~"
        ui.text(surface, label, (rect.x + 6, rect.centery), ui.WHITE, "bold", valign="middle")
        ui.text(surface, tag, (rect.right - 6, rect.centery), ui.TEXT_DIM, align="right", valign="middle")
