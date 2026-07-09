"""
Centralised visual style tokens for napari-flopa widgets.

Usage
-----
    from napari_flopa.ui.style import C, SS, MPL, apply_style

    apply_style(group_box, SS.GROUP_A)
    apply_style(label, SS.STATUS)
    apply_style(btn, SS.BTN_DANGER)
    ax.set_facecolor(MPL.AXES_BG)

All Qt stylesheet strings are module-level constants on ``SS``.
All raw hex colours are on ``C``.
Matplotlib plot colours are on ``MPL``.

Group box title variants
------------------------
  SS.GROUP_A  — primary sections; yellow title (#f5ea1d).
                Supports the ``#plain`` object-name selector for a muted
                gray title: ``box.setObjectName("plain")``.
  SS.GROUP_B  — secondary / nested sections; orange title (#ffbc2b).
  SS.GROUP_COMPACT — bordered compact box used in dense panel layouts.
"""

# ──────────────────────────────────────────────────────────────────────────────
# Colour tokens
# ──────────────────────────────────────────────────────────────────────────────


class C:
    """Raw hex colour tokens — single source of truth for the dark theme."""

    # Backgrounds
    BG_DEEP = "#1a1a1a"  # deepest background (log, canvas fill)
    BG_DARK = "#1e1e1e"  # panel / plot background
    BG_MID = "#2b2b2b"  # input field / axes background
    BG_RAISED = "#333333"  # slider groove, subtle raised surface
    BG_SUBTLE = "#0D394D"  # "#1e2a38" before

    # Borders / separators
    BORDER = "#444444"
    BORDER_SOFT = "#333333"
    BORDER_DEFAULT = "#555555"

    # Text
    TEXT = "#cccccc"  # primary text
    TEXT_MUTED = "#aaaaaa"  # secondary / read-only text
    TEXT_DIM = "#888888"  # status / hint text
    TEXT_FAINT = "#777777"  # extra-faint hint
    TEXT_DARK = "#555555"  # disabled / inactive

    # Accent — cyan (contrast / view slider)
    CYAN = "#00dcdc"
    CYAN_DIM = "#00aaaa"
    CYAN_BG = "#2a4a4a"
    CYAN_BG_HOV = "#2e5555"

    # Accent — red (mask / danger)
    RED = "#dc4444"
    RED_DIM = "#aa2222"
    RED_BG = "#4a1a1a"
    RED_TEXT = "#f25555"
    RED_SOFT = "#ff8080"
    RED_DARK = "#664444"
    RED_BG_DIS = "#2a1a1a"

    # Accent — green (success / done)
    GREEN = "#88ff88"
    GREEN_BG = "#2a4a2a"

    # Accent — orange (warning / secondary title)
    ORANGE = "#ffaa44"  # warning label
    TITLE_PRIMARY = "#f5df1d"  # GROUP_A main title (yellow) # was "#f5ea1d"
    TITLE_PLAIN = "#e0e0e0"  # GROUP_A #plain variant (light gray)
    TITLE_SECONDARY = "#ffbc2b"  # GROUP_B title (amber)

    # Stale indicator
    STALE_INACTIVE = "#555555"
    STALE_STALE = "#ff4444"
    STALE_FRESH = "#44cc44"

    # Parameter provenance (source: metadata / default / user)
    # Yellow=metadata; grey=default; blue=user. (Red/green are reserved for
    # the plot stale/fresh indicator.)
    PROV_METADATA = "#f5df1d"
    PROV_DEFAULT = "#888888"
    PROV_USER = "#4a90d9"


# ──────────────────────────────────────────────────────────────────────────────
# Qt stylesheet strings
# ──────────────────────────────────────────────────────────────────────────────


class SS:
    """Qt stylesheet strings for common widget roles."""

    # ── Labels ────────────────────────────────────────────────────────────────

    # Provenance dot, keyed by source ('metadata' | 'default' | 'user')
    PROV_DOT = {
        "metadata": f"color: {C.PROV_METADATA}; font-size: 11px;",
        "default": f"color: {C.PROV_DEFAULT}; font-size: 11px;",
        "user": f"color: {C.PROV_USER}; font-size: 11px;",
    }

    STATUS = f"color: {C.TEXT_DIM}; font-size: 10px;"
    HINT = f"color: {C.TEXT_FAINT}; font-size: 9px; font-weight: normal;"
    MUTED = f"color: {C.TEXT_MUTED}; font-weight: normal;"
    WARNING = f"color: {C.ORANGE}; font-size: 9px; font-weight: normal;"
    SEPARATOR = f"color: {C.BORDER};"

    # ── Stale indicator (● dot label) ─────────────────────────────────────────

    STALE_INACTIVE = f"color: {C.STALE_INACTIVE}; font-size: 16px;"
    STALE_STALE = f"color: {C.STALE_STALE};    font-size: 16px;"
    STALE_FRESH = f"color: {C.STALE_FRESH};    font-size: 16px;"

    # ── Read-only display (e.g. calibration factor) ───────────────────────────

    DISPLAY = f"color: {C.TEXT_MUTED}; font-family: monospace;"

    # ── Buttons ───────────────────────────────────────────────────────────────

    BTN_DANGER = (
        f"QPushButton {{ color: {C.RED_TEXT}; }}"
        f"QPushButton:disabled {{ color: {C.RED_DIM}; }}"
    )

    BTN_SUCCESS = f"QPushButton {{ color: {C.GREEN}; }}"

    BTN_RUN = (
        f"QPushButton {{ background: {C.GREEN_BG}; color: {C.GREEN}; "
        f"font-weight: bold; padding: 3px 12px; }}"
    )

    BTN_STOP = (
        f"QPushButton {{ background: {C.RED_BG}; color: {C.RED_SOFT}; "
        f"font-weight: bold; padding: 3px 12px; }}"
        f"QPushButton:disabled {{ background: {C.RED_BG_DIS}; color: {C.RED_DARK}; }}"
    )

    BTN_SMALL = "font-size: 10px;"

    # Detector toggle buttons (cyan = active, gray = inactive, dark = disabled)
    BTN_DET_ON = (
        f"QPushButton {{ background: {C.CYAN_BG}; color: {C.CYAN}; "
        f"border: 1px solid {C.CYAN_DIM}; border-radius: 3px; "
        f"padding: 1px 6px; font-size: 10px; }}"
        f"QPushButton:hover {{ background: {C.CYAN_BG_HOV}; }}"
    )
    BTN_DET_OFF = (
        f"QPushButton {{ background: {C.BG_MID}; color: {C.TEXT_DARK}; "
        f"border: 1px solid {C.BORDER}; border-radius: 3px; "
        f"padding: 1px 6px; font-size: 10px; }}"
    )
    BTN_DET_DISABLED = (
        f"QPushButton {{ background: #222222; color: #444444; "
        f"border: 1px solid {C.BORDER_SOFT}; border-radius: 3px; "
        f"padding: 1px 6px; font-size: 10px; }}"
    )

    # ── Input fields ──────────────────────────────────────────────────────────

    LINE_EDIT = (
        f"QLineEdit {{ background: {C.BG_MID}; color: {C.TEXT}; "
        f"border: 1px solid {C.BORDER}; border-radius: 2px; "
        f"padding: 1px 2px; font-size: 9px; }}"
        f"QLineEdit:focus {{ border: 1px solid #888888; }}"
    )

    # ── Sliders ───────────────────────────────────────────────────────────────

    SLIDER_VIEW = (
        f"QSlider::groove:horizontal {{ background: {C.BG_RAISED}; height: 4px; border-radius: 2px; }}"
        f"QSlider::handle:horizontal {{ background: {C.CYAN}; width: 10px; height: 10px;"
        f" margin: -3px 0; border-radius: 5px; }}"
        f"QSlider::sub-page:horizontal {{ background: {C.CYAN_DIM}; border-radius: 2px; }}"
    )

    SLIDER_MASK = (
        f"QSlider::groove:horizontal {{ background: {C.BG_RAISED}; height: 4px; border-radius: 2px; }}"
        f"QSlider::handle:horizontal {{ background: {C.RED}; width:10px; height: 10px;"
        f" margin: -3px 0; border-radius: 5px; }}"
        f"QSlider::sub-page:horizontal {{ background: {C.RED_DIM}; border-radius: 2px; }}"
    )

    # ── Group boxes ───────────────────────────────────────────────────────────

    # Primary sections — yellow title; set objectName("plain") for the gray variant.
    GROUP_A = f"""
    QGroupBox {{
        margin-top: 14px;
        border: 1px {C.TITLE_PRIMARY};
        border-radius: 0px;
        background-color: {C.BG_SUBTLE};
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        padding: 0px 2px;
        font-size: 12pt;
        font-weight: bold;
        color: {C.TITLE_PRIMARY};
    }}
    QGroupBox#plain::title {{
        subcontrol-origin: margin;
        padding: 0px 2px;
        font-size: 12pt;
        font-weight: bold;
        color: {C.TITLE_PLAIN};
    }}
    """

    # Secondary / nested sections — amber title.
    GROUP_B = f"""
    QGroupBox {{
        margin-top: 1px;
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        padding: 0px 2px;
        font-size: 12pt;
        font-weight: bold;
        color: {C.TITLE_SECONDARY};
    }}
    """

    # Top-level container title in cyan — used for FLIM View and similar dock wrappers.
    GROUP_TITLE = f"""
    QGroupBox {{
        margin-top: 14px;
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        padding: 0px 2px;
        font-size: 12pt;
        color: {C.CYAN_DIM};
    }}
    """

    # Compact bordered box for dense panel layouts (e.g. batch_panel sections).
    GROUP_COMPACT = f"""
    QGroupBox {{
        border: 1px solid {C.BORDER};
        border-radius: 3px;
        margin-top: 8px;
        padding-top: 4px;
        font-weight: bold;
        color: {C.TEXT};
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        left: 8px;
        padding: 0 4px;
    }}
    """

    # ── Log / console ─────────────────────────────────────────────────────────

    LOG = (
        f"QPlainTextEdit {{ background: #000000; color: {C.TEXT_MUTED}; "
        f"border: 1px solid {C.BORDER_SOFT}; font-family: monospace; font-size: 9px; }}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Helper
# ──────────────────────────────────────────────────────────────────────────────


def apply_style(widget, style_string: str) -> None:
    """Apply a Qt stylesheet string to *widget*. Convenience wrapper."""
    widget.setStyleSheet(style_string)


# ──────────────────────────────────────────────────────────────────────────────
# Matplotlib theme
# ──────────────────────────────────────────────────────────────────────────────


class MPL:
    """Colour values for matplotlib figure/axes styling (not Qt stylesheets)."""

    FIG_BG = C.BG_DARK  # figure.facecolor
    AXES_BG = C.BG_MID  # axes.facecolor
    TICK = C.TEXT  # tick label colour
    SPINE = C.TEXT_DARK  # axes spine colour
    GRID = "#3a3a3a"  # grid line colour
