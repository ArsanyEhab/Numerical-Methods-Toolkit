import io
import ast
import math
import re
import subprocess
import sys
import time
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import tkinter as tk
from tkinter import messagebox, ttk


@dataclass(frozen=True)
class MethodScript:
    label: str
    filename: str

    @property
    def path(self) -> Path:
        return Path(__file__).resolve().parent / self.filename


@dataclass
class Field:
    """Declarative description of a MethodWindow input field.

    kind:
      - "entry"   -> regular text entry (default)
      - "combo"   -> dropdown of `choices`
      - "check"   -> checkbox; value is "0"/"1" string
      - "spin"    -> integer spinbox between [min, max] step `step`
    """
    name: str
    default: str = ""
    kind: str = "entry"
    choices: Optional[list] = None
    spin_from: float = 0.0
    spin_to: float = 100.0
    spin_step: float = 1.0
    help: Optional[str] = None


# Per-method short description shown in the method window header.
METHOD_DESCRIPTIONS: dict[str, str] = {
    "secant": "Find a root by secant updates between two estimates (no derivative needed).",
    "fixed_point": "Iterate x = g(x) starting from x0; converges if |g'| < 1 near the root.",
    "false_position": "Bracket-based linear interpolation; keeps a sign change between brackets.",
    "halley": "Cubically-convergent root finder using f, f' and f''.",
    "newton": "Quadratic-convergence root finder using f and f'.",
    "bisection": "Robust bracket method: halve [a,b] each step until tolerance is met.",
    "jacobi": "Iterative solver for linear systems; uses only the previous iterate.",
    "gauss_seidel": "Iterative solver that uses freshly updated components within each sweep.",
    "newton_system": "Newton's method on F(x)=0 for nonlinear systems via the Jacobian J.",
    "trapezoidal": "Composite Trapezoidal Rule. Approximates integral by trapezoids.",
    "simpson": "Composite Simpson's Rule (n must be even). Quartic-accurate.",
    "gauss_quad": "Gauss-Legendre quadrature with 1, 2 or 3 nodes (exact for polynomial degree 2k-1).",
    "romberg": "Romberg = trapezoidal rule + Richardson extrapolation. Very fast convergence.",
    "differentiation": "Finite-difference derivatives from a function or tabulated data.",
}


# Modern palette (Tailwind slate + indigo)
COLORS = {
    "bg":         "#f1f5f9",   # slate-100  (app body)
    "bg_alt":     "#e2e8f0",   # slate-200
    "panel":      "#ffffff",   # cards / form
    "border":     "#e2e8f0",
    "border_lo":  "#f1f5f9",
    "text":       "#0f172a",   # slate-900
    "text_mute":  "#64748b",   # slate-500
    "text_soft":  "#94a3b8",   # slate-400
    "accent":     "#6366f1",   # indigo-500
    "accent_hi":  "#4f46e5",   # indigo-600
    "accent_lo":  "#eef2ff",   # indigo-50
    "sidebar":    "#0f172a",   # slate-900
    "sidebar_hi": "#1e293b",   # slate-800
    "sidebar_text": "#e2e8f0",
    "sidebar_mute": "#94a3b8",
    "console_bg": "#0b1220",
    "console_fg": "#e2e8f0",
    "chip_idle_bg":  "#e2e8f0",  "chip_idle_fg":  "#475569",
    "chip_run_bg":   "#dbeafe",  "chip_run_fg":   "#1e40af",
    "chip_done_bg":  "#dcfce7",  "chip_done_fg":  "#166534",
    "chip_err_bg":   "#fee2e2",  "chip_err_fg":   "#991b1b",
}


# Category metadata (icon + accent color used for the sidebar nav and method chips).
CATEGORY_INFO: dict[str, dict] = {
    "root":           {"label": "Root Finding",     "icon": "\u0192",   "color": "#6366f1",
                        "desc":  "Single-variable equation solvers f(x) = 0."},
    "systems":        {"label": "Systems",          "icon": "\u229E",   "color": "#0ea5e9",
                        "desc":  "Linear systems Ax = b and nonlinear systems F(x) = 0."},
    "integration":    {"label": "Integration",      "icon": "\u222B",   "color": "#10b981",
                        "desc":  "Approximate definite integrals \u222B f(x) dx."},
    "differentiation": {"label": "Differentiation", "icon": "\u2202",   "color": "#f59e0b",
                        "desc":  "Estimate derivatives from a function or tabulated data."},
}

METHOD_CATEGORY: dict[str, str] = {
    "bisection": "root", "false_position": "root", "fixed_point": "root",
    "secant": "root", "newton": "root", "halley": "root",
    "jacobi": "systems", "gauss_seidel": "systems", "newton_system": "systems",
    "trapezoidal": "integration", "simpson": "integration",
    "gauss_quad": "integration", "romberg": "integration",
    "differentiation": "differentiation",
}


def _normalize_fields(specs) -> list[Field]:
    """Accept legacy 2-tuples and Field instances; return list[Field]."""
    out: list[Field] = []
    for s in specs:
        if isinstance(s, Field):
            out.append(s)
        elif isinstance(s, (list, tuple)) and len(s) >= 2:
            out.append(Field(name=str(s[0]), default=str(s[1])))
        else:
            raise TypeError(f"Unsupported field spec: {s!r}")
    return out


def _center_on_screen(win: tk.Misc, parent: Optional[tk.Misc] = None) -> None:
    """Center a Toplevel/Tk window on screen (or relative to parent if provided)."""
    win.update_idletasks()
    w = win.winfo_width() or win.winfo_reqwidth()
    h = win.winfo_height() or win.winfo_reqheight()
    if parent is not None and parent.winfo_viewable():
        px = parent.winfo_rootx()
        py = parent.winfo_rooty()
        pw = parent.winfo_width()
        ph = parent.winfo_height()
        x = px + (pw - w) // 2
        y = py + (ph - h) // 2
    else:
        sw = win.winfo_screenwidth()
        sh = win.winfo_screenheight()
        x = (sw - w) // 2
        y = (sh - h) // 3
    win.geometry(f"+{max(0, x)}+{max(0, y)}")


def _apply_global_theme(root: tk.Misc) -> None:
    """Apply a modern slate + indigo theme via ttk styles. Idempotent."""
    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass

    bg = COLORS["bg"]
    panel = COLORS["panel"]
    text = COLORS["text"]
    muted = COLORS["text_mute"]
    border = COLORS["border"]
    accent = COLORS["accent"]
    accent_hi = COLORS["accent_hi"]

    root.configure(bg=bg)

    base_font = ("Segoe UI", 10)
    style.configure(".", background=bg, foreground=text, font=base_font)
    style.configure("TFrame", background=bg)
    style.configure("App.TFrame", background=bg)
    style.configure("Panel.TFrame", background=panel, relief="flat")
    style.configure("Sidebar.TFrame", background=COLORS["sidebar"])

    style.configure("Card.TLabelframe", background=panel, relief="flat",
                    borderwidth=1, bordercolor=border)
    style.configure("Card.TLabelframe.Label", background=panel, foreground=muted,
                    font=("Segoe UI", 9, "bold"))

    style.configure("TLabel", background=bg, foreground=text)
    style.configure("Header.TLabel", background=bg, foreground=text,
                    font=("Segoe UI", 22, "bold"))
    style.configure("PageHeader.TLabel", background=bg, foreground=text,
                    font=("Segoe UI", 22, "bold"))
    style.configure("Sub.TLabel", background=bg, foreground=muted,
                    font=("Segoe UI", 10))
    style.configure("PageSub.TLabel", background=bg, foreground=muted,
                    font=("Segoe UI", 10))
    style.configure("Section.TLabel", background=bg, foreground=text,
                    font=("Segoe UI", 11, "bold"))
    style.configure("Help.TLabel", background=panel, foreground=muted,
                    font=("Segoe UI", 9, "italic"))
    style.configure("Status.TLabel", background=bg, foreground=muted,
                    font=("Segoe UI", 9))

    style.configure("TButton", padding=(10, 6),
                    background=panel, foreground=text,
                    bordercolor=border, focusthickness=0)
    style.map("TButton",
              background=[("active", COLORS["bg_alt"]), ("pressed", COLORS["bg_alt"])],
              bordercolor=[("active", border)])

    style.configure("Accent.TButton", padding=(16, 9),
                    foreground="white", background=accent,
                    font=("Segoe UI", 10, "bold"), borderwidth=0,
                    focusthickness=0)
    style.map("Accent.TButton",
              background=[("active", accent_hi), ("pressed", accent_hi)],
              foreground=[("active", "white"), ("pressed", "white")])

    style.configure("Ghost.TButton", padding=(12, 7),
                    background=bg, foreground=text,
                    bordercolor=border, focusthickness=0)
    style.map("Ghost.TButton",
              background=[("active", COLORS["bg_alt"]), ("pressed", COLORS["bg_alt"])])

    style.configure("Tab.TButton", padding=(12, 10), anchor="w",
                    font=("Segoe UI", 10))

    # Notebook is no longer used by the start menu, but other code may use it.
    style.configure("TNotebook", background=bg, borderwidth=0)
    style.configure("TNotebook.Tab", padding=(14, 8), font=("Segoe UI", 10))
    style.map("TNotebook.Tab",
              background=[("selected", panel)],
              foreground=[("selected", text)])

    style.configure("TEntry", fieldbackground="white", foreground=text,
                    bordercolor=border, lightcolor=border, darkcolor=border,
                    padding=4)
    style.map("TEntry",
              bordercolor=[("focus", accent)],
              lightcolor=[("focus", accent)],
              darkcolor=[("focus", accent)])
    style.configure("TCombobox", fieldbackground="white", foreground=text,
                    arrowsize=14, padding=4)
    style.map("TCombobox",
              fieldbackground=[("readonly", "white")],
              bordercolor=[("focus", accent)])
    style.configure("TSpinbox", fieldbackground="white", foreground=text, padding=4)
    style.configure("TCheckbutton", background=panel, foreground=text)
    style.configure("Vertical.TScrollbar", background=bg, troughcolor=bg,
                    bordercolor=bg, arrowcolor=muted)


# ----------------------------------------------------------------------
#  Modern custom widgets (raw tk so we can fully control colors/hover)
# ----------------------------------------------------------------------


class _SidebarButton(tk.Frame):
    """Dark sidebar nav button: icon + label, with hover and active states."""

    def __init__(self, master, *, text: str, icon: str, accent: str,
                 on_click: Callable[[], None]):
        super().__init__(master, bg=COLORS["sidebar"], cursor="hand2")
        self._on_click = on_click
        self._accent = accent
        self._active = False

        # Left accent stripe (visible when active)
        self._stripe = tk.Frame(self, bg=COLORS["sidebar"], width=3)
        self._stripe.pack(side="left", fill="y")

        self._row = tk.Frame(self, bg=COLORS["sidebar"], padx=14, pady=10,
                             cursor="hand2")
        self._row.pack(side="left", fill="both", expand=True)

        self._icon = tk.Label(self._row, text=icon,
                              fg=COLORS["sidebar_text"], bg=COLORS["sidebar"],
                              font=("Segoe UI", 14), cursor="hand2")
        self._icon.pack(side="left")
        self._text = tk.Label(self._row, text=text,
                              fg=COLORS["sidebar_text"], bg=COLORS["sidebar"],
                              font=("Segoe UI", 11), cursor="hand2")
        self._text.pack(side="left", padx=(12, 0))

        for w in (self, self._row, self._icon, self._text):
            w.bind("<Enter>", self._on_enter)
            w.bind("<Leave>", self._on_leave)
            w.bind("<Button-1>", self._on_press)

    def _set_bg(self, color: str) -> None:
        for w in (self, self._row, self._icon, self._text):
            try:
                w.configure(bg=color)
            except tk.TclError:
                pass

    def _on_enter(self, _e):
        if not self._active:
            self._set_bg(COLORS["sidebar_hi"])

    def _on_leave(self, _e):
        if not self._active:
            self._set_bg(COLORS["sidebar"])

    def _on_press(self, _e):
        self._on_click()

    def set_active(self, active: bool) -> None:
        self._active = active
        if active:
            self._set_bg(COLORS["sidebar_hi"])
            self._stripe.configure(bg=self._accent)
            self._text.configure(fg="#ffffff", font=("Segoe UI", 11, "bold"))
            self._icon.configure(fg=self._accent, font=("Segoe UI", 14, "bold"))
        else:
            self._set_bg(COLORS["sidebar"])
            self._stripe.configure(bg=COLORS["sidebar"])
            self._text.configure(fg=COLORS["sidebar_text"], font=("Segoe UI", 11))
            self._icon.configure(fg=COLORS["sidebar_text"], font=("Segoe UI", 14))


class _SidebarFooterButton(tk.Frame):
    """Minimal sidebar button used for About / Exit at the bottom."""

    def __init__(self, master, *, text: str, on_click: Callable[[], None]):
        super().__init__(master, bg=COLORS["sidebar"], cursor="hand2")
        self._on_click = on_click

        self._lbl = tk.Label(self, text=text,
                             fg=COLORS["sidebar_mute"], bg=COLORS["sidebar"],
                             font=("Segoe UI", 10), padx=14, pady=8, anchor="w",
                             cursor="hand2")
        self._lbl.pack(fill="x")

        for w in (self, self._lbl):
            w.bind("<Enter>", lambda _e: self._lbl.configure(
                fg="#ffffff", bg=COLORS["sidebar_hi"]))
            w.bind("<Leave>", lambda _e: self._lbl.configure(
                fg=COLORS["sidebar_mute"], bg=COLORS["sidebar"]))
            w.bind("<Button-1>", lambda _e: on_click())


class _MethodCard(tk.Frame):
    """White card with accent stripe, icon, title, description; clickable."""

    def __init__(self, master, *, title: str, description: str, accent: str,
                 icon: str, on_click: Callable[[], None]):
        super().__init__(master, bg=COLORS["panel"],
                         highlightthickness=1,
                         highlightbackground=COLORS["border"],
                         highlightcolor=COLORS["border"],
                         cursor="hand2")
        self._accent = accent
        self._on_click = on_click

        # Top accent strip
        self._stripe = tk.Frame(self, bg=accent, height=3)
        self._stripe.pack(fill="x")

        body = tk.Frame(self, bg=COLORS["panel"], padx=18, pady=16, cursor="hand2")
        body.pack(fill="both", expand=True)

        head = tk.Frame(body, bg=COLORS["panel"], cursor="hand2")
        head.pack(fill="x")

        # Circular-ish icon badge using a colored background label
        self._icon = tk.Label(head, text=icon, fg=accent, bg=COLORS["accent_lo"],
                              font=("Segoe UI", 16, "bold"),
                              padx=10, pady=2, cursor="hand2")
        self._icon.pack(side="left")
        # Tint the icon background based on accent: accent_lo is global indigo -- replace.
        self._icon.configure(bg=self._tint(accent))

        self._title = tk.Label(head, text=title, fg=COLORS["text"],
                               bg=COLORS["panel"],
                               font=("Segoe UI", 12, "bold"), cursor="hand2",
                               anchor="w", justify="left")
        self._title.pack(side="left", padx=(12, 0), fill="x", expand=True)

        # Arrow indicator on the right
        self._arrow = tk.Label(head, text="\u203A", fg=COLORS["text_soft"],
                               bg=COLORS["panel"],
                               font=("Segoe UI", 16, "bold"), cursor="hand2")
        self._arrow.pack(side="right")

        self._desc = tk.Label(body, text=description, fg=COLORS["text_mute"],
                              bg=COLORS["panel"],
                              font=("Segoe UI", 9),
                              wraplength=320, justify="left", anchor="w",
                              cursor="hand2")
        self._desc.pack(fill="x", anchor="w", pady=(10, 0))

        self._all_widgets = (self, body, head, self._title, self._desc, self._arrow)
        for w in self._all_widgets:
            w.bind("<Enter>", self._on_enter)
            w.bind("<Leave>", self._on_leave)
            w.bind("<Button-1>", lambda _e: on_click())

    @staticmethod
    def _tint(hex_color: str) -> str:
        """Return a very light tint of the given color for the icon background."""
        try:
            h = hex_color.lstrip("#")
            r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
            r = int(r + (255 - r) * 0.86)
            g = int(g + (255 - g) * 0.86)
            b = int(b + (255 - b) * 0.86)
            return f"#{r:02x}{g:02x}{b:02x}"
        except Exception:
            return COLORS["accent_lo"]

    def _on_enter(self, _e):
        self.configure(highlightbackground=self._accent,
                       highlightcolor=self._accent)
        self._arrow.configure(fg=self._accent)

    def _on_leave(self, _e):
        self.configure(highlightbackground=COLORS["border"],
                       highlightcolor=COLORS["border"])
        self._arrow.configure(fg=COLORS["text_soft"])


class _Chip(tk.Frame):
    """Small rounded-rect-style status chip with colored bg/fg."""

    KINDS = {
        "idle":    ("chip_idle_bg",  "chip_idle_fg"),
        "running": ("chip_run_bg",   "chip_run_fg"),
        "done":    ("chip_done_bg",  "chip_done_fg"),
        "error":   ("chip_err_bg",   "chip_err_fg"),
    }

    def __init__(self, master, *, text: str = "Ready", kind: str = "idle"):
        super().__init__(master, bg=COLORS["bg"])
        self._lbl = tk.Label(self, text=text,
                             font=("Segoe UI", 9, "bold"),
                             padx=10, pady=3)
        self._lbl.pack()
        self.set(text, kind)

    def set(self, text: str, kind: str = "idle") -> None:
        bg_key, fg_key = self.KINDS.get(kind, self.KINDS["idle"])
        self._lbl.configure(text=text, bg=COLORS[bg_key], fg=COLORS[fg_key])


SCRIPTS = {
    "secant": MethodScript("Secant Method", "Secant Method.py"),
    "fixed_point": MethodScript("Fixed Point Iteration", "Fixed Point.py"),
    "false_position": MethodScript("False Position", "false_position.py"),
    "halley": MethodScript("Halley's Method", "Halley's Method.py"),
    "newton": MethodScript("Newton-Raphson Method", "newton_raphson_method.py"),
    "bisection": MethodScript("Bisection Method", "bisection_method.py"),
    "jacobi": MethodScript("Jacobi Method (Linear Systems)", "jacobi.py"),
    "gauss_seidel": MethodScript("Gauss-Seidel Method (Linear Systems)", "gauss_seidel.py"),
    "newton_system": MethodScript(
        "Newton-Raphson (Nonlinear Systems)",
        "non_linear_systems_(newton_raphson_method).py",
    ),
    "trapezoidal": MethodScript("Composite Trapezoidal Rule", "numerical_integration.py"),
    "simpson": MethodScript("Composite Simpson's Rule", "numerical_integration.py"),
    "gauss_quad": MethodScript("Gaussian Quadrature", "gaussian_quadrature.py"),
    "romberg": MethodScript("Romberg Integration", "romberg_integration.py"),
    "differentiation": MethodScript("Numerical Differentiation", "numerical_differentiation.py"),
}


def _load_module_from_path(path: Path):
    import importlib.util

    module_name = re.sub(r"\W+", "_", path.stem)
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_function(expr: str) -> Callable[[float], float]:
    expr = (expr or "").strip()
    if not expr:
        raise ValueError("Function expression is empty.")

    allowed = {
        "np": np,
        "math": math,
        "abs": abs,
        "pow": pow,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "exp": math.exp,
        "log": math.log,
        "ln": math.log,
        "sqrt": math.sqrt,
        "pi": math.pi,
        "e": math.e,
    }

    def f(x: float) -> float:
        return eval(expr, {"__builtins__": {}}, {**allowed, "x": x})

    return f


def _make_array_function(expr: str) -> Callable:
    """Build a function f(x) that works on both scalars and numpy arrays.

    Used by integration / differentiation routines that pass numpy arrays into f.
    """
    expr_norm = _normalize_user_expr(expr)
    if not expr_norm:
        raise ValueError("Function expression is empty.")

    allowed = {
        "np": np,
        "abs": np.abs,
        "pow": np.power,
        "sin": np.sin,
        "cos": np.cos,
        "tan": np.tan,
        "asin": np.arcsin,
        "acos": np.arccos,
        "atan": np.arctan,
        "sinh": np.sinh,
        "cosh": np.cosh,
        "tanh": np.tanh,
        "exp": np.exp,
        "log": np.log,
        "ln": np.log,
        "log10": np.log10,
        "sqrt": np.sqrt,
        "pi": np.pi,
        "e": np.e,
    }

    code = compile(expr_norm, "<expr>", "eval")

    def f(x):
        return eval(code, {"__builtins__": {}}, {**allowed, "x": x})

    return f


_NEWTON_SYSTEM_FN_CACHE: dict[str, Callable] = {}


def _load_function_def_without_running_script(path: Path, func_name: str) -> Callable:
    cache_key = f"{path.resolve()}::{func_name}"
    cached = _NEWTON_SYSTEM_FN_CACHE.get(cache_key)
    if cached is not None:
        return cached

    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src, filename=str(path))
    fn_node = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            fn_node = node
            break
    if fn_node is None:
        raise RuntimeError(f"Could not find function `{func_name}` in {path.name}")

    mod = ast.Module(body=[fn_node], type_ignores=[])
    ast.fix_missing_locations(mod)
    code = compile(mod, filename=str(path), mode="exec")
    ns: dict[str, object] = {"np": np}
    exec(code, ns, ns)
    fn = ns.get(func_name)
    if not callable(fn):
        raise RuntimeError(f"Failed to load `{func_name}` from {path.name}")

    _NEWTON_SYSTEM_FN_CACHE[cache_key] = fn  # type: ignore[assignment]
    return fn  # type: ignore[return-value]


def _normalize_user_expr(expr: str) -> str:
    expr = (expr or "").strip()
    if not expr:
        return expr
    # Users often type x^2 meaning power; in Python that's XOR.
    expr = expr.replace("^", "**")

    # Insert implicit multiplication so inputs like "2x", "x(y+1)", "2(x+y)", "xy" work.
    num = r"(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"

    # number followed by identifier or '('  -> number*...
    expr = re.sub(rf"({num})\s*(?=[A-Za-z_(])", r"\1*", expr)

    # variable followed by '(' -> x*(...)
    expr = re.sub(r"([xyz])\s*(?=\()", r"\1*", expr)

    # closing ')' followed by '(' or variable -> )*(... or )*x
    expr = re.sub(r"\)\s*(?=[(xyz])", r")*", expr)

    # adjacent variables like "xy" -> x*y (limited to x,y,z)
    expr = re.sub(r"([xyz])\s*(?=[xyz])", r"\1*", expr)

    return expr


def _make_system_expr(expr: str, n: int) -> Callable[[np.ndarray], float]:
    expr = _normalize_user_expr(expr)
    if not expr:
        raise ValueError("Expression is empty.")

    allowed = {
        "np": np,
        "math": math,
        "abs": abs,
        "pow": pow,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "exp": math.exp,
        "log": math.log,
        "ln": math.log,
        "sqrt": math.sqrt,
        "pi": math.pi,
        "e": math.e,
    }

    code = compile(expr, "<expr>", "eval")

    def f(v: np.ndarray) -> float:
        x = float(v[0])
        y = float(v[1])
        local: dict[str, float] = {"x": x, "y": y}
        if n == 3:
            local["z"] = float(v[2])
        return float(eval(code, {"__builtins__": {}}, {**allowed, **local}))

    return f


def _run_and_capture(fn: Callable[[], object]) -> str:
    buf = io.StringIO()
    with redirect_stdout(buf):
        result = fn()
    out = buf.getvalue()
    if result is not None:
        out = (out.rstrip() + "\n\nResult:\n" + repr(result) + "\n")
    return out or "(No output)"


def _run_and_capture_with_result(fn: Callable[[], object]) -> tuple[str, object]:
    buf = io.StringIO()
    with redirect_stdout(buf):
        result = fn()
    out = buf.getvalue()
    if result is not None:
        out = (out.rstrip() + "\n\nResult:\n" + repr(result) + "\n")
    return (out or "(No output)"), result


def _xs_from_history(history) -> Optional[list[float]]:
    try:
        xs = [float(row[1]) for row in history]
        return xs if xs else None
    except Exception:
        return None


def _xs_from_history_index(history, x_index: int) -> Optional[list[float]]:
    try:
        xs = [float(row[x_index]) for row in history]
        return xs if xs else None
    except Exception:
        return None


def _xs_from_false_position_output(output_text: str) -> Optional[list[float]]:
    # Matches the "c" column in lines like: "1          1.234567       -3.456789e-01"
    matches = re.findall(r"^\s*\d+\s+([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s+", output_text, flags=re.MULTILINE)
    xs: list[float] = []
    for m in matches:
        try:
            xs.append(float(m))
        except Exception:
            continue
    return xs if xs else None


def _parse_number_list(text: str) -> list[float]:
    text = (text or "").strip()
    if not text:
        return []
    parts = re.split(r"[,\s]+", text.strip())
    out: list[float] = []
    for p in parts:
        if not p:
            continue
        out.append(float(p))
    return out


def _parse_matrix(text: str, n: int) -> np.ndarray:
    raw = (text or "").strip()
    if not raw:
        raise ValueError("Matrix A is empty.")

    rows = [r.strip() for r in re.split(r"[;\n]+", raw) if r.strip()]
    if len(rows) != n:
        raise ValueError(f"Matrix A must have exactly {n} rows (separated by ';' or newlines).")

    data: list[list[float]] = []
    for r in rows:
        nums = _parse_number_list(r)
        if len(nums) != n:
            raise ValueError(f"Each row of A must have exactly {n} numbers.")
        data.append(nums)
    return np.array(data, dtype=float)


def _parse_vector(text: str, n: int, *, name: str) -> np.ndarray:
    nums = _parse_number_list(text)
    if len(nums) != n:
        raise ValueError(f"{name} must have exactly {n} numbers.")
    return np.array(nums, dtype=float)


def _errors_from_linear_history(history) -> Optional[list[float]]:
    try:
        errs: list[float] = []
        for row in history:
            if len(row) < 3:
                continue
            e = float(row[2])
            if np.isfinite(e):
                errs.append(e)
        return errs if errs else None
    except Exception:
        return None


def _var_traces_from_linear_history(history, n: int) -> list[list[float]]:
    """Return one list of iteration values per variable (n lists)."""
    try:
        traces: list[list[float]] = [[] for _ in range(n)]
        for row in history:
            if len(row) < 2:
                continue
            vec = row[1]
            for i in range(n):
                traces[i].append(float(vec[i]))
        return [t for t in traces if t]
    except Exception:
        return []


def _component_traces_from_newton_output(output_text: str, n: int) -> list[list[float]]:
    """Return one list of iteration values per variable, parsed from printed output."""
    matches = re.findall(r"Iteration\s+\d+:\s*\[([^\]]+)\]", output_text)
    traces: list[list[float]] = [[] for _ in range(n)]
    for m in matches:
        nums = re.findall(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", m)
        if len(nums) < n:
            continue
        for i in range(n):
            try:
                traces[i].append(float(nums[i]))
            except Exception:
                pass
    return [t for t in traces if t]


def _norms_from_system_newton_output(output_text: str) -> Optional[list[float]]:
    # Matches lines like: "Iteration 1: [1. 2.]" or "Iteration 2: [1.234  5.678]"
    matches = re.findall(r"Iteration\s+\d+:\s*\[([^\]]+)\]", output_text)
    norms: list[float] = []
    for m in matches:
        nums = re.findall(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", m)
        if not nums:
            continue
        try:
            v = np.array([float(x) for x in nums], dtype=float)
            norms.append(float(np.linalg.norm(v)))
        except Exception:
            continue
    return norms if norms else None


def _rewrite_system_expr(expr: str) -> str:
    expr = (expr or "").strip()
    if not expr:
        return expr

    # The nonlinear-systems script evaluates expressions with module globals.
    # It has `np` but does not define `log`, `sqrt`, etc (unless user writes `np.log`).
    # We rewrite common unqualified math calls to numpy equivalents.
    replacements = {
        "ln": "np.log",
        "log": "np.log",
        "sqrt": "np.sqrt",
        "sin": "np.sin",
        "cos": "np.cos",
        "tan": "np.tan",
        "exp": "np.exp",
        "abs": "np.abs",
    }

    for name, target in replacements.items():
        # replace "name(" when it's not part of "np.name(" or "math.name(" or another identifier
        expr = re.sub(rf"(?<![\w.]){name}\s*\(", f"{target}(", expr)

    # Also rewrite explicit math.<fn>(...) since the script doesn't import math.
    expr = re.sub(r"(?<![\w.])math\.(log|sqrt|sin|cos|tan|exp|fabs)\s*\(", lambda m: {
        "log": "np.log(",
        "sqrt": "np.sqrt(",
        "sin": "np.sin(",
        "cos": "np.cos(",
        "tan": "np.tan(",
        "exp": "np.exp(",
        "fabs": "np.abs(",
    }[m.group(1)], expr)

    return expr


def _zero_plot_function(x):
    a = np.asarray(x, dtype=float)
    return np.zeros_like(a, dtype=float)


class App(tk.Tk):
    SECTION_ORDER = ["root", "systems", "integration", "differentiation"]

    def __init__(self):
        super().__init__()
        self.title("Numerical Methods Toolkit")
        self.geometry("1040x700")
        self.minsize(960, 640)

        _apply_global_theme(self)
        self.configure(background=COLORS["bg"])

        self._open_figures = []

        # Methods organised by section.  (Title, method-key, command) tuples.
        self._methods_by_section: dict[str, list[tuple[str, str, Callable]]] = {
            "root": [
                ("Bisection Method", "bisection", self.open_bisection),
                ("False Position", "false_position", self.open_false_position),
                ("Fixed Point Iteration", "fixed_point", self.open_fixed_point),
                ("Secant Method", "secant", self.open_secant),
                ("Newton-Raphson", "newton", self.open_newton),
                ("Halley's Method", "halley", self.open_halley),
            ],
            "systems": [
                ("Jacobi Method", "jacobi", self.open_jacobi),
                ("Gauss-Seidel", "gauss_seidel", self.open_gauss_seidel),
                ("Newton-Raphson (Nonlinear)", "newton_system", self.open_newton_system),
            ],
            "integration": [
                ("Composite Trapezoidal", "trapezoidal", self.open_trapezoidal),
                ("Composite Simpson's", "simpson", self.open_simpson),
                ("Gaussian Quadrature", "gauss_quad", self.open_gauss_quadrature),
                ("Romberg Integration", "romberg", self.open_romberg),
            ],
            "differentiation": [
                ("Numerical Differentiation", "differentiation", self.open_differentiation),
            ],
        }

        # ---------- Layout: sidebar (left) + main panel (right) ----------
        body = tk.Frame(self, bg=COLORS["bg"])
        body.pack(fill="both", expand=True)

        sidebar = tk.Frame(body, bg=COLORS["sidebar"], width=240)
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)
        self._build_sidebar(sidebar)

        main = tk.Frame(body, bg=COLORS["bg"])
        main.pack(side="left", fill="both", expand=True)
        self._main = main

        # Page header (icon chip + title + subtitle)
        head_row = tk.Frame(main, bg=COLORS["bg"])
        head_row.pack(fill="x", padx=28, pady=(28, 0))

        self._page_icon_lbl = tk.Label(
            head_row, text="", bg=COLORS["bg"],
            fg=COLORS["text"], font=("Segoe UI", 26, "bold")
        )
        self._page_icon_lbl.pack(side="left")
        head_text = tk.Frame(head_row, bg=COLORS["bg"])
        head_text.pack(side="left", padx=(14, 0), anchor="w")
        self._page_header_var = tk.StringVar()
        self._page_sub_var = tk.StringVar()
        ttk.Label(head_text, textvariable=self._page_header_var,
                  style="PageHeader.TLabel").pack(anchor="w")
        ttk.Label(head_text, textvariable=self._page_sub_var,
                  style="PageSub.TLabel", wraplength=720, justify="left").pack(
            anchor="w", pady=(2, 0)
        )

        # Subtle divider
        tk.Frame(main, bg=COLORS["border"], height=1).pack(
            fill="x", padx=28, pady=(18, 14)
        )

        # Cards container
        self._cards_frame = tk.Frame(main, bg=COLORS["bg"])
        self._cards_frame.pack(fill="both", expand=True, padx=22, pady=(0, 8))

        # Footer tip
        tip = tk.Frame(main, bg=COLORS["bg"])
        tip.pack(fill="x", padx=28, pady=(8, 16))
        ttk.Label(
            tip,
            text="Tip: each method window supports Enter = Run, Esc = Close.",
            style="Status.TLabel",
        ).pack(anchor="w")

        self._select_section(self.SECTION_ORDER[0])

        self.after(0, lambda: _center_on_screen(self))

    # ---------------------------------------------------------------- sidebar

    def _build_sidebar(self, sidebar: tk.Frame) -> None:
        # Brand / logo header
        brand = tk.Frame(sidebar, bg=COLORS["sidebar"])
        brand.pack(fill="x", padx=18, pady=(22, 18))
        tk.Label(brand, text="\u03A3", fg=COLORS["accent"], bg=COLORS["sidebar"],
                 font=("Segoe UI", 22, "bold")).pack(side="left")
        wrap = tk.Frame(brand, bg=COLORS["sidebar"])
        wrap.pack(side="left", padx=(10, 0))
        tk.Label(wrap, text="Numerical", fg="#ffffff", bg=COLORS["sidebar"],
                 font=("Segoe UI", 12, "bold")).pack(anchor="w")
        tk.Label(wrap, text="Methods Toolkit", fg=COLORS["sidebar_mute"],
                 bg=COLORS["sidebar"], font=("Segoe UI", 9)).pack(anchor="w")

        tk.Frame(sidebar, bg=COLORS["sidebar_hi"], height=1).pack(fill="x", padx=14)

        tk.Label(sidebar, text="CATEGORIES", fg=COLORS["text_mute"],
                 bg=COLORS["sidebar"],
                 font=("Segoe UI", 8, "bold")).pack(anchor="w", padx=18, pady=(16, 4))

        self._sidebar_buttons: dict[str, _SidebarButton] = {}
        for key in self.SECTION_ORDER:
            info = CATEGORY_INFO[key]
            btn = _SidebarButton(
                sidebar,
                text=info["label"], icon=info["icon"], accent=info["color"],
                on_click=lambda k=key: self._select_section(k),
            )
            btn.pack(fill="x", padx=8, pady=2)
            self._sidebar_buttons[key] = btn

        # Spacer pushes footer to the bottom
        tk.Frame(sidebar, bg=COLORS["sidebar"]).pack(fill="both", expand=True)

        tk.Frame(sidebar, bg=COLORS["sidebar_hi"], height=1).pack(fill="x", padx=14)
        footer = tk.Frame(sidebar, bg=COLORS["sidebar"])
        footer.pack(fill="x", padx=10, pady=(8, 14))
        _SidebarFooterButton(footer, text="About", on_click=self.show_about).pack(fill="x", pady=2)
        _SidebarFooterButton(footer, text="Exit",  on_click=self.destroy).pack(fill="x", pady=2)

    # ---------------------------------------------------------------- pages

    def _select_section(self, key: str) -> None:
        for k, btn in self._sidebar_buttons.items():
            btn.set_active(k == key)

        info = CATEGORY_INFO[key]
        self._page_icon_lbl.configure(text=info["icon"], fg=info["color"])
        self._page_header_var.set(info["label"])
        self._page_sub_var.set(info["desc"])

        self._render_cards(key)

    def _render_cards(self, key: str) -> None:
        for w in self._cards_frame.winfo_children():
            w.destroy()

        info = CATEGORY_INFO[key]
        accent = info["color"]
        icon = info["icon"]

        cols = 2
        for i, (title, mkey, cmd) in enumerate(self._methods_by_section[key]):
            r, c = divmod(i, cols)
            card = _MethodCard(
                self._cards_frame,
                title=title,
                description=METHOD_DESCRIPTIONS.get(mkey, ""),
                accent=accent,
                icon=icon,
                on_click=cmd,
            )
            card.grid(row=r, column=c, sticky="nsew", padx=8, pady=8)

        for c in range(cols):
            self._cards_frame.columnconfigure(c, weight=1)
        # Make all rows have equal vertical space.
        rows = (len(self._methods_by_section[key]) + cols - 1) // cols
        for r in range(rows):
            self._cards_frame.rowconfigure(r, weight=1)

    def show_about(self):
        about = tk.Toplevel(self)
        about.title("About")
        about.resizable(False, False)
        about.transient(self)
        about.grab_set()
        _apply_global_theme(about)
        about.configure(background=COLORS["bg"])

        # Top accent strip
        tk.Frame(about, bg=COLORS["accent"], height=4).pack(fill="x")

        frame = ttk.Frame(about, padding=24, style="App.TFrame")
        frame.pack(fill="both", expand=True)

        # Brand row
        brand = tk.Frame(frame, bg=COLORS["bg"])
        brand.pack(anchor="center")
        tk.Label(brand, text="\u03A3", fg=COLORS["accent"], bg=COLORS["bg"],
                 font=("Segoe UI", 28, "bold")).pack(side="left")
        wrap = tk.Frame(brand, bg=COLORS["bg"])
        wrap.pack(side="left", padx=(12, 0))
        ttk.Label(wrap, text="Numerical Methods Toolkit",
                  style="Header.TLabel").pack(anchor="w")
        ttk.Label(wrap, text="Course project \u2014 Numerical Methods",
                  style="Sub.TLabel").pack(anchor="w", pady=(2, 0))

        tk.Frame(frame, bg=COLORS["border"], height=1).pack(fill="x", pady=(16, 14))

        team = (
            "Arsany Ehab Alfy \u2014 ID: 24100390\n"
            "Omar Mustafa Elnainay \u2014 ID: 24100142\n"
            "Abdelrahman Ahmed Ibrahim \u2014 ID: 24100417\n"
            "Mohamed Wael Abdelmaqsoud \u2014 ID: 24100444"
        )
        ttk.Label(frame, text="Team", style="Section.TLabel").pack(anchor="center")
        ttk.Label(frame, text=team, justify="center").pack(anchor="center", pady=(4, 14))

        ttk.Label(frame, text="Instructor", style="Section.TLabel").pack(anchor="center")
        ttk.Label(frame, text="Dr. Abdellatif Mahmoud").pack(anchor="center", pady=(4, 18))

        ttk.Button(frame, text="Close", style="Accent.TButton",
                   command=about.destroy).pack()

        _center_on_screen(about, self)

    def _open_graph_window(self, f, xs: list[float], title: str):
        try:
            from graph import plot_iteration_trace
            import matplotlib.pyplot as plt
        except Exception as e:
            messagebox.showerror(
                "Graph not available",
                "Could not open the graph window.\n\nMake sure matplotlib is installed.\n\nError:\n"
                + str(e),
            )
            return

        try:
            fig, ax, ani = plot_iteration_trace(f, xs, title=title, frames=200, interval=1200, repeat=True, root_marker=True)
            self._open_figures.append((fig, ax, ani))
            plt.show(block=False)
        except Exception as e:
            messagebox.showerror("Graph failed", str(e))

    def _open_trajectory_window(self, traces: list[list[float]], title: str):
        """Open an animated split per-variable iteration window (one subplot per variable)."""
        try:
            import matplotlib.pyplot as plt
            from graph import plot_variable_traces

            if not traces:
                return
            fig, axes, ani = plot_variable_traces(
                traces, title=title, frames=200, interval=1200, repeat=True
            )
            self._open_figures.append((fig, axes, ani))
            plt.show(block=False)
        except Exception as e:
            messagebox.showerror("Graph failed", str(e))

    def _open_integration_graph(self, *, f, a, b, n, integral_value, method, title,
                                x_data=None, y_data=None):
        try:
            import matplotlib.pyplot as plt
            from graph import plot_integration_method

            fig, ax, ani = plot_integration_method(
                f, a, b, n, integral_value, method=method, title=title,
                x_data=x_data, y_data=y_data,
            )
            self._open_figures.append((fig, ax, ani))
            plt.show(block=False)
        except Exception as e:
            messagebox.showerror("Graph failed", str(e))

    def _open_gauss_graph(self, *, f, a, b, points, integral_value, title):
        try:
            import matplotlib.pyplot as plt
            from graph import plot_gauss_quadrature

            fig, ax, ani = plot_gauss_quadrature(f, a, b, points, integral_value, title=title)
            self._open_figures.append((fig, ax, ani))
            plt.show(block=False)
        except Exception as e:
            messagebox.showerror("Graph failed", str(e))

    def _open_romberg_graph(self, *, R, title):
        try:
            import matplotlib.pyplot as plt
            from graph import plot_romberg_convergence

            fig, ax, ani = plot_romberg_convergence(R, title=title)
            self._open_figures.append((fig, ax, ani))
            plt.show(block=False)
        except Exception as e:
            messagebox.showerror("Graph failed", str(e))

    def _open_diff_graph(self, *, x_data, y_data, x_targets, derivs, f=None, title):
        try:
            import matplotlib.pyplot as plt
            from graph import plot_numerical_derivative

            fig, ax, ani = plot_numerical_derivative(
                x_data, y_data, x_targets, derivs, f=f, title=title
            )
            self._open_figures.append((fig, ax, ani))
            plt.show(block=False)
        except Exception as e:
            messagebox.showerror("Graph failed", str(e))

    def _ensure_exists(self, script: MethodScript) -> bool:
        if script.path.exists():
            return True
        messagebox.showerror(
            "File not found",
            f"Could not find:\n{script.path}\n\nCheck the filename in the folder.",
        )
        return False

    def open_secant(self):
        if not self._ensure_exists(SCRIPTS["secant"]):
            return

        mod = _load_module_from_path(SCRIPTS["secant"].path)
        secant_method = getattr(mod, "secant_method", None)
        if secant_method is None:
            messagebox.showerror("Missing function", "Could not find `secant_method` in Secant Method.py")
            return

        win = MethodWindow(
            self,
            title="Secant Method",
            subtitle=METHOD_DESCRIPTIONS["secant"],
            category=METHOD_CATEGORY["secant"],
            fields=[
                Field("f(x)", "x**3 - x - 2", help="Function whose root we seek."),
                Field("x0", "1", help="First initial guess."),
                Field("x1", "2", help="Second initial guess (different from x0)."),
                Field("tol", "1e-6", help="Stopping tolerance on |x_{n+1}-x_n|."),
                Field("max_iter", "100", kind="spin", spin_from=1, spin_to=10000, spin_step=1),
                Field("print_table (0/1)", "1", kind="check"),
            ],
            on_run=lambda values, out: self._run_secant(secant_method, values, out),
        )
        win.show()

    def _run_secant(self, secant_method, values: dict, out_widget: "OutputBox"):
        try:
            f = _make_function(values["f(x)"])
            x0 = float(values["x0"])
            x1 = float(values["x1"])
            tol = float(values["tol"])
            max_iter = int(float(values["max_iter"]))
            print_table = bool(int(float(values["print_table (0/1)"])))
        except Exception as e:
            messagebox.showerror("Invalid input", str(e))
            return

        def runner():
            return secant_method(f, x0, x1, tol=tol, max_iter=max_iter, print_table=print_table)

        out_text, result = _run_and_capture_with_result(runner)
        out_widget.set_text(out_text)
        xs = None
        try:
            if isinstance(result, tuple) and len(result) >= 4:
                xs = _xs_from_history(result[3])
        except Exception:
            xs = None
        if xs:
            self._open_graph_window(f, xs, "Secant Method")

    def open_fixed_point(self):
        if not self._ensure_exists(SCRIPTS["fixed_point"]):
            return

        mod = _load_module_from_path(SCRIPTS["fixed_point"].path)
        fixed_point_iteration = getattr(mod, "fixed_point_iteration", None)
        if fixed_point_iteration is None:
            messagebox.showerror("Missing function", "Could not find `fixed_point_iteration` in Fixed Piont .py")
            return

        win = MethodWindow(
            self,
            title="Fixed Point Iteration",
            subtitle=METHOD_DESCRIPTIONS["fixed_point"],
            category=METHOD_CATEGORY["fixed_point"],
            fields=[
                Field("f(x)", "x**3 - x - 2", help="Original equation rearranged so that g(x)=x is equivalent to f(x)=0."),
                Field("g(x)", "(x + 2)**(1/3)", help="Iteration map: x_{n+1} = g(x_n)."),
                Field("x0", "1.5"),
                Field("tol", "1e-6"),
                Field("max_iter", "100", kind="spin", spin_from=1, spin_to=10000),
                Field("print_table (0/1)", "1", kind="check"),
            ],
            on_run=lambda values, out: self._run_fixed_point(fixed_point_iteration, values, out),
        )
        win.show()

    def _run_fixed_point(self, fixed_point_iteration, values: dict, out_widget: "OutputBox"):
        try:
            f = _make_function(values["f(x)"])
            g = _make_function(values["g(x)"])
            x0 = float(values["x0"])
            tol = float(values["tol"])
            max_iter = int(float(values["max_iter"]))
            print_table = bool(int(float(values["print_table (0/1)"])))
        except Exception as e:
            messagebox.showerror("Invalid input", str(e))
            return

        def runner():
            return fixed_point_iteration(f, g, x0, tol=tol, max_iter=max_iter, print_table=print_table)

        out_text, result = _run_and_capture_with_result(runner)
        out_widget.set_text(out_text)
        xs = None
        try:
            if isinstance(result, tuple) and len(result) >= 4:
                xs = _xs_from_history(result[3])
        except Exception:
            xs = None
        if xs:
            self._open_graph_window(f, xs, "Fixed Point Iteration")

    def open_false_position(self):
        if not self._ensure_exists(SCRIPTS["false_position"]):
            return

        mod = _load_module_from_path(SCRIPTS["false_position"].path)
        false_position = getattr(mod, "false_position", None)
        if false_position is None:
            messagebox.showerror("Missing function", "Could not find `false_position` in false_position.py")
            return

        win = MethodWindow(
            self,
            title="False Position",
            subtitle=METHOD_DESCRIPTIONS["false_position"],
            category=METHOD_CATEGORY["false_position"],
            fields=[
                Field("f(x)", "x**3 - x - 2"),
                Field("a", "1", help="Left bracket; f(a) and f(b) must have opposite signs."),
                Field("b", "2", help="Right bracket."),
                Field("tol", "1e-6"),
                Field("max_iter", "100", kind="spin", spin_from=1, spin_to=10000),
            ],
            on_run=lambda values, out: self._run_false_position(false_position, values, out),
        )
        win.show()

    def _run_false_position(self, false_position, values: dict, out_widget: "OutputBox"):
        try:
            f = _make_function(values["f(x)"])
            a = float(values["a"])
            b = float(values["b"])
            tol = float(values["tol"])
            max_iter = int(float(values["max_iter"]))
        except Exception as e:
            messagebox.showerror("Invalid input", str(e))
            return

        def runner():
            return false_position(f, a, b, tol=tol, max_iter=max_iter)

        out_text, result = _run_and_capture_with_result(runner)
        out_widget.set_text(out_text)
        xs = _xs_from_false_position_output(out_text)
        if xs:
            self._open_graph_window(f, xs, "False Position")

    def launch_halley_script(self):
        script = SCRIPTS["halley"]
        if not self._ensure_exists(script):
            return

        try:
            flags = 0
            if sys.platform.startswith("win"):
                flags = subprocess.CREATE_NEW_CONSOLE  # type: ignore[attr-defined]
            subprocess.Popen([sys.executable, str(script.path)], creationflags=flags)
        except Exception as e:
            messagebox.showerror("Launch failed", str(e))

    def open_halley(self):
        if not self._ensure_exists(SCRIPTS["halley"]):
            return

        mod = _load_module_from_path(SCRIPTS["halley"].path)
        halley_method = getattr(mod, "halley_method", None)
        if halley_method is None:
            messagebox.showerror("Missing function", "Could not find `halley_method` in Halley's Method.py")
            return

        win = MethodWindow(
            self,
            title="Halley's Method",
            subtitle=METHOD_DESCRIPTIONS["halley"],
            category=METHOD_CATEGORY["halley"],
            fields=[
                Field("f(x)", "x**3 - x - 2"),
                Field("f'(x)", "3*x**2 - 1", help="First derivative of f."),
                Field("f''(x)", "6*x", help="Second derivative of f."),
                Field("x0", "1.5"),
                Field("tol", "1e-6"),
                Field("max_iter", "100", kind="spin", spin_from=1, spin_to=10000),
                Field("print_table (0/1)", "1", kind="check"),
            ],
            on_run=lambda values, out: self._run_halley(halley_method, values, out),
        )
        win.show()

    def _run_halley(self, halley_method, values: dict, out_widget: "OutputBox"):
        try:
            f = _make_function(values["f(x)"])
            df = _make_function(values["f'(x)"])
            ddf = _make_function(values["f''(x)"])
            x0 = float(values["x0"])
            tol = float(values["tol"])
            max_iter = int(float(values["max_iter"]))
            print_table = bool(int(float(values["print_table (0/1)"])))
        except Exception as e:
            messagebox.showerror("Invalid input", str(e))
            return

        def runner():
            return halley_method(f, df, ddf, x0, tol=tol, max_iter=max_iter, print_table=print_table)

        out_text, result = _run_and_capture_with_result(runner)
        out_widget.set_text(out_text)
        xs = None
        if isinstance(result, (list, tuple)) and result:
            try:
                xs = [float(v) for v in result]
            except Exception:
                xs = None
        if xs:
            self._open_graph_window(f, xs, "Halley's Method")

    def open_newton(self):
        if not self._ensure_exists(SCRIPTS["newton"]):
            return

        mod = _load_module_from_path(SCRIPTS["newton"].path)
        newton_method = getattr(mod, "newton_method", None)
        if newton_method is None:
            messagebox.showerror("Missing function", "Could not find `newton_method` in newton_raphson_method.py")
            return

        win = MethodWindow(
            self,
            title="Newton-Raphson Method",
            subtitle=METHOD_DESCRIPTIONS["newton"],
            category=METHOD_CATEGORY["newton"],
            fields=[
                Field("f(x)", "x**3 - x - 2"),
                Field("f'(x)", "3*x**2 - 1"),
                Field("x0", "1.5"),
                Field("tol", "1e-6"),
                Field("max_iter", "100", kind="spin", spin_from=1, spin_to=10000),
                Field("print_table (0/1)", "1", kind="check"),
            ],
            on_run=lambda values, out: self._run_newton(newton_method, values, out),
        )
        win.show()

    def _run_newton(self, newton_method, values: dict, out_widget: "OutputBox"):
        try:
            f = _make_function(values["f(x)"])
            df = _make_function(values["f'(x)"])
            x0 = float(values["x0"])
            tol = float(values["tol"])
            max_iter = int(float(values["max_iter"]))
            print_table = bool(int(float(values["print_table (0/1)"])))
        except Exception as e:
            messagebox.showerror("Invalid input", str(e))
            return

        def runner():
            return newton_method(f, df, x0, tol=tol, max_iter=max_iter, print_table=print_table)

        out_text, result = _run_and_capture_with_result(runner)
        out_widget.set_text(out_text)
        xs = None
        try:
            if isinstance(result, tuple) and len(result) >= 4:
                xs = _xs_from_history(result[3])
        except Exception:
            xs = None
        if xs:
            self._open_graph_window(f, xs, "Newton-Raphson Method")

    def open_bisection(self):
        if not self._ensure_exists(SCRIPTS["bisection"]):
            return

        mod = _load_module_from_path(SCRIPTS["bisection"].path)
        bisection_method = getattr(mod, "bisection_method", None)
        if bisection_method is None:
            messagebox.showerror("Missing function", "Could not find `bisection_method` in bisection_method.py")
            return

        win = MethodWindow(
            self,
            title="Bisection Method",
            subtitle=METHOD_DESCRIPTIONS["bisection"],
            category=METHOD_CATEGORY["bisection"],
            fields=[
                Field("f(x)", "x**3 - x - 2"),
                Field("a", "1", help="f(a) and f(b) must have opposite signs."),
                Field("b", "2"),
                Field("tol", "1e-6"),
                Field("max_iter", "100", kind="spin", spin_from=1, spin_to=10000),
                Field("print_table (0/1)", "1", kind="check"),
            ],
            on_run=lambda values, out: self._run_bisection(bisection_method, values, out),
        )
        win.show()

    def _run_bisection(self, bisection_method, values: dict, out_widget: "OutputBox"):
        try:
            f = _make_function(values["f(x)"])
            a = float(values["a"])
            b = float(values["b"])
            tol = float(values["tol"])
            max_iter = int(float(values["max_iter"]))
            print_table = bool(int(float(values["print_table (0/1)"])))
        except Exception as e:
            messagebox.showerror("Invalid input", str(e))
            return

        def runner():
            return bisection_method(f, a, b, tol=tol, max_iter=max_iter, print_table=print_table)

        out_text, result = _run_and_capture_with_result(runner)
        out_widget.set_text(out_text)
        xs = None
        try:
            if isinstance(result, tuple) and len(result) >= 4:
                xs = _xs_from_history_index(result[3], 3)  # c column
        except Exception:
            xs = None
        if xs:
            self._open_graph_window(f, xs, "Bisection Method")

    def open_jacobi(self):
        if not self._ensure_exists(SCRIPTS["jacobi"]):
            return

        mod = _load_module_from_path(SCRIPTS["jacobi"].path)
        jacobi_method = getattr(mod, "jacobi_method", None)
        if jacobi_method is None:
            messagebox.showerror("Missing function", "Could not find `jacobi_method` in jacobi.py")
            return

        win = MethodWindow(
            self,
            title="Jacobi Method (Linear Systems)",
            subtitle=METHOD_DESCRIPTIONS["jacobi"],
            category=METHOD_CATEGORY["jacobi"],
            fields=[
                Field("n (2 or 3)", "3", kind="combo", choices=["2", "3"]),
                Field("A rows (; or newline)", "10 2 1; 1 5 1; 2 3 10",
                      help="Diagonally dominant systems converge well. Separate rows with ';' or newlines."),
                Field("b", "14 10 14", help="Right-hand side vector."),
                Field("x0 (optional)", "0 0 0", help="Initial guess. Leave blank for zero."),
                Field("tol", "1e-6"),
                Field("max_iter", "100", kind="spin", spin_from=1, spin_to=10000),
                Field("print_table (0/1)", "1", kind="check"),
            ],
            on_run=lambda values, out: self._run_jacobi(jacobi_method, values, out),
        )
        win.show()

    def _run_jacobi(self, jacobi_method, values: dict, out_widget: "OutputBox"):
        try:
            n = int(float(values["n (2 or 3)"]))
            if n not in (2, 3):
                raise ValueError("n must be 2 or 3.")
            A = _parse_matrix(values["A rows (; or newline)"], n)
            b = _parse_vector(values["b"], n, name="b")
            x0_text = (values["x0 (optional)"] or "").strip()
            x0 = _parse_vector(x0_text, n, name="x0") if x0_text else None
            tol = float(values["tol"])
            max_iter = int(float(values["max_iter"]))
            print_table = bool(int(float(values["print_table (0/1)"])))
        except Exception as e:
            messagebox.showerror("Invalid input", str(e))
            return

        def runner():
            return jacobi_method(A, b, x0=x0, tol=tol, max_iter=max_iter, print_table=print_table)

        out_text, result = _run_and_capture_with_result(runner)
        out_widget.set_text(out_text)

        try:
            if isinstance(result, tuple) and len(result) >= 4:
                traces = _var_traces_from_linear_history(result[3], n)
                if traces:
                    self._open_trajectory_window(traces, "Jacobi Method - Phase Trajectory")
        except Exception:
            pass

    def open_gauss_seidel(self):
        if not self._ensure_exists(SCRIPTS["gauss_seidel"]):
            return

        mod = _load_module_from_path(SCRIPTS["gauss_seidel"].path)
        gauss_seidel_method = getattr(mod, "gauss_seidel_method", None)
        if gauss_seidel_method is None:
            messagebox.showerror("Missing function", "Could not find `gauss_seidel_method` in gauss_seidel.py")
            return

        win = MethodWindow(
            self,
            title="Gauss-Seidel Method (Linear Systems)",
            subtitle=METHOD_DESCRIPTIONS["gauss_seidel"],
            category=METHOD_CATEGORY["gauss_seidel"],
            fields=[
                Field("n (2 or 3)", "3", kind="combo", choices=["2", "3"]),
                Field("A rows (; or newline)", "10 2 1; 1 5 1; 2 3 10"),
                Field("b", "14 10 14"),
                Field("x0 (optional)", "0 0 0"),
                Field("tol", "1e-6"),
                Field("max_iter", "100", kind="spin", spin_from=1, spin_to=10000),
                Field("print_table (0/1)", "1", kind="check"),
            ],
            on_run=lambda values, out: self._run_gauss_seidel(gauss_seidel_method, values, out),
        )
        win.show()

    def _run_gauss_seidel(self, gauss_seidel_method, values: dict, out_widget: "OutputBox"):
        try:
            n = int(float(values["n (2 or 3)"]))
            if n not in (2, 3):
                raise ValueError("n must be 2 or 3.")
            A = _parse_matrix(values["A rows (; or newline)"], n)
            b = _parse_vector(values["b"], n, name="b")
            x0_text = (values["x0 (optional)"] or "").strip()
            x0 = _parse_vector(x0_text, n, name="x0") if x0_text else None
            tol = float(values["tol"])
            max_iter = int(float(values["max_iter"]))
            print_table = bool(int(float(values["print_table (0/1)"])))
        except Exception as e:
            messagebox.showerror("Invalid input", str(e))
            return

        def runner():
            return gauss_seidel_method(A, b, x0=x0, tol=tol, max_iter=max_iter, print_table=print_table)

        out_text, result = _run_and_capture_with_result(runner)
        out_widget.set_text(out_text)

        try:
            if isinstance(result, tuple) and len(result) >= 4:
                traces = _var_traces_from_linear_history(result[3], n)
                if traces:
                    self._open_trajectory_window(traces, "Gauss-Seidel Method - Phase Trajectory")
        except Exception:
            pass

    def open_newton_system(self):
        if not self._ensure_exists(SCRIPTS["newton_system"]):
            return

        win = MethodWindow(
            self,
            title="Newton-Raphson (Nonlinear Systems)",
            subtitle=METHOD_DESCRIPTIONS["newton_system"],
            category=METHOD_CATEGORY["newton_system"],
            fields=[
                Field("n (2 or 3)", "3", kind="combo", choices=["2", "3"]),
                Field("initial (x0 y0 [z0])", "0.5 0.5 0.8"),
                Field("tol", "1e-6"),
                Field("max_iter", "50", kind="spin", spin_from=1, spin_to=10000),
                Field("f1(x,y,z)", "x^2+y^2+z^2-1"),
                Field("f2(x,y,z)", "x+y-z"),
                Field("f3(x,y,z) (only if n=3)", "x-y"),
                Field("df1/dx", "2x"),
                Field("df1/dy", "2y"),
                Field("df1/dz (only if n=3)", "2z"),
                Field("df2/dx", "1"),
                Field("df2/dy", "1"),
                Field("df2/dz (only if n=3)", "-1"),
                Field("df3/dx (only if n=3)", "1"),
                Field("df3/dy (only if n=3)", "-1"),
                Field("df3/dz (only if n=3)", "0"),
            ],
            on_run=lambda values, out: self._run_newton_system(values, out),
        )
        win.show()

    def _run_newton_system(self, values: dict, out_widget: "OutputBox"):
        script = SCRIPTS["newton_system"]
        try:
            n = int(float(values["n (2 or 3)"]))
            if n not in (2, 3):
                raise ValueError("n must be 2 or 3.")

            init = _parse_number_list(values["initial (x0 y0 [z0])"])
            if len(init) != n:
                raise ValueError(f"Initial values must have exactly {n} numbers.")

            tol = float(values["tol"])
            max_iter = int(float(values["max_iter"]))

            f1 = (values["f1(x,y,z)"] or "").strip()
            f2 = (values["f2(x,y,z)"] or "").strip()
            f3 = (values["f3(x,y,z) (only if n=3)"] or "").strip()
            if not f1 or not f2 or (n == 3 and not f3):
                raise ValueError("Please enter all required equations for the chosen system size.")

            j11 = (values["df1/dx"] or "").strip()
            j12 = (values["df1/dy"] or "").strip()
            j13 = (values["df1/dz (only if n=3)"] or "").strip()
            j21 = (values["df2/dx"] or "").strip()
            j22 = (values["df2/dy"] or "").strip()
            j23 = (values["df2/dz (only if n=3)"] or "").strip()
            j31 = (values["df3/dx (only if n=3)"] or "").strip()
            j32 = (values["df3/dy (only if n=3)"] or "").strip()
            j33 = (values["df3/dz (only if n=3)"] or "").strip()

            if not (j11 and j12 and j21 and j22):
                raise ValueError("Please enter the Jacobian entries for f1 and f2.")
            if n == 3 and not (j13 and j23 and j31 and j32 and j33):
                raise ValueError("For n=3, please enter all Jacobian entries including z terms and f3 row.")
        except Exception as e:
            messagebox.showerror("Invalid input", str(e))
            return

        try:
            newton_raphson_system = _load_function_def_without_running_script(
                script.path, "newton_raphson_system"
            )
        except Exception as e:
            messagebox.showerror("Load failed", str(e))
            return

        try:
            f1_fn = _make_system_expr(f1, n)
            f2_fn = _make_system_expr(f2, n)
            f3_fn = _make_system_expr(f3, n) if n == 3 else None

            j11_fn = _make_system_expr(j11, n)
            j12_fn = _make_system_expr(j12, n)
            j13_fn = _make_system_expr(j13, n) if n == 3 else None
            j21_fn = _make_system_expr(j21, n)
            j22_fn = _make_system_expr(j22, n)
            j23_fn = _make_system_expr(j23, n) if n == 3 else None
            j31_fn = _make_system_expr(j31, n) if n == 3 else None
            j32_fn = _make_system_expr(j32, n) if n == 3 else None
            j33_fn = _make_system_expr(j33, n) if n == 3 else None
        except Exception as e:
            messagebox.showerror("Invalid expression", str(e))
            return

        def F(v):
            if n == 2:
                return [f1_fn(v), f2_fn(v)]
            return [f1_fn(v), f2_fn(v), f3_fn(v)]  # type: ignore[misc]

        def J(v):
            if n == 2:
                return [
                    [j11_fn(v), j12_fn(v)],
                    [j21_fn(v), j22_fn(v)],
                ]
            return [
                [j11_fn(v), j12_fn(v), j13_fn(v)],  # type: ignore[misc]
                [j21_fn(v), j22_fn(v), j23_fn(v)],  # type: ignore[misc]
                [j31_fn(v), j32_fn(v), j33_fn(v)],  # type: ignore[misc]
            ]

        def runner():
            return newton_raphson_system(F, J, init, tol=tol, max_iter=max_iter)

        out_text, result = _run_and_capture_with_result(runner)
        out_widget.set_text(out_text)

        traces = _component_traces_from_newton_output(out_text or "", n)
        if traces:
            self._open_trajectory_window(traces, "Newton-Raphson System - Iteration Traces")

    def open_trapezoidal(self):
        if not self._ensure_exists(SCRIPTS["trapezoidal"]):
            return
        mod = _load_module_from_path(SCRIPTS["trapezoidal"].path)
        composite_trapezoidal = getattr(mod, "composite_trapezoidal", None)
        if composite_trapezoidal is None:
            messagebox.showerror("Missing function", "Could not find `composite_trapezoidal` in numerical_integration.py")
            return

        win = MethodWindow(
            self,
            title="Composite Trapezoidal Rule",
            subtitle=METHOD_DESCRIPTIONS["trapezoidal"],
            category=METHOD_CATEGORY["trapezoidal"],
            fields=[
                Field("Mode (function/table)", "function", kind="combo",
                      choices=["function", "table"],
                      help="Choose 'function' to integrate f(x) on [a,b] with n subintervals, or 'table' to use tabulated data."),
                Field("f(x)  [function mode]", "sin(x)"),
                Field("a  [function mode]", "0"),
                Field("b  [function mode]", "pi", help="Supports expressions like pi, 2*pi/3."),
                Field("n subintervals  [function mode]", "10",
                      kind="spin", spin_from=1, spin_to=10000),
                Field("x_data  [table mode, space sep]", "",
                      help="Equally spaced x values, e.g. '0 0.25 0.5 0.75 1'."),
                Field("y_data  [table mode, space sep]", "",
                      help="Same length as x_data."),
                Field("print_table (0/1)", "1", kind="check"),
            ],
            on_run=lambda values, out: self._run_trapezoidal(composite_trapezoidal, values, out),
        )
        win.show()

    def _run_trapezoidal(self, composite_trapezoidal, values: dict, out_widget: "OutputBox"):
        try:
            mode = (values["Mode (function/table)"] or "function").strip().lower()
            print_table = bool(int(float(values["print_table (0/1)"])))

            if mode.startswith("t"):
                xs = _parse_number_list(values["x_data  [table mode, space sep]"])
                ys = _parse_number_list(values["y_data  [table mode, space sep]"])
                if len(xs) < 2 or len(xs) != len(ys):
                    raise ValueError("x_data and y_data must be non-empty and the same length.")
                x_arr = np.array(xs, dtype=float)
                y_arr = np.array(ys, dtype=float)

                def runner_t():
                    return composite_trapezoidal(x_data=x_arr, y_data=y_arr, print_table=print_table)

                out_text, integral = _run_and_capture_with_result(runner_t)
                out_widget.set_text(out_text)
                if integral is not None:
                    self._open_integration_graph(
                        f=None, a=float(x_arr[0]), b=float(x_arr[-1]),
                        n=len(x_arr) - 1, integral_value=float(integral),
                        method="trapezoidal", title="Composite Trapezoidal (table)",
                        x_data=x_arr, y_data=y_arr,
                    )
                return

            f = _make_array_function(values["f(x)  [function mode]"])
            a = float(_make_array_function(values["a  [function mode]"])(0.0))
            b = float(_make_array_function(values["b  [function mode]"])(0.0))
            n = int(float(values["n subintervals  [function mode]"]))
            if n < 1:
                raise ValueError("n must be >= 1.")
        except Exception as e:
            messagebox.showerror("Invalid input", str(e))
            return

        def runner():
            return composite_trapezoidal(f=f, a=a, b=b, n=n, print_table=print_table)

        out_text, integral = _run_and_capture_with_result(runner)
        out_widget.set_text(out_text)
        if integral is not None:
            self._open_integration_graph(
                f=f, a=a, b=b, n=n, integral_value=float(integral),
                method="trapezoidal", title="Composite Trapezoidal Rule",
            )

    def open_simpson(self):
        if not self._ensure_exists(SCRIPTS["simpson"]):
            return
        mod = _load_module_from_path(SCRIPTS["simpson"].path)
        composite_simpson = getattr(mod, "composite_simpson", None)
        if composite_simpson is None:
            messagebox.showerror("Missing function", "Could not find `composite_simpson` in numerical_integration.py")
            return

        win = MethodWindow(
            self,
            title="Composite Simpson's Rule",
            subtitle=METHOD_DESCRIPTIONS["simpson"],
            category=METHOD_CATEGORY["simpson"],
            fields=[
                Field("Mode (function/table)", "function", kind="combo",
                      choices=["function", "table"]),
                Field("f(x)  [function mode]", "sin(x)"),
                Field("a  [function mode]", "0"),
                Field("b  [function mode]", "pi"),
                Field("n subintervals (even)  [function mode]", "10",
                      kind="spin", spin_from=2, spin_to=10000, spin_step=2,
                      help="Must be a positive even integer."),
                Field("x_data  [table mode, space sep]", "",
                      help="Need an odd number of points (even number of intervals)."),
                Field("y_data  [table mode, space sep]", ""),
                Field("print_table (0/1)", "1", kind="check"),
            ],
            on_run=lambda values, out: self._run_simpson(composite_simpson, values, out),
        )
        win.show()

    def _run_simpson(self, composite_simpson, values: dict, out_widget: "OutputBox"):
        try:
            mode = (values["Mode (function/table)"] or "function").strip().lower()
            print_table = bool(int(float(values["print_table (0/1)"])))

            if mode.startswith("t"):
                xs = _parse_number_list(values["x_data  [table mode, space sep]"])
                ys = _parse_number_list(values["y_data  [table mode, space sep]"])
                if len(xs) < 3 or len(xs) != len(ys):
                    raise ValueError("Need at least 3 equally-spaced points and equal-length x/y lists.")
                if (len(xs) - 1) % 2 != 0:
                    raise ValueError("Simpson needs an even number of intervals (odd number of points).")
                x_arr = np.array(xs, dtype=float)
                y_arr = np.array(ys, dtype=float)

                def runner_t():
                    return composite_simpson(x_data=x_arr, y_data=y_arr, print_table=print_table)

                out_text, integral = _run_and_capture_with_result(runner_t)
                out_widget.set_text(out_text)
                if integral is not None:
                    self._open_integration_graph(
                        f=None, a=float(x_arr[0]), b=float(x_arr[-1]),
                        n=len(x_arr) - 1, integral_value=float(integral),
                        method="simpson", title="Composite Simpson (table)",
                        x_data=x_arr, y_data=y_arr,
                    )
                return

            f = _make_array_function(values["f(x)  [function mode]"])
            a = float(_make_array_function(values["a  [function mode]"])(0.0))
            b = float(_make_array_function(values["b  [function mode]"])(0.0))
            n = int(float(values["n subintervals (even)  [function mode]"]))
            if n < 2 or n % 2 != 0:
                raise ValueError("n must be a positive even integer for Simpson's rule.")
        except Exception as e:
            messagebox.showerror("Invalid input", str(e))
            return

        def runner():
            return composite_simpson(f=f, a=a, b=b, n=n, print_table=print_table)

        out_text, integral = _run_and_capture_with_result(runner)
        out_widget.set_text(out_text)
        if integral is not None:
            self._open_integration_graph(
                f=f, a=a, b=b, n=n, integral_value=float(integral),
                method="simpson", title="Composite Simpson's Rule",
            )

    def open_gauss_quadrature(self):
        if not self._ensure_exists(SCRIPTS["gauss_quad"]):
            return
        mod = _load_module_from_path(SCRIPTS["gauss_quad"].path)
        gaussian_quadrature = getattr(mod, "gaussian_quadrature", None)
        if gaussian_quadrature is None:
            messagebox.showerror("Missing function", "Could not find `gaussian_quadrature` in gaussian_quadrature.py")
            return

        win = MethodWindow(
            self,
            title="Gaussian Quadrature",
            subtitle=METHOD_DESCRIPTIONS["gauss_quad"],
            category=METHOD_CATEGORY["gauss_quad"],
            fields=[
                Field("f(x)", "exp(x)",
                      help="Default: \u222b\u2080\u00b2 e^x dx \u2248 6.389056 ; 3-point Gauss \u2248 6.388978."),
                Field("a", "0"),
                Field("b", "2"),
                Field("points (1/2/3)", "3", kind="combo", choices=["1", "2", "3"]),
                Field("print_table (0/1)", "1", kind="check"),
            ],
            on_run=lambda values, out: self._run_gauss_quadrature(gaussian_quadrature, values, out),
        )
        win.show()

    def _run_gauss_quadrature(self, gaussian_quadrature, values: dict, out_widget: "OutputBox"):
        try:
            f = _make_array_function(values["f(x)"])
            a = float(_make_array_function(values["a"])(0.0))
            b = float(_make_array_function(values["b"])(0.0))
            points = int(float(values["points (1/2/3)"]))
            if points not in (1, 2, 3):
                raise ValueError("points must be 1, 2, or 3.")
            print_table = bool(int(float(values["print_table (0/1)"])))
        except Exception as e:
            messagebox.showerror("Invalid input", str(e))
            return

        def runner():
            return gaussian_quadrature(f, a, b, points=points, print_table=print_table)

        out_text, integral = _run_and_capture_with_result(runner)
        out_widget.set_text(out_text)
        if integral is not None:
            self._open_gauss_graph(
                f=f, a=a, b=b, points=points, integral_value=float(integral),
                title=f"{points}-point Gauss-Legendre on [{a}, {b}]",
            )

    def open_romberg(self):
        if not self._ensure_exists(SCRIPTS["romberg"]):
            return
        mod = _load_module_from_path(SCRIPTS["romberg"].path)
        romberg_integration = getattr(mod, "romberg_integration", None)
        if romberg_integration is None:
            messagebox.showerror("Missing function", "Could not find `romberg_integration` in romberg_integration.py")
            return

        win = MethodWindow(
            self,
            title="Romberg Integration",
            subtitle=METHOD_DESCRIPTIONS["romberg"],
            category=METHOD_CATEGORY["romberg"],
            fields=[
                Field("f(x)", "4/(1+x**2)",
                      help="Classical demo: \u222b\u2080\u00b9 4/(1+x\u00b2) dx = \u03c0."),
                Field("a", "0"),
                Field("b", "1"),
                Field("max_order", "5", kind="spin", spin_from=2, spin_to=20),
                Field("tol", "1e-8", help="Convergence threshold for |R[k][k] - R[k-1][k-1]|."),
                Field("print_table (0/1)", "1", kind="check"),
            ],
            on_run=lambda values, out: self._run_romberg(romberg_integration, values, out),
        )
        win.show()

    def _run_romberg(self, romberg_integration, values: dict, out_widget: "OutputBox"):
        try:
            f = _make_array_function(values["f(x)"])
            a = float(_make_array_function(values["a"])(0.0))
            b = float(_make_array_function(values["b"])(0.0))
            max_order = int(float(values["max_order"]))
            tol = float(values["tol"])
            print_table = bool(int(float(values["print_table (0/1)"])))
            if max_order < 2:
                raise ValueError("max_order must be >= 2.")
        except Exception as e:
            messagebox.showerror("Invalid input", str(e))
            return

        def runner():
            return romberg_integration(f, a, b, max_order=max_order, tol=tol, print_table=print_table)

        out_text, result = _run_and_capture_with_result(runner)
        out_widget.set_text(out_text)
        try:
            if isinstance(result, tuple) and len(result) >= 3:
                I_val, _converged, table = result[0], result[1], result[2]
                self._open_romberg_graph(R=table, title=f"Romberg Convergence  (\u2248 {float(I_val):.12f})")
        except Exception:
            pass

    def open_differentiation(self):
        if not self._ensure_exists(SCRIPTS["differentiation"]):
            return
        mod = _load_module_from_path(SCRIPTS["differentiation"].path)
        numerical_derivative_from_data = getattr(mod, "numerical_derivative_from_data", None)
        if numerical_derivative_from_data is None:
            messagebox.showerror("Missing function", "Could not find `numerical_derivative_from_data` in numerical_differentiation.py")
            return

        win = MethodWindow(
            self,
            title="Numerical Differentiation",
            subtitle=METHOD_DESCRIPTIONS["differentiation"],
            category=METHOD_CATEGORY["differentiation"],
            fields=[
                Field("Mode (function/table)", "function", kind="combo",
                      choices=["function", "table"]),
                Field("f(x)  [function mode]", "sin(x)",
                      help="Default demo: f(x) = sin(x); exact f'(\u03c0/4) = cos(\u03c0/4) \u2248 0.7071."),
                Field("x_target(s) (space sep)", "0.7853981633974483",
                      help="Point(s) at which to evaluate the derivative."),
                Field("h  [function mode]", "0.1",
                      help="Step size used to build the local stencil."),
                Field("x_data  [table mode, space sep]", "",
                      help="Equally-spaced x values (table mode only)."),
                Field("y_data  [table mode, space sep]", ""),
                Field("method", "central2", kind="combo",
                      choices=["auto", "forward2", "backward2", "central2", "forward3", "backward3"],
                      help="O(h\u00b2) methods: central2, forward3, backward3."),
                Field("print_table (0/1)", "1", kind="check"),
            ],
            on_run=lambda values, out: self._run_differentiation(numerical_derivative_from_data, values, out),
        )
        win.show()

    def _run_differentiation(self, numerical_derivative_from_data, values: dict, out_widget: "OutputBox"):
        try:
            mode = (values["Mode (function/table)"] or "function").strip().lower()
            method = (values["method"] or "auto").strip().lower()
            print_table = bool(int(float(values["print_table (0/1)"])))
            targets = _parse_number_list(values["x_target(s) (space sep)"])
            if not targets:
                raise ValueError("Provide at least one x_target.")
            x_target_arr = np.array(targets, dtype=float)

            f_for_plot = None

            if mode.startswith("t"):
                xs = _parse_number_list(values["x_data  [table mode, space sep]"])
                ys = _parse_number_list(values["y_data  [table mode, space sep]"])
                if len(xs) < 2 or len(xs) != len(ys):
                    raise ValueError("x_data and y_data must be non-empty and the same length.")
                x_data = np.array(xs, dtype=float)
                y_data = np.array(ys, dtype=float)
            else:
                f = _make_array_function(values["f(x)  [function mode]"])
                f_for_plot = f
                h = float(_make_array_function(values["h  [function mode]"])(0.0))
                if h <= 0:
                    raise ValueError("h must be > 0.")
                # Build a small equally-spaced grid that covers all targets, with extra points
                # so any 3-point stencil works at any target.
                t_min = float(x_target_arr.min())
                t_max = float(x_target_arr.max())
                grid_start = t_min - 2 * h
                grid_end = t_max + 2 * h
                n_pts = int(round((grid_end - grid_start) / h)) + 1
                x_data = grid_start + h * np.arange(n_pts)
                y_data = f(x_data)
                # snap targets onto grid (within tolerance) so the script finds them
                snapped = []
                for t in x_target_arr:
                    i = int(np.argmin(np.abs(x_data - t)))
                    snapped.append(float(x_data[i]))
                x_target_arr = np.array(snapped, dtype=float)
        except Exception as e:
            messagebox.showerror("Invalid input", str(e))
            return

        def runner():
            return numerical_derivative_from_data(
                x_data, y_data, x_target_arr, method=method, print_table=print_table
            )

        out_text, deriv = _run_and_capture_with_result(runner)
        out_widget.set_text(out_text)
        if deriv is not None:
            d_arr = np.atleast_1d(np.asarray(deriv, dtype=float))
            self._open_diff_graph(
                x_data=x_data, y_data=y_data,
                x_targets=x_target_arr, derivs=d_arr, f=f_for_plot,
                title=f"Numerical Differentiation ({method})",
            )


class OutputBox(ttk.Frame):
    def __init__(self, master):
        super().__init__(master, style="Panel.TFrame")

        toolbar = ttk.Frame(self, style="Panel.TFrame")
        toolbar.pack(fill="x", pady=(0, 4))
        ttk.Label(toolbar, text="Output", style="Section.TLabel").pack(side="left")
        ttk.Button(toolbar, text="Copy", command=self.copy_to_clipboard).pack(side="right")
        ttk.Button(toolbar, text="Clear", command=self.clear).pack(side="right", padx=(0, 6))

        body = ttk.Frame(self, style="Panel.TFrame")
        body.pack(fill="both", expand=True)

        self.text = tk.Text(
            body, wrap="none", height=14,
            background="#0f172a", foreground="#e2e8f0",
            insertbackground="#e2e8f0", selectbackground="#334155",
            relief="flat", padx=8, pady=6,
        )
        self.text.configure(font=("Consolas", 10))
        self.text.pack(side="left", fill="both", expand=True)

        yscroll = ttk.Scrollbar(body, orient="vertical", command=self.text.yview)
        yscroll.pack(side="right", fill="y")
        self.text.configure(yscrollcommand=yscroll.set)

    def set_text(self, value: str):
        self.text.configure(state="normal")
        self.text.delete("1.0", "end")
        self.text.insert("1.0", value)
        self.text.configure(state="disabled")
        self.text.see("1.0")

    def clear(self):
        self.set_text("")

    def copy_to_clipboard(self):
        try:
            data = self.text.get("1.0", "end-1c")
            self.clipboard_clear()
            self.clipboard_append(data)
        except Exception:
            pass


class _ScrollableFrame(ttk.Frame):
    """Vertical scrollable container that resizes its inner frame to its width."""

    def __init__(self, master, *, height: int = 280):
        super().__init__(master)
        self._canvas = tk.Canvas(self, highlightthickness=0, height=height,
                                 background="#ffffff", borderwidth=0)
        self._vbar = ttk.Scrollbar(self, orient="vertical", command=self._canvas.yview)
        self._canvas.configure(yscrollcommand=self._vbar.set)

        self._canvas.pack(side="left", fill="both", expand=True)
        self._vbar.pack(side="right", fill="y")

        self.inner = ttk.Frame(self._canvas, style="Panel.TFrame")
        self._win_id = self._canvas.create_window((0, 0), window=self.inner, anchor="nw")

        self.inner.bind("<Configure>", self._on_inner_configure)
        self._canvas.bind("<Configure>", self._on_canvas_configure)

        # Mouse wheel only while pointer is inside.
        self._canvas.bind("<Enter>", lambda _e: self._bind_wheel())
        self._canvas.bind("<Leave>", lambda _e: self._unbind_wheel())

    def _on_inner_configure(self, _e):
        self._canvas.configure(scrollregion=self._canvas.bbox("all"))

    def _on_canvas_configure(self, e):
        self._canvas.itemconfig(self._win_id, width=e.width)

    def _bind_wheel(self):
        self._canvas.bind_all("<MouseWheel>", self._on_mousewheel)

    def _unbind_wheel(self):
        self._canvas.unbind_all("<MouseWheel>")

    def _on_mousewheel(self, e):
        self._canvas.yview_scroll(int(-e.delta / 120), "units")


class MethodWindow:
    """Modern method window: category chip, scrollable form, status pill, shortcuts."""

    def __init__(
        self,
        root: tk.Tk,
        title: str,
        fields: list,
        on_run: Callable[[dict, OutputBox], None],
        *,
        subtitle: Optional[str] = None,
        category: Optional[str] = None,
    ):
        self.root = root
        self.title = title
        self.fields: list[Field] = _normalize_fields(fields)
        self.on_run = on_run
        self.category = category

        self.win = tk.Toplevel(root)
        self.win.title(title)
        self.win.geometry("820x680")
        self.win.minsize(740, 580)
        _apply_global_theme(self.win)
        self.win.configure(background=COLORS["bg"])

        # Top accent strip (uses the category color when known)
        accent = (CATEGORY_INFO.get(category, {}) or {}).get("color", COLORS["accent"])
        tk.Frame(self.win, bg=accent, height=3).pack(fill="x")

        outer = ttk.Frame(self.win, padding=18, style="App.TFrame")
        outer.pack(fill="both", expand=True)

        # ---------- Header (icon chip + title + chip pill) ----------
        header = tk.Frame(outer, bg=COLORS["bg"])
        header.pack(fill="x")

        if category and category in CATEGORY_INFO:
            cinfo = CATEGORY_INFO[category]
            tk.Label(
                header, text=cinfo["icon"], fg=cinfo["color"], bg=COLORS["bg"],
                font=("Segoe UI", 24, "bold"),
            ).pack(side="left")
            text_wrap = tk.Frame(header, bg=COLORS["bg"])
            text_wrap.pack(side="left", padx=(12, 0), anchor="w")
            ttk.Label(text_wrap, text=title, style="Header.TLabel").pack(anchor="w")
            if subtitle:
                ttk.Label(text_wrap, text=subtitle, style="Sub.TLabel",
                          wraplength=680, justify="left").pack(anchor="w", pady=(2, 0))

            # Category chip on the right
            chip = tk.Label(
                header, text=f"  {cinfo['icon']}  {cinfo['label']}  ",
                bg=_MethodCard._tint(cinfo["color"]),
                fg=cinfo["color"],
                font=("Segoe UI", 9, "bold"), padx=4, pady=3,
            )
            chip.pack(side="right", anchor="ne")
        else:
            ttk.Label(header, text=title, style="Header.TLabel").pack(anchor="w")
            if subtitle:
                ttk.Label(header, text=subtitle, style="Sub.TLabel",
                          wraplength=720, justify="left").pack(anchor="w", pady=(2, 0))

        tk.Frame(outer, bg=COLORS["border"], height=1).pack(fill="x", pady=(14, 12))

        # ---------- Form (scrollable card) ----------
        form_card = ttk.Labelframe(outer, text="Inputs", style="Card.TLabelframe",
                                   padding=12)
        form_card.pack(fill="x")

        scroller = _ScrollableFrame(
            form_card,
            height=min(340, max(120, 36 * len(self.fields) + 12)),
        )
        scroller.pack(fill="both", expand=True)
        form = scroller.inner
        form.columnconfigure(1, weight=1)

        self.vars: dict[str, tk.Variable] = {}
        self._defaults: dict[str, str] = {f.name: f.default for f in self.fields}
        first_widget: Optional[tk.Widget] = None

        for r, fld in enumerate(self.fields):
            ttk.Label(form, text=fld.name, background=COLORS["panel"]).grid(
                row=r, column=0, sticky="w", padx=(4, 14), pady=6
            )

            widget = self._build_widget(form, fld)
            widget.grid(row=r, column=1, sticky="ew", pady=6)
            if first_widget is None:
                first_widget = widget

            if fld.help:
                ttk.Label(form, text=fld.help, style="Help.TLabel",
                          wraplength=480).grid(row=r, column=2, sticky="w", padx=(12, 4))

        # ---------- Controls + status pill ----------
        controls = ttk.Frame(outer)
        controls.pack(fill="x", pady=(14, 6))

        self.run_btn = ttk.Button(
            controls, text="Run", style="Accent.TButton", command=self._run
        )
        self.run_btn.pack(side="left")
        ttk.Button(controls, text="Reset", style="Ghost.TButton",
                   command=self._reset).pack(side="left", padx=(10, 0))
        ttk.Button(controls, text="Close", style="Ghost.TButton",
                   command=self.win.destroy).pack(side="right")

        status_row = tk.Frame(outer, bg=COLORS["bg"])
        status_row.pack(fill="x", pady=(6, 8))
        self.status_chip = _Chip(status_row, text="Ready", kind="idle")
        self.status_chip.pack(side="left")
        ttk.Label(
            status_row,
            text="  Press Enter to run, Esc to close.",
            style="Status.TLabel",
        ).pack(side="left")

        # ---------- Output ----------
        self.output = OutputBox(outer)
        self.output.pack(fill="both", expand=True)

        # Shortcuts
        self.win.bind("<Return>", lambda _e: self._run())
        self.win.bind("<KP_Enter>", lambda _e: self._run())
        self.win.bind("<Escape>", lambda _e: self.win.destroy())

        if first_widget is not None:
            first_widget.focus_set()

    def _build_widget(self, master: tk.Widget, fld: Field) -> tk.Widget:
        if fld.kind == "combo":
            v = tk.StringVar(value=fld.default)
            self.vars[fld.name] = v
            cb = ttk.Combobox(master, textvariable=v, values=list(fld.choices or []),
                              state="readonly")
            return cb
        if fld.kind == "check":
            v = tk.StringVar(value=str(fld.default or "0"))
            self.vars[fld.name] = v
            holder = ttk.Frame(master, style="Panel.TFrame")
            chk = ttk.Checkbutton(
                holder, variable=v, onvalue="1", offvalue="0",
                text="enabled",
            )
            chk.pack(side="left")
            return holder
        if fld.kind == "spin":
            v = tk.StringVar(value=fld.default)
            self.vars[fld.name] = v
            sp = ttk.Spinbox(master, from_=fld.spin_from, to=fld.spin_to,
                             increment=fld.spin_step, textvariable=v)
            return sp
        v = tk.StringVar(value=fld.default)
        self.vars[fld.name] = v
        return ttk.Entry(master, textvariable=v)

    def _reset(self):
        for name, default in self._defaults.items():
            if name in self.vars:
                self.vars[name].set(default)
        self.status_chip.set("Reset to defaults", "idle")

    def show(self):
        self.win.transient(self.root)
        self.win.grab_set()
        _center_on_screen(self.win, self.root)
        self.win.focus_set()

    def _run(self):
        values = {k: v.get() for k, v in self.vars.items()}
        self.status_chip.set("Computing...", "running")
        self.run_btn.state(["disabled"])
        self.win.update_idletasks()
        t0 = time.perf_counter()
        try:
            self.on_run(values, self.output)
            dt_ms = (time.perf_counter() - t0) * 1000.0
            self.status_chip.set(f"Done in {dt_ms:.1f} ms", "done")
        except Exception as e:
            self.status_chip.set(f"Error: {e}", "error")
            raise
        finally:
            self.run_btn.state(["!disabled"])


def main():
    try:
        app = App()
        app.mainloop()
    except Exception as e:
        messagebox.showerror("Fatal error", str(e))


if __name__ == "__main__":
    main()
