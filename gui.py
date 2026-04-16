import io
import ast
import math
import re
import subprocess
import sys
from contextlib import redirect_stdout
from dataclasses import dataclass
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
    def __init__(self):
        super().__init__()
        self.title("Numerical Methods - Start Menu")
        self.geometry("520x520")
        self.minsize(520, 520)

        container = ttk.Frame(self, padding=16)
        container.pack(fill="both", expand=True)

        title = ttk.Label(
            container,
            text="Select a numerical method",
            font=("Segoe UI", 14, "bold"),
        )
        title.pack(anchor="w")

        buttons = ttk.Frame(container)
        buttons.pack(fill="x")

        ttk.Button(
            buttons,
            text=SCRIPTS["secant"].label,
            command=self.open_secant,
        ).pack(fill="x", pady=6)

        ttk.Button(
            buttons,
            text=SCRIPTS["fixed_point"].label,
            command=self.open_fixed_point,
        ).pack(fill="x", pady=6)

        ttk.Button(
            buttons,
            text=SCRIPTS["false_position"].label,
            command=self.open_false_position,
        ).pack(fill="x", pady=6)

        ttk.Button(
            buttons,
            text=SCRIPTS["halley"].label,
            command=self.open_halley,
        ).pack(fill="x", pady=6)

        ttk.Button(
            buttons,
            text=SCRIPTS["newton"].label,
            command=self.open_newton,
        ).pack(fill="x", pady=6)

        ttk.Button(
            buttons,
            text=SCRIPTS["bisection"].label,
            command=self.open_bisection,
        ).pack(fill="x", pady=6)

        ttk.Button(
            buttons,
            text=SCRIPTS["jacobi"].label,
            command=self.open_jacobi,
        ).pack(fill="x", pady=6)

        ttk.Button(
            buttons,
            text=SCRIPTS["gauss_seidel"].label,
            command=self.open_gauss_seidel,
        ).pack(fill="x", pady=6)

        ttk.Button(
            buttons,
            text=SCRIPTS["newton_system"].label,
            command=self.open_newton_system,
        ).pack(fill="x", pady=6)

        ttk.Separator(container).pack(fill="x", pady=14)

        bottom = ttk.Frame(container)
        bottom.pack(fill="x")
        
        ttk.Button(bottom, text="About", command=self.show_about).pack(side="left")
        ttk.Button(bottom, text="Exit", command=self.destroy).pack(side="right")
        self._open_figures = []

    def show_about(self):
        about = tk.Toplevel(self)
        about.title("About")
        about.resizable(False, False)
        about.transient(self)
        about.grab_set()

        frame = ttk.Frame(about, padding=16)
        frame.pack(fill="both", expand=True)

        text = (
            "Made by:\n"
            "Arsany Ehab Alfy -- ID: 24100390\n"
            "Omar Mustafa Elnainay -- ID: 24100142\n"
            "Abdelrahman Ahmed Ibrahim -- ID: 24100417\n"
            "Mohamed Wael Abdelmaqsoud -- ID: 24100444\n\n"
            "Course: Numerical Methods\n"
            "Instructor: Dr. Abdellatif Mahmoud"
        )

        lbl = ttk.Label(frame, text=text, justify="center", anchor="center")
        lbl.pack(fill="both", expand=True)

        ttk.Button(frame, text="Close", command=about.destroy).pack(pady=(12, 0))

        about.update_idletasks()
        w = about.winfo_width()
        h = about.winfo_height()
        x = self.winfo_rootx() + (self.winfo_width() - w) // 2
        y = self.winfo_rooty() + (self.winfo_height() - h) // 2
        about.geometry(f"+{x}+{y}")

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
            fig, ax, ani = plot_iteration_trace(f, xs, title=title, frames=200, interval=600, repeat=True, root_marker=True)
            self._open_figures.append((fig, ax, ani))
            plt.show(block=False)
        except Exception as e:
            messagebox.showerror("Graph failed", str(e))

    def _open_trajectory_window(self, traces: list[list[float]], title: str):
        """Open a 2-D or 3-D animated phase-trajectory window."""
        try:
            import matplotlib.pyplot as plt
            if len(traces) == 2:
                from graph import plot_phase_trajectory_2d
                fig, ax, ani = plot_phase_trajectory_2d(
                    traces[0], traces[1], title=title, frames=200, interval=600, repeat=True
                )
            elif len(traces) == 3:
                from graph import plot_phase_trajectory_3d
                fig, ax, ani = plot_phase_trajectory_3d(
                    traces[0], traces[1], traces[2], title=title, frames=200, interval=600, repeat=True
                )
            else:
                return
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
            fields=[
                ("f(x)", "x**3 - x - 2"),
                ("x0", "1"),
                ("x1", "2"),
                ("tol", "1e-6"),
                ("max_iter", "100"),
                ("print_table (0/1)", "1"),
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
            fields=[
                ("f(x)", "x**3 - x - 2"),
                ("g(x)", "(x + 2)**(1/3)"),
                ("x0", "1.5"),
                ("tol", "1e-6"),
                ("max_iter", "100"),
                ("print_table (0/1)", "1"),
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
            fields=[
                ("f(x)", "x**3 - x - 2"),
                ("a", "1"),
                ("b", "2"),
                ("tol", "1e-6"),
                ("max_iter", "100"),
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
            fields=[
                ("f(x)", "x**3 - x - 2"),
                ("f'(x)", "3*x**2 - 1"),
                ("f''(x)", "6*x"),
                ("x0", "1.5"),
                ("tol", "1e-6"),
                ("max_iter", "100"),
                ("print_table (0/1)", "1"),
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
            fields=[
                ("f(x)", "x**3 - x - 2"),
                ("f'(x)", "3*x**2 - 1"),
                ("x0", "1.5"),
                ("tol", "1e-6"),
                ("max_iter", "100"),
                ("print_table (0/1)", "1"),
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
            fields=[
                ("f(x)", "x**3 - x - 2"),
                ("a", "1"),
                ("b", "2"),
                ("tol", "1e-6"),
                ("max_iter", "100"),
                ("print_table (0/1)", "1"),
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
            fields=[
                ("n (2 or 3)", "3"),
                ("A rows (; or newline)", "10 2 1; 1 5 1; 2 3 10"),
                ("b", "14 10 14"),
                ("x0 (optional)", "0 0 0"),
                ("tol", "1e-6"),
                ("max_iter", "100"),
                ("print_table (0/1)", "1"),
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
            fields=[
                ("n (2 or 3)", "3"),
                ("A rows (; or newline)", "10 2 1; 1 5 1; 2 3 10"),
                ("b", "14 10 14"),
                ("x0 (optional)", "0 0 0"),
                ("tol", "1e-6"),
                ("max_iter", "100"),
                ("print_table (0/1)", "1"),
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
            fields=[
                ("n (2 or 3)", "3"),
                ("initial (x0 y0 [z0])", "0.5 0.5 0.8"),
                ("tol", "1e-6"),
                ("max_iter", "50"),
                ("f1(x,y,z)", "x^2+y^2+z^2-1"),
                ("f2(x,y,z)", "x+y-z"),
                ("f3(x,y,z) (only if n=3)", "x-y"),
                ("df1/dx", "2x"),
                ("df1/dy", "2y"),
                ("df1/dz (only if n=3)", "2z"),
                ("df2/dx", "1"),
                ("df2/dy", "1"),
                ("df2/dz (only if n=3)", "-1"),
                ("df3/dx (only if n=3)", "1"),
                ("df3/dy (only if n=3)", "-1"),
                ("df3/dz (only if n=3)", "0"),
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
            self._open_trajectory_window(traces, "Newton-Raphson System - Phase Trajectory")


class OutputBox(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.text = tk.Text(self, wrap="none", height=14)
        self.text.configure(font=("Consolas", 10))
        self.text.pack(side="left", fill="both", expand=True)

        yscroll = ttk.Scrollbar(self, orient="vertical", command=self.text.yview)
        yscroll.pack(side="right", fill="y")
        self.text.configure(yscrollcommand=yscroll.set)

    def set_text(self, value: str):
        self.text.configure(state="normal")
        self.text.delete("1.0", "end")
        self.text.insert("1.0", value)
        self.text.configure(state="disabled")


class MethodWindow:
    def __init__(
        self,
        root: tk.Tk,
        title: str,
        fields: list[tuple[str, str]],
        on_run: Callable[[dict, OutputBox], None],
    ):
        self.root = root
        self.title = title
        self.fields = fields
        self.on_run = on_run

        self.win = tk.Toplevel(root)
        self.win.title(title)
        self.win.geometry("720x520")
        self.win.minsize(680, 480)

        frame = ttk.Frame(self.win, padding=12)
        frame.pack(fill="both", expand=True)

        form = ttk.Frame(frame)
        form.pack(fill="x")

        self.vars: dict[str, tk.StringVar] = {}
        for r, (name, default) in enumerate(fields):
            ttk.Label(form, text=name).grid(row=r, column=0, sticky="w", padx=(0, 10), pady=4)
            v = tk.StringVar(value=default)
            self.vars[name] = v
            entry = ttk.Entry(form, textvariable=v)
            entry.grid(row=r, column=1, sticky="ew", pady=4)

        form.columnconfigure(1, weight=1)

        controls = ttk.Frame(frame)
        controls.pack(fill="x", pady=(10, 6))

        ttk.Button(controls, text="Run", command=self._run).pack(side="left")
        ttk.Button(controls, text="Close", command=self.win.destroy).pack(side="left", padx=8)

        self.output = OutputBox(frame)
        self.output.pack(fill="both", expand=True, pady=(6, 0))

    def show(self):
        self.win.transient(self.root)
        self.win.grab_set()
        self.win.focus_set()

    def _run(self):
        values = {k: v.get() for k, v in self.vars.items()}
        self.on_run(values, self.output)


def main():
    try:
        app = App()
        app.mainloop()
    except Exception as e:
        messagebox.showerror("Fatal error", str(e))


if __name__ == "__main__":
    main()
