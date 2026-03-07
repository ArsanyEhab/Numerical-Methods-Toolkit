import io
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
        "sqrt": math.sqrt,
        "pi": math.pi,
        "e": math.e,
    }

    def f(x: float) -> float:
        return eval(expr, {"__builtins__": {}}, {**allowed, "x": x})

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


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Numerical Methods - Start Menu")
        self.geometry("520x360")
        self.minsize(520, 360)

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
            fig, ax, ani = plot_iteration_trace(f, xs, title=title, frames=200, interval=400, repeat=True, root_marker=True)
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
