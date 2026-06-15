"""
Pedagogical iteration logger for transparent solver tracking.

The IterationLogger class is the core of NumAnalytica's "glass-box"
philosophy. It captures iteration-by-iteration details, enabling students
and researchers to study the inner workings of numerical algorithms.
"""

import io
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class IterationRecord:
    """Single iteration record."""

    iteration: int
    state: Dict[str, Any]  # Current solution state
    residual: float
    function_evals: int = 0
    derivative_evals: int = 0
    step_length: Optional[float] = None
    note: str = ""


class IterationLogger:
    """
    Builds and formats iteration history for pedagogical output.

    This class tracks every iteration of a numerical algorithm, recording:
    - Current solution estimate
    - Residual/error measure
    - Step properties (step size, convergence rate)
    - Diagnostic notes

    Design Philosophy:
        Unlike production solvers that only report final results, NumAnalytica
        logs every step to expose the mathematical journey. This makes it
        ideal for teaching and debugging.

    Example
    -------
    >>> logger = IterationLogger()
    >>> logger.record_iteration(
    ...     iteration=0,
    ...     state={'x': 1.5},
    ...     residual=0.25,
    ...     step_length=0.5
    ... )
    >>> print(logger.format_table())
    """

    def __init__(self, verbose: bool = True):
        """
        Initialize the logger.

        Parameters
        ----------
        verbose : bool, default=True
            If True, print iteration details in real-time.
        """
        self.verbose = verbose
        self.records: List[IterationRecord] = []
        self._headers: Optional[List[str]] = None
        self._col_widths: Dict[str, int] = {}

    def record_iteration(
        self,
        iteration: int,
        state: Dict[str, Any],
        residual: float,
        function_evals: int = 0,
        derivative_evals: int = 0,
        step_length: Optional[float] = None,
        note: str = "",
    ) -> None:
        """
        Record a single iteration.

        Parameters
        ----------
        iteration : int
            Iteration counter.
        state : Dict[str, Any]
            Current solution state {variable_name: value}.
        residual : float
            Current residual or error.
        function_evals : int, optional
            Number of function evaluations in this step.
        derivative_evals : int, optional
            Number of derivative evaluations in this step.
        step_length : float, optional
            Step length or distance moved.
        note : str, optional
            Diagnostic note (e.g., "convergence detected", "step rejected").
        """
        record = IterationRecord(
            iteration=iteration,
            state=state,
            residual=residual,
            function_evals=function_evals,
            derivative_evals=derivative_evals,
            step_length=step_length,
            note=note,
        )
        self.records.append(record)

        if self.verbose:
            self._print_iteration(record)

    def _print_iteration(self, record: IterationRecord) -> None:
        """Pretty-print a single iteration."""
        state_str = ", ".join(
            f"{k}={v:.6g}" if isinstance(v, (int, float)) else f"{k}={v}"
            for k, v in record.state.items()
        )
        step_str = f", |step|={record.step_length:.2e}" if record.step_length else ""
        note_str = f" [{record.note}]" if record.note else ""
        print(
            f"  Iter {record.iteration:3d}: {state_str} | "
            f"residual={record.residual:.2e}{step_str}{note_str}"
        )

    def format_table(self) -> str:
        """
        Format iteration history as a formatted ASCII table.

        Returns
        -------
        str
            Formatted table suitable for printing or logging.
        """
        if not self.records:
            return "No iterations recorded."

        output = io.StringIO()
        output.write("\n")
        output.write(" Iteration History Table\n")
        output.write("=" * 80 + "\n")

        # Build header
        headers = ["Iter", "Residual"]

        # Add state variable headers from first record
        first_state_keys = list(self.records[0].state.keys())
        headers.extend(first_state_keys)

        # Add optional headers
        if any(r.step_length is not None for r in self.records):
            headers.append("|step|")
        if any(r.note for r in self.records):
            headers.append("Note")

        # Compute column widths
        col_widths = {h: len(h) + 2 for h in headers}

        # Adjust based on data
        for record in self.records:
            col_widths["Iter"] = max(col_widths["Iter"], len(str(record.iteration)))
            col_widths["Residual"] = max(col_widths["Residual"], 12)  # Scientific notation
            for key in first_state_keys:
                val_str = f"{record.state[key]:.6g}"
                col_widths[key] = max(col_widths[key], len(val_str))
            if record.step_length is not None:
                step_str = f"{record.step_length:.2e}"
                col_widths["|step|"] = max(col_widths["|step|"], len(step_str))
            if record.note:
                col_widths["Note"] = max(col_widths["Note"], len(record.note))

        # Write header row
        header_row = " | ".join(f"{h:>{col_widths[h]}}" for h in headers)
        output.write(header_row + "\n")
        output.write("-+-".join("-" * col_widths[h] for h in headers) + "\n")

        # Write data rows
        for record in self.records:
            row_parts = [
                f"{record.iteration:{col_widths['Iter']}}",
                f"{record.residual:>.2e}"[: col_widths["Residual"]].rjust(col_widths["Residual"]),
            ]

            for key in first_state_keys:
                val_str = f"{record.state[key]:.6g}"
                row_parts.append(val_str.rjust(col_widths[key]))

            if any(r.step_length is not None for r in self.records):
                if record.step_length is not None:
                    step_str = f"{record.step_length:.2e}"
                    row_parts.append(step_str.rjust(col_widths["|step|"]))
                else:
                    row_parts.append(" " * col_widths["|step|"])

            if any(r.note for r in self.records):
                row_parts.append(record.note[: col_widths["Note"]])

            output.write(" | ".join(row_parts) + "\n")

        output.write("=" * 80 + "\n")
        return output.getvalue()

    def get_records(self) -> List[IterationRecord]:
        """Return all iteration records."""
        return self.records

    def to_dict_list(self) -> List[Dict[str, Any]]:
        """
        Convert records to list of dictionaries.

        Returns
        -------
        List[Dict[str, Any]]
            List where each dict contains iteration data.
        """
        result = []
        for record in self.records:
            row = {"iteration": record.iteration, "residual": record.residual}
            row.update(record.state)
            if record.step_length is not None:
                row["step_length"] = record.step_length
            if record.note:
                row["note"] = record.note
            result.append(row)
        return result

    def clear(self) -> None:
        """Clear all recorded iterations."""
        self.records.clear()
