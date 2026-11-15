"""LoupMD's exceptions and their message shape.

Every exception LoupMD raises derives from ``LoupError``. Its message has one shape, so that no traceback needs
reading: what happened; why, in a sentence or two; what to do next. This module imports nothing else of the package.
"""
from __future__ import annotations


def _restore(cls, args, state):
    """Unpickle an exception without calling its ``__init__`` (whose signature differs per class)."""
    error = cls.__new__(cls)
    error.args = args
    error.__dict__.update(state)
    return error


class LoupError(Exception):
    """Base class of every error LoupMD raises: ``LoupError(what, why=None, next_step=None)``."""

    default_next_step = None

    def __init__(self, what, why=None, next_step=None):
        super().__init__(str(what))
        self.what = str(what)
        self.why = why
        self.next_step = self.default_next_step if next_step is None else next_step

    def __str__(self):
        lines = [self.what]
        if self.why:
            lines.append(f"Why: {self.why}")
        if self.next_step:
            lines.append(f"Next: {self.next_step}")
        return "\n".join(lines)

    def __reduce__(self):
        return _restore, (type(self), self.args, dict(self.__dict__))


class BackendError(LoupError, RuntimeError):
    """A call into libloupmd failed or was refused. ``what`` is the library's own message (``loup_last_error``);
    ``call`` names the C function."""

    def __init__(self, what, why=None, next_step=None, call=None):
        super().__init__(what, why, next_step)
        self.call = call


class InputError(LoupError, ValueError):
    """An argument is refused before the library reads it (a wrong array shape, for example)."""


class CheckpointTrustError(LoupError):
    """A checkpoint would be unpickled without a matching SHA-256 or ``trusted=True``."""

    default_next_step = ("Pass sha256= with the file's published hash, or trusted=True for a file you trust; "
                         "a checkpoint is a pickle and runs code when loaded.")


class CacheLocationError(LoupError):
    """The cache root is somewhere it must not be (inside the Python installation)."""
