"""The error classes of loupmd.errors: printed shape, bases, pickling. Needs neither the library nor a GPU."""
from __future__ import annotations

import pickle

import pytest

from loupmd.errors import BackendError, InputError, LoupError


def test_a_plain_error_prints_only_what_happened():
    assert str(LoupError("the bundle has no oxygen")) == "the bundle has no oxygen"


def test_why_and_next_step_are_printed_on_their_own_lines():
    err = LoupError("the bundle has no oxygen", why="it was exported for Sr and Ti", next_step="export it again")
    assert str(err) == "the bundle has no oxygen\nWhy: it was exported for Sr and Ti\nNext: export it again"
    assert (err.what, err.why, err.next_step) == ("the bundle has no oxygen", "it was exported for Sr and Ti",
                                                  "export it again")


def test_backend_error_carries_the_call_and_the_engine_text():
    err = BackendError("loup_evaluate: singular or zero cell", call="loup_evaluate")
    assert err.call == "loup_evaluate"
    assert str(err) == "loup_evaluate: singular or zero cell"


def test_bases_keep_existing_handlers_working():
    """Callers that caught RuntimeError from the engine or ValueError from the wrapper's shape checks still do."""
    assert issubclass(BackendError, LoupError) and issubclass(BackendError, RuntimeError)
    assert issubclass(InputError, LoupError) and issubclass(InputError, ValueError)
    with pytest.raises(RuntimeError, match="singular"):
        raise BackendError("loup_evaluate: singular or zero cell", call="loup_evaluate")


@pytest.mark.parametrize("err", [
    LoupError("a", why="b", next_step="c"),
    BackendError("loup_run: no system set", call="loup_run"),
    InputError("positions has shape (3,), expected (5, 3)"),
])
def test_errors_survive_pickling(err):
    back = pickle.loads(pickle.dumps(err))
    assert type(back) is type(err) and str(back) == str(err) and back.__dict__ == err.__dict__
