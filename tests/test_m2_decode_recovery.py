import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "recovery_test", ROOT / "scripts/diagnostics/m2_decode_recovery.py"
)
D = importlib.util.module_from_spec(spec)
spec.loader.exec_module(D)


def serialized(reason="stop", ids=None):
    response = SimpleNamespace(
        index=0,
        token_ids=[17, 151645] if ids is None else ids,
        finish_reason=reason,
        stop_reason=None,
        text="fixture",
    )
    return D.P.H.raw_output(
        SimpleNamespace(request_id="0", prompt_token_ids=[1], outputs=[response])
    )


def test_accept_actual_shared_serializer_contract():
    raw = serialized()
    assert "finish_reason" not in raw["outputs"][0]
    assert D.completed_response(raw, [151645]) == [17, 151645]


@pytest.mark.parametrize(
    "reason,ids",
    [("length", [17, 151645]), ("stop", [17]), ("stop", []), ("stop", [17] * 4095 + [151645])],
)
def test_reject_incomplete_or_cap_response(reason, ids):
    with pytest.raises(RuntimeError):
        D.completed_response(serialized(reason, ids), [151645])


def test_reject_multiple_responses():
    raw = serialized()
    raw["outputs"] *= 2
    with pytest.raises(RuntimeError):
        D.completed_response(raw, [151645])
