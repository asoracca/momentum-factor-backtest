import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from fama_french import fit_artifacts


def write(path, frame, metadata):
    frame.to_csv(path, index_label="date")
    metadata["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_suffix(".json").write_text(json.dumps(metadata))


def artifacts(tmp_path, frequency="monthly", kind="zero_investment", missing=False):
    dates = pd.date_range("2020-01-31", periods=48, freq="ME")
    rng = np.random.default_rng(18)
    factors = pd.DataFrame(
        rng.normal(0, 0.01, (48, 5)),
        index=dates,
        columns=["Mkt-RF", "SMB", "HML", "MOM", "RF"],
    )
    returns = 0.002 + factors["Mkt-RF"] * 0.7 + factors["MOM"] * 0.3
    if kind == "total":
        returns += factors.RF
    if missing:
        factors = factors.iloc[1:]
    rp, fp = tmp_path / "returns.csv", tmp_path / "factors.csv"
    write(
        rp,
        returns.to_frame("net_return"),
        dict(frequency="monthly", return_column="net_return", return_kind=kind),
    )
    write(
        fp,
        factors,
        dict(frequency=frequency, units="decimal", source="synthetic factors"),
    )
    return rp, fp


@pytest.mark.parametrize("kind", ["zero_investment", "total"])
def test_selected_artifact_and_rf_convention(tmp_path, kind):
    paths = artifacts(tmp_path, kind=kind)
    result = fit_artifacts(*paths)
    assert result["n"] == 48
    assert result["alpha"] == pytest.approx(0.002)
    assert result["betas"]["Mkt-RF"] == pytest.approx(0.7)
    assert result["frequency"] == "monthly"


def test_frequency_and_date_mismatch_fail(tmp_path):
    with pytest.raises(ValueError, match="frequency mismatch"):
        fit_artifacts(*artifacts(tmp_path, frequency="daily"))
    with pytest.raises(ValueError, match="date coverage"):
        fit_artifacts(*artifacts(tmp_path, missing=True))


def test_tampering_and_missing_artifact_do_not_trigger_daily_demo(tmp_path):
    rp, fp = artifacts(tmp_path)
    rp.write_text(rp.read_text() + "\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        fit_artifacts(rp, fp)
    with pytest.raises(FileNotFoundError):
        fit_artifacts(tmp_path / "absent.csv", fp)
