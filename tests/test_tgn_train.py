"""S1.7: pure helpers of the DyGLib training wrapper (no torch needed).

Spec under test (src/citecurv/tgn/train.py):
- dyglib_split_ratios(ts, train_end_t, val_end_t) returns (val_ratio, test_ratio) such that
  DyGLib's own quantile rule (val_time, test_time = np.quantile(ts, [1 - v - t, 1 - t]);
  train = ts <= val_time; val = val_time < ts <= test_time) reproduces the date cut exactly,
  including when many events share a timestamp;
- parse_dyglib_messages turns DyGLib's log lines into per-epoch and final metrics;
- argument helpers map Python kwargs and devices to DyGLib's CLI.
"""

import numpy as np
import pytest

from citecurv.tgn.train import _cli_args, _gpu_index, dyglib_split_ratios, parse_dyglib_messages


def dyglib_masks(ts, val_ratio, test_ratio):
    """DyGLib utils/DataLoader.py split, copied as a reference."""
    val_time, test_time = np.quantile(ts, [(1 - val_ratio - test_ratio), (1 - test_ratio)])
    train = ts <= val_time
    val = (ts <= test_time) & (ts > val_time)
    test = ts > test_time
    return train, val, test


def test_split_ratios_hand_case():
    ts = np.array([1, 1, 2, 2, 2, 3, 4, 4, 5, 6], dtype=float)
    v, t = dyglib_split_ratios(ts, train_end_t=2, val_end_t=4)
    train, val, test = dyglib_masks(ts, v, t)
    assert train.tolist() == [True] * 5 + [False] * 5
    assert val.tolist() == [False] * 5 + [True] * 3 + [False] * 2
    assert test.tolist() == [False] * 8 + [True] * 2


def test_split_ratios_random_ties_brute_force():
    rng = np.random.default_rng(0)
    for _ in range(200):
        n = int(rng.integers(10, 400))
        ts = np.sort(rng.integers(0, int(rng.integers(5, 60)), size=n)).astype(float)
        uniq = np.unique(ts)
        if len(uniq) < 3:
            continue
        a, b = sorted(rng.choice(uniq[:-1], size=2, replace=False))
        v, t = dyglib_split_ratios(ts, a, b)
        train, val, test = dyglib_masks(ts, v, t)
        np.testing.assert_array_equal(train, ts <= a)
        np.testing.assert_array_equal(val, (ts > a) & (ts <= b))
        np.testing.assert_array_equal(test, ts > b)


def test_split_ratios_needs_three_nonempty_periods():
    with pytest.raises(ValueError):
        dyglib_split_ratios([1, 2, 3], train_end_t=3, val_end_t=3)


RECORDS = [
    (100.0, "********** Run 1 starts. **********"),
    (100.1, "configuration is Namespace(...)"),
    (130.0, "Epoch: 1, learning rate: 0.001, train loss: 0.6901"),
    (130.0, "train average_precision, 0.6755"),
    (130.0, "train roc_auc, 0.7154"),
    (130.0, "validate loss: 0.6897"),
    (130.0, "validate average_precision, 0.7815"),
    (130.0, "validate roc_auc, 0.7522"),
    (130.0, "new node validate loss: 0.6920"),
    (130.0, "new node validate average_precision, 0.6125"),
    (130.0, "new node validate roc_auc, 0.6113"),
    (130.1, "save model ./saved_models/TGN/x/TGN_seed0/TGN_seed0.pkl"),
    (155.5, "Epoch: 2, learning rate: 0.001, train loss: 0.6500"),
    (155.5, "train average_precision, 0.7000"),
    (155.5, "validate average_precision, 0.7700"),
    (156.0, "load model ./saved_models/TGN/x/TGN_seed0/TGN_seed0.pkl"),
    (156.0, "get final performance on dataset x..."),
    (157.0, "test loss: 0.6912"),
    (157.0, "test average_precision, 0.6972"),
    (157.0, "test roc_auc, 0.6776"),
    (157.0, "new node test average_precision, 0.6503"),
    (157.0, "test average_precision, [np.float64(0.6971940370066095)]"),
    (157.0, "average test average_precision, 0.6972 ± nan"),
]


def test_parse_dyglib_messages():
    m = parse_dyglib_messages(RECORDS)
    assert [e["epoch"] for e in m["epochs"]] == [1, 2]
    e1, e2 = m["epochs"]
    assert e1["train_loss"] == 0.6901 and e1["train_ap"] == 0.6755 and e1["train_auc"] == 0.7154
    assert e1["val_loss"] == 0.6897 and e1["val_ap"] == 0.7815 and e1["val_auc"] == 0.7522
    assert e1["new_node_val_ap"] == 0.6125 and e1["new_node_val_loss"] == 0.6920
    assert e1["wall_clock_s"] == pytest.approx(30.0)
    assert e2["wall_clock_s"] == pytest.approx(25.5)
    assert e2["val_ap"] == 0.77 and e2["learning_rate"] == 0.001
    assert m["best_epoch"] == 1  # last "save model" came after epoch 1
    assert m["test"] == {"loss": 0.6912, "ap": 0.6972, "auc": 0.6776}
    assert m["new_node_test"] == {"ap": 0.6503}


def test_cli_args_and_gpu_index():
    assert _cli_args({"learning_rate": 0.001, "load_best_configs": False, "x": None, "flag": True}) == [
        "--learning_rate", "0.001", "--flag"]
    assert _gpu_index("cpu") == -1 and _gpu_index("cuda") == 0 and _gpu_index("cuda:2") == 2
    with pytest.raises(ValueError):
        _gpu_index("tpu")


def test_notebook_40_parses_and_checks_versions():
    nbformat = pytest.importorskip("nbformat")
    from citecurv.config import REPO_ROOT

    nb = nbformat.read(REPO_ROOT / "notebooks" / "40_tgn.ipynb", as_version=4)
    nbformat.validate(nb)
    first_code = next(c for c in nb.cells if c.cell_type == "code").source
    assert "sys.version_info[:2] != (3, 12)" in first_code and "torch.__version__" in first_code
    src = "\n".join(c.source for c in nb.cells if c.cell_type == "code")
    for call in ("to_jodie(", "train_tgn(", "infer_tgn(", "drive.mount("):
        assert call in src
