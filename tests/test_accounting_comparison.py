"""rfl/experiments/accounting_comparison.py: presets, argument checks and outputs."""
import csv

import pytest

from rfl.experiments import accounting_comparison as AC


# The (B, T) of the DP-RFedProj paper at eps = 20 (SPDNet, ProjAvg and RetAvg).
PAPER_B_T = {"BNCI2014_001": (256, 200), "Cho2017": (64, 600), "Weibo2014": (32, 450)}


def test_presets_match_the_paper_retavg_configuration():
    for ds, cfg in AC.PRESETS.items():
        assert (cfg["B"], cfg["T"]) == PAPER_B_T[ds]
        assert cfg["B"] <= cfg["m_star"] <= cfg["m_max"]


def test_preset_overrides():
    a = AC.parse_args(["--dataset", "Cho2017", "--batch-size", "128", "--rounds", "300"])
    fed = AC.federations(a)
    assert list(fed) == ["Cho2017"]
    assert fed["Cho2017"]["B"] == 128 and fed["Cho2017"]["T"] == 300
    assert fed["Cho2017"]["m_star"] == AC.PRESETS["Cho2017"]["m_star"]


@pytest.mark.parametrize("argv", [
    ["--dataset", "custom", "--batch-size", "8", "--rounds", "2"],          # incomplete
    ["--dataset", "custom", "--batch-size", "8", "--rounds", "2",
     "--m-star", "100", "--m-max", "50"],                                   # m_max < m_star
    ["--dataset", "BNCI2014_001", "--batch-size", "5000"],                  # B > m_star
])
def test_invalid_federations_exit(argv):
    with pytest.raises(SystemExit):
        AC.federations(AC.parse_args(argv))


def test_custom_run_writes_outputs(tmp_path):
    AC.main(["--dataset", "custom", "--batch-size", "24", "--rounds", "2",
             "--m-star", "300", "--m-max", "340", "--eps", "5",
             "--out", str(tmp_path), "--pgfplots"])
    with open(tmp_path / "accounting_comparison.csv") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    r = rows[0]
    assert r["tau"] == "15" and r["n_steps"] == "30"                         # ceil(340/24)
    assert float(r["eps_ours"]) <= 5.0 + 1e-9
    assert float(r["eps_prirfed"]) <= 5.0 + 1e-9
    assert float(r["ratio"]) >= 1.0
    assert "\\addplot coordinates {(5," in (tmp_path / "accounting_comparison_pgfplots.tex").read_text()
