import numpy as np
import pandas as pd
import pytest

from hffx.backtest import backtest_round, performance_summary


def test_backtest_round_pnl_and_costs_by_hand():
    test = pd.DataFrame({
        "future_return_60s": [0.0010, 0.0005, 0.0020],
        "spread_bps": [1.4, 1.4, 1.4],
    })
    result = backtest_round(test, np.array([1, -1, 0]))

    assert list(result["gross_bps"]) == pytest.approx([10.0, -5.0, 0.0])
    assert list(result["cost_bps"]) == pytest.approx([1.4, 1.4, 0.0])
    assert list(result["net_bps"]) == pytest.approx([8.6, -6.4, 0.0])


def test_performance_summary_by_hand():
    results = pd.DataFrame({
        "session_id": [1, 1, 2, 2],
        "position": [1, 1, 1, 1],
        "gross_bps": [3.0, -2.0, 2.0, -3.0],
        "net_bps": [2.0, -3.0, 1.0, -4.0],
    })
    summary = performance_summary(results)

    assert summary["total_net_bps"] == pytest.approx(-4.0)
    assert summary["max_drawdown_bps"] == pytest.approx(-6.0)
    assert summary["n_trades"] == 4
    assert summary["hit_rate"] == pytest.approx(0.5)
    assert summary["sharpe_per_session"] == pytest.approx(-2.0 / np.sqrt(2))