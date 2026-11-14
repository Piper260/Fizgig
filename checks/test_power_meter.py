"""The run energy meter (web/power.py): it counts only while a run is live, prefers the card's own energy counter,
falls back to draw x time, keeps a stopped run's figures, and estimates the rest of Training from this run's own
progress (a resume that starts half way doesn't read as nearly done)."""
from fizgig.web import power


class _Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def _meter(monkeypatch):
    clock = _Clock()
    monkeypatch.setattr(power.time, "time", clock)
    return power.EnergyMeter(), clock


def test_counts_only_while_live(monkeypatch):
    m, clock = _meter(monkeypatch)
    m.sample(300, None)
    clock.t += 1
    m.sample(300, None)
    assert m.snapshot() is None                     # no run yet
    m.start()
    for _ in range(10):
        m.sample(360, None)
        clock.t += 1
    m.sample(360, None)
    snap = m.snapshot()
    assert abs(snap["gpu_wh"] - 1.0) < 1e-6         # 360 W for 10 s = 1 Wh
    m.stop()
    clock.t += 100
    m.sample(360, None)
    assert abs(m.snapshot()["gpu_wh"] - 1.0) < 1e-6 # stopped: kept, not added to


def test_prefers_the_energy_counter(monkeypatch):
    m, clock = _meter(monkeypatch)
    m.start()
    m.sample(100, 5_000_000)
    clock.t += 1
    m.sample(100, 5_000_000 + 7_200_000)           # 7.2 kJ in a second whatever the instant draw says
    assert abs(m.snapshot()["gpu_wh"] - 2.0) < 1e-6


def test_rest_of_pc_and_estimate(monkeypatch):
    m, clock = _meter(monkeypatch)
    m.start()
    frac = 0.5                                      # a resume: half done before this run's Training began
    for i in range(61):
        m.sample(300, None, training=True, frac=frac)
        clock.t += 1
        frac += 0.25 / 60                           # a quarter of the run in this minute
    snap = m.snapshot(extra_w=100)
    assert abs(snap["wall_wh"] - snap["gpu_wh"] - 100 * snap["seconds"] / 3600) < 1e-6
    # a quarter left at the same pace: about another minute at 300 W = 5 Wh more
    assert 9.0 < snap["est_gpu_wh"] < 11.5, snap
    assert snap["est_wall_wh"] > snap["est_gpu_wh"]


def test_price_settings():
    assert power.price_settings({"power_price": "0,30", "power_currency": "€", "power_extra_watts": "120"}) == (0.3, "€", 120.0)
    assert power.price_settings({}) == (None, "$", 0.0)
    assert power.price_settings({"power_price": "abc"})[0] is None
