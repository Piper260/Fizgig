"""The GPU's power draw and the energy a run uses, for the sidebar's Power meter and the run summary's Power section.

The stats loop (server._poll_stats_loop) reads the card once a second and hands each reading to `meter.sample`. Energy
comes from the card's own counter where it has one (NVML's total energy, Volta and newer: exact, nothing missed between
readings), else the draw times the time between readings. It is the GPU alone; Preferences' "Rest of the PC" watts are
added on top for an estimate at the wall, and its price per kWh turns it into a cost.

Nothing here touches torch or the GPU beyond the readings it is given."""
import threading
import time

WH = 3600.0                 # joules in a watt-hour


class EnergyMeter:
    def __init__(self):
        self._lock = threading.Lock()
        self._reset()
        self.live = False

    def _reset(self):
        self.started = None            # this run's start (time.time())
        self.joules = 0.0              # the GPU's energy since the start
        self.train_joules = 0.0        # ...since the Training stage began (the estimate's rate)
        self.train_seconds = 0.0
        self.peak_w = 0.0
        self.frac0 = None              # how far through it was when this run's Training began (a resume starts part way)
        self.frac = 0.0
        self._last_t = None
        self._last_counter = None
        self.ended = None

    def start(self):
        """A run starts: count from zero."""
        with self._lock:
            self._reset()
            self.started = time.time()
            self.live = True

    def stop(self):
        """The run ended (finished, failed, stopped, paused): its figures stay until the next start."""
        with self._lock:
            if self.live:
                self.ended = time.time()
            self.live = False
            self._last_t = self._last_counter = None

    def sample(self, watts, counter_mj=None, training=False, frac=0.0):
        """One reading: the draw now (W), the card's energy counter (mJ, None without one), and whether the Training
        stage is on and how far through the run it is (0-1). Counts only while a run is live."""
        now = time.time()
        with self._lock:
            if not self.live or watts is None:
                self._last_t, self._last_counter = (now, counter_mj) if self.live else (None, None)
                return
            dt = now - self._last_t if self._last_t is not None else 0.0
            if counter_mj is not None and self._last_counter is not None and counter_mj >= self._last_counter:
                j = (counter_mj - self._last_counter) / 1000.0
            else:
                j = max(0.0, float(watts)) * min(dt, 5.0)       # a stalled loop: don't credit a long gap
            if dt > 0:
                self.joules += j
                if training:
                    if self.frac0 is None:
                        self.frac0 = max(0.0, min(1.0, float(frac or 0.0)))
                    self.train_joules += j
                    self.train_seconds += dt
                    self.frac = max(self.frac, float(frac or 0.0))
            self.peak_w = max(self.peak_w, float(watts))
            self._last_t, self._last_counter = now, counter_mj

    def snapshot(self, extra_w=0.0):
        """The run's figures for the page: watt-hours so far (GPU, and with the rest of the PC), the average draw, and
        an estimate of the whole run's: the rest of Training at the rate this run's Training has gone so far."""
        with self._lock:
            if self.started is None:
                return None
            end = time.time() if self.live else (self.ended or time.time())
            secs = max(0.0, end - self.started)
            gpu_wh = self.joules / WH
            avg_w = self.joules / secs if secs > 0 else 0.0
            out = {"live": self.live, "seconds": secs, "gpu_wh": gpu_wh, "avg_w": avg_w, "peak_w": self.peak_w,
                   "wall_wh": gpu_wh + max(0.0, extra_w) * secs / WH, "est_gpu_wh": None, "est_wall_wh": None}
            done = self.frac - (self.frac0 or 0.0)          # progress made in this run's Training
            if self.live and done >= 0.01 and self.frac < 1.0 and self.train_seconds > 30:
                # the rest of Training at its own average so far (caching and loading draw less, so they're not the rate)
                train_w = self.train_joules / self.train_seconds
                left = self.train_seconds * (1.0 - self.frac) / done
                out["est_gpu_wh"] = gpu_wh + train_w * left / WH
                out["est_wall_wh"] = out["est_gpu_wh"] + max(0.0, extra_w) * (secs + left) / WH
            return out


meter = EnergyMeter()


def read_nvml(nvml, handle):
    """{power, power_limit, util, temp, energy_mj} from NVML; a field the card or driver doesn't give is left out."""
    out = {}
    for key, fn, scale in (("power", "nvmlDeviceGetPowerUsage", 1000.0),
                           ("power_limit", "nvmlDeviceGetEnforcedPowerLimit", 1000.0),
                           ("energy_mj", "nvmlDeviceGetTotalEnergyConsumption", 1.0)):
        try:
            out[key] = getattr(nvml, fn)(handle) / scale
        except Exception:
            pass
    try:
        out["util"] = float(nvml.nvmlDeviceGetUtilizationRates(handle).gpu)
    except Exception:
        pass
    try:
        out["temp"] = float(nvml.nvmlDeviceGetTemperature(handle, 0))     # NVML_TEMPERATURE_GPU
    except Exception:
        pass
    return out


def price_settings(prefs):
    """(price per kWh or None, currency symbol, the rest of the PC's watts) from Preferences."""
    def num(k):
        try:
            v = float(str(prefs.get(k) or "").strip().replace(",", "."))
            return v if v >= 0 else None
        except ValueError:
            return None
    return num("power_price"), str(prefs.get("power_currency") or "$").strip()[:4] or "$", num("power_extra_watts") or 0.0
