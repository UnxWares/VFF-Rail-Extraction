"""GPU utilisation cap by duty cycle (no administrator rights needed).

After each batch, the GPU compute time of the batch is measured (synchronisation) and
the process sleeps `busy × (100 / target - 1) × factor`. The factor is readjusted every
few seconds on the measured utilisation (torch.cuda.utilization() via NVML,
otherwise `nvidia-smi`): above target + 3 points it increases, below
target - 8 it decreases. Memory: per-process cap (fraction of the VRAM).
"""

from __future__ import annotations

import subprocess
import time

import torch


def limit_memory(max_gb: float | None):
    if max_gb and torch.cuda.is_available():
        total = torch.cuda.get_device_properties(0).total_memory / 2**30
        torch.cuda.set_per_process_memory_fraction(min(1.0, max_gb / total))


def gpu_util() -> float | None:
    try:
        return float(torch.cuda.utilization())
    except Exception:
        pass
    try:
        out = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
                             capture_output=True, text=True, timeout=5).stdout
        return float(out.strip().splitlines()[0])
    except Exception:
        return None


class GpuThrottle:
    def __init__(self, max_util: float = 80.0, period_s: float = 5.0, log=print):
        self.target = max_util
        self.on = torch.cuda.is_available() and 0 < max_util < 100
        self.factor = 1.0
        self.period = period_s
        self.last_check = time.monotonic()
        self.t0 = None
        self.samples = []
        self.log = log

    def start(self):
        if self.on:
            torch.cuda.synchronize()
            self.t0 = time.monotonic()

    def end(self):
        if not self.on or self.t0 is None:
            return
        torch.cuda.synchronize()
        # Capped: a container freeze (podman pause) in the middle of a batch must not impose an excessive pause.
        busy = min(time.monotonic() - self.t0, 5.0)
        time.sleep(busy * (100.0 / self.target - 1.0) * self.factor)
        now = time.monotonic()
        if now - self.last_check >= self.period:
            u = gpu_util()
            if u is not None:
                self.samples.append(u)
                if u > self.target + 3:
                    self.factor = min(self.factor * 1.25, 8.0)
                elif u < self.target - 8:
                    self.factor = max(self.factor / 1.15, 0.2)
            self.last_check = now

    def summary(self) -> str:
        if not self.samples:
            return 'GPU utilisation not measured'
        s = sorted(self.samples)
        return f'measured GPU utilisation: median {s[len(s) // 2]:.0f} %, max {s[-1]:.0f} % (target {self.target:.0f} %, factor {self.factor:.2f})'
