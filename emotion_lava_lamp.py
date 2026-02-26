from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable, Iterable


VAD = tuple[float, float, float]


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def hsv_to_rgb(h: float, s: float, v: float) -> tuple[float, float, float]:
    h = h % 360.0
    c = v * s
    x = c * (1 - abs((h / 60.0) % 2 - 1))
    m = v - c
    if h < 60:
        rp, gp, bp = c, x, 0
    elif h < 120:
        rp, gp, bp = x, c, 0
    elif h < 180:
        rp, gp, bp = 0, c, x
    elif h < 240:
        rp, gp, bp = 0, x, c
    elif h < 300:
        rp, gp, bp = x, 0, c
    else:
        rp, gp, bp = c, 0, x
    return rp + m, gp + m, bp + m


@dataclass
class TemporalFilter:
    tau_v: float = 2.0
    tau_a: float = 0.6
    tau_d: float = 1.2
    max_step: float = 0.25
    state: VAD = (0.0, 0.0, 0.0)

    def update(self, target: VAD, dt: float) -> VAD:
        current = list(self.state)
        tau = (self.tau_v, self.tau_a, self.tau_d)
        out: list[float] = []
        for i, (cur, t, axis_tau) in enumerate(zip(current, target, tau)):
            alpha = 1 - math.exp(-dt / axis_tau)
            bounded_target = cur + clamp(t - cur, -self.max_step, self.max_step)
            nxt = lerp(cur, bounded_target, alpha)
            out.append(clamp(nxt, -1.0, 1.0))
        self.state = (out[0], out[1], out[2])
        return self.state


@dataclass
class EmotionEnergyModel:
    decay_per_frame: float = 0.995
    energy: float = 0.0

    def update(self, target: VAD, smoothed: VAD) -> float:
        delta = sum(abs(t - s) for t, s in zip(target, smoothed)) / 3.0
        self.energy += delta
        self.energy *= self.decay_per_frame
        self.energy = clamp(self.energy, 0.0, 10.0)
        return self.energy


@dataclass
class Blob:
    position: list[float]
    velocity: list[float]
    radius: float
    color: tuple[float, float, float]


@dataclass
class VisualParams:
    hsv_primary: tuple[float, float, float]
    hsv_secondary: tuple[float, float, float]
    rgb_primary: tuple[float, float, float]
    blob_count: int
    blob_size_mean: float
    surface_tension: float
    viscosity: float
    buoyancy: float
    turbulence: float
    threshold: float
    gravity_x: float


@dataclass
class VisualMapping:
    base_turbulence: float = 0.1
    arousal_gain: float = 0.9
    energy_gain: float = 0.4

    def map(self, vad: VAD, energy: float, t: float) -> VisualParams:
        v, a, d = vad
        nv, na, nd = ((v + 1.0) / 2.0, (a + 1.0) / 2.0, (d + 1.0) / 2.0)

        h = lerp(220.0, 20.0, nv)
        s = 0.3 + 0.7 * na
        val = 0.4 + 0.6 * na
        hue_noise = random.uniform(-24.0, 24.0) * (1.0 - nd)
        h2 = (h + hue_noise) % 360.0

        blob_count = 3 + int(na * 10)
        blob_size_mean = lerp(0.14, 0.05, na)
        surface_tension = lerp(0.2, 1.0, nd)
        viscosity = lerp(1.0, 0.2, nv)
        buoyancy = lerp(-0.3, 0.3, nv)
        turbulence = self.base_turbulence + na * self.arousal_gain + energy * self.energy_gain
        threshold = lerp(1.2, 0.9, nd)

        freq = 0.1 + na * 1.5
        amp = 0.02 + energy * 0.05
        gravity_x = math.sin(t * freq * math.tau) * amp

        hsv_primary = (h, s, val)
        hsv_secondary = (h2, s, val)

        return VisualParams(
            hsv_primary=hsv_primary,
            hsv_secondary=hsv_secondary,
            rgb_primary=hsv_to_rgb(*hsv_primary),
            blob_count=blob_count,
            blob_size_mean=blob_size_mean,
            surface_tension=surface_tension,
            viscosity=viscosity,
            buoyancy=buoyancy,
            turbulence=turbulence,
            threshold=threshold,
            gravity_x=gravity_x,
        )


@dataclass
class FluidSimulation:
    width: float = 1.0
    height: float = 1.0
    damping_base: float = 0.995
    blobs: list[Blob] = field(default_factory=list)

    def reset(self, params: VisualParams, seed: int | None = None) -> None:
        rng = random.Random(seed)
        self.blobs = []
        for _ in range(params.blob_count):
            self.blobs.append(
                Blob(
                    position=[rng.random() * self.width, rng.random() * self.height],
                    velocity=[rng.uniform(-0.05, 0.05), rng.uniform(-0.05, 0.05)],
                    radius=max(0.01, rng.gauss(params.blob_size_mean, 0.015)),
                    color=params.rgb_primary,
                )
            )

    def _curl_noise(self, x: float, y: float, t: float) -> tuple[float, float]:
        nx = math.sin(3.0 * y + 1.7 * t) * 0.5 + math.sin(7.0 * y - 0.6 * t) * 0.5
        ny = math.cos(3.0 * x - 1.3 * t) * 0.5 + math.cos(5.0 * x + 0.8 * t) * 0.5
        return nx, ny

    def step(self, params: VisualParams, nd: float, na: float, dt: float, t: float) -> None:
        if not self.blobs:
            self.reset(params)

        while len(self.blobs) < params.blob_count:
            self.blobs.append(
                Blob(
                    position=[random.random() * self.width, random.random() * self.height],
                    velocity=[0.0, 0.0],
                    radius=params.blob_size_mean,
                    color=params.rgb_primary,
                )
            )
        if len(self.blobs) > params.blob_count:
            self.blobs = self.blobs[: params.blob_count]

        damping = self.damping_base - (1.0 - params.viscosity) * 0.02
        for blob in self.blobs:
            cx, cy = self._curl_noise(blob.position[0], blob.position[1], t)
            blob.velocity[0] += (cx * params.turbulence + params.gravity_x) * dt
            blob.velocity[1] += (cy * params.turbulence + params.buoyancy) * dt
            blob.velocity[0] *= damping
            blob.velocity[1] *= damping
            blob.position[0] = (blob.position[0] + blob.velocity[0] * dt) % self.width
            blob.position[1] = clamp(blob.position[1] + blob.velocity[1] * dt, 0.0, self.height)
            blob.color = params.rgb_primary

        self._merge_and_split(nd, na, params)

    def _merge_and_split(self, nd: float, na: float, params: VisualParams) -> None:
        i = 0
        while i < len(self.blobs):
            j = i + 1
            while j < len(self.blobs):
                a = self.blobs[i]
                b = self.blobs[j]
                dx = a.position[0] - b.position[0]
                dy = a.position[1] - b.position[1]
                dist2 = dx * dx + dy * dy
                threshold = (a.radius + b.radius) * (1.5 - 0.5 * nd)
                if dist2 < threshold * threshold and random.random() < nd:
                    area = a.radius * a.radius + b.radius * b.radius
                    a.radius = math.sqrt(area)
                    a.velocity[0] = (a.velocity[0] + b.velocity[0]) * 0.5
                    a.velocity[1] = (a.velocity[1] + b.velocity[1]) * 0.5
                    self.blobs.pop(j)
                    continue
                j += 1
            i += 1

        idx = 0
        while idx < len(self.blobs):
            b = self.blobs[idx]
            if b.radius > params.blob_size_mean * 1.8 and random.random() < na:
                r = b.radius / math.sqrt(2)
                b.radius = r
                child = Blob(
                    position=[(b.position[0] + 0.03) % self.width, clamp(b.position[1] + 0.03, 0.0, self.height)],
                    velocity=[-b.velocity[0], b.velocity[1]],
                    radius=r,
                    color=b.color,
                )
                self.blobs.append(child)
            idx += 1


@dataclass
class EmotionLavaLampEngine:
    vad_getter: Callable[[], VAD | None]
    filter: TemporalFilter = field(default_factory=TemporalFilter)
    energy_model: EmotionEnergyModel = field(default_factory=EmotionEnergyModel)
    mapper: VisualMapping = field(default_factory=VisualMapping)
    sim: FluidSimulation = field(default_factory=FluidSimulation)
    target_vad: VAD = (0.0, 0.0, 0.0)
    time_s: float = 0.0

    def tick(self, dt: float = 0.016) -> VisualParams:
        sample = self.vad_getter()
        if sample is not None:
            self.target_vad = tuple(clamp(v, -1.0, 1.0) for v in sample)  # type: ignore[assignment]

        smoothed = self.filter.update(self.target_vad, dt)
        energy = self.energy_model.update(self.target_vad, smoothed)
        self.time_s += dt
        params = self.mapper.map(smoothed, energy, self.time_s)
        na = (smoothed[1] + 1.0) / 2.0
        nd = (smoothed[2] + 1.0) / 2.0
        self.sim.step(params, nd=nd, na=na, dt=dt, t=self.time_s)
        return params


def get_global_vad() -> VAD | None:
    """Replace this with the real upstream interface."""
    return None


def run_engine(frames: int = 300, dt: float = 0.016) -> Iterable[VisualParams]:
    engine = EmotionLavaLampEngine(vad_getter=get_global_vad)
    for _ in range(frames):
        yield engine.tick(dt=dt)
