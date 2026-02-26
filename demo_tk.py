from __future__ import annotations

import argparse
import math
import random
import tkinter as tk
from typing import Callable

from emotion_lava_lamp import EmotionLavaLampEngine, VAD, clamp


class VADSignal:
    """Simple built-in VAD sources for local visualization."""

    def __init__(self, mode: str = "sine") -> None:
        self.mode = mode
        self.t = 0.0
        self.step_state: VAD = (-0.8, -0.6, -0.6)
        self.next_flip = 1.5

    def __call__(self) -> VAD:
        dt = 0.016
        self.t += dt

        if self.mode == "sine":
            v = math.sin(self.t * 0.5)
            a = math.sin(self.t * 1.4)
            d = math.sin(self.t * 0.8 + 1.2)
            return clamp(v, -1.0, 1.0), clamp(a, -1.0, 1.0), clamp(d, -1.0, 1.0)

        if self.mode == "noise":
            v = math.sin(self.t * 0.25) * 0.5 + random.uniform(-0.5, 0.5)
            a = math.sin(self.t * 0.9) * 0.4 + random.uniform(-0.6, 0.6)
            d = math.sin(self.t * 0.45 + 1.0) * 0.3 + random.uniform(-0.4, 0.4)
            return clamp(v, -1.0, 1.0), clamp(a, -1.0, 1.0), clamp(d, -1.0, 1.0)

        if self.mode == "step":
            if self.t > self.next_flip:
                self.next_flip += 1.5
                self.step_state = tuple(-x for x in self.step_state)  # type: ignore[assignment]
            return self.step_state

        raise ValueError(f"unknown mode: {self.mode}")


class LavaLampApp:
    def __init__(self, vad_getter: Callable[[], VAD], width: int = 540, height: int = 760) -> None:
        self.width = width
        self.height = height
        self.dt = 0.016

        self.engine = EmotionLavaLampEngine(vad_getter=vad_getter)

        self.root = tk.Tk()
        self.root.title("Emotion Lava Lamp (VAD)")
        self.canvas = tk.Canvas(self.root, width=self.width, height=self.height, bg="#0f0f16", highlightthickness=0)
        self.canvas.pack()

        self.label = tk.Label(self.root, text="", anchor="w", justify="left", bg="#0f0f16", fg="#e6e6f0")
        self.label.pack(fill="x")

    def _draw(self) -> None:
        params = self.engine.tick(self.dt)
        blobs = self.engine.sim.blobs

        self.canvas.delete("all")

        # Lamp body
        margin = 48
        self.canvas.create_rectangle(margin, margin, self.width - margin, self.height - margin, outline="#6c6c8a", width=2)

        for blob in blobs:
            x = margin + blob.position[0] * (self.width - margin * 2)
            y = margin + (1.0 - blob.position[1]) * (self.height - margin * 2)
            r = blob.radius * min(self.width, self.height) * 0.35
            rgb = tuple(max(0, min(255, int(c * 255))) for c in blob.color)
            color = f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
            self.canvas.create_oval(x - r, y - r, x + r, y + r, fill=color, outline="")

        s = self.engine.filter.state
        self.label.configure(
            text=(
                f"Smoothed VAD: V={s[0]:+.2f}, A={s[1]:+.2f}, D={s[2]:+.2f}   "
                f"Energy={self.engine.energy_model.energy:.3f}   "
                f"Blobs={len(blobs)}   Turb={params.turbulence:.2f}"
            )
        )

        self.root.after(int(self.dt * 1000), self._draw)

    def run(self) -> None:
        self._draw()
        self.root.mainloop()


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize the emotion lava lamp in a Tkinter window")
    parser.add_argument("--mode", choices=["sine", "noise", "step"], default="sine", help="Built-in VAD test signal")
    args = parser.parse_args()

    signal = VADSignal(mode=args.mode)
    app = LavaLampApp(vad_getter=signal)
    app.run()


if __name__ == "__main__":
    main()
