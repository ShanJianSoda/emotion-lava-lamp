# Emotion Lava Lamp Engine

A Python reference implementation that maps continuous VAD emotion values (`Valence`, `Arousal`, `Dominance`) to a real-time "emotion lava lamp" parameter + simulation system.

## Pipeline

1. `get_global_vad()` polling (can be slower than render loop).
2. Temporal filtering with per-axis time constants.
3. Emotion energy pool with decay for afterglow/wake effects.
4. Visual parameter mapping (color, count, size, viscosity, buoyancy, turbulence, shake).
5. Metaball-oriented fluid state update (movement, merge, split).
6. Renderer-facing output (`VisualParams`) and blob states.

## Run tests

```bash
pytest -q
```

## How to run and see the effect（如何运行并查看效果）

### 1) Start a local visualization window (Tkinter, no third-party deps)

```bash
python demo_tk.py --mode sine
```

Optional modes:

```bash
python demo_tk.py --mode noise
python demo_tk.py --mode step
```

You will see:
- moving/merging/splitting colored blobs
- smoothed VAD values
- current emotion energy, blob count, turbulence

> Note: This needs a desktop GUI environment. If your environment is headless (no display), use step 2 below.

### 2) Headless quick check (no GUI)

```bash
python - <<'PY'
from emotion_lava_lamp import EmotionLavaLampEngine

engine = EmotionLavaLampEngine(vad_getter=lambda: (0.4, 0.9, -0.2))
for i in range(10):
    p = engine.tick()
    print(i, p.blob_count, round(p.turbulence, 3), tuple(round(x, 2) for x in p.rgb_primary))
PY
```

## Integrate with your own VAD source

- Replace `get_global_vad()` in `emotion_lava_lamp.py` with your upstream source.
- Create an `EmotionLavaLampEngine` and call `tick(dt)` each frame.
- Read `engine.sim.blobs` + returned `VisualParams` to drive your renderer/shader.
