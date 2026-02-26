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



- Replace `get_global_vad()` in `emotion_lava_lamp.py` with your upstream source.
- Create an `EmotionLavaLampEngine` and call `tick(dt)` each frame.
- Read `engine.sim.blobs` + returned `VisualParams` to drive your renderer/shader.
