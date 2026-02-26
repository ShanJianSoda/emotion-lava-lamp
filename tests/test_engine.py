import math

from emotion_lava_lamp import EmotionLavaLampEngine, TemporalFilter


def test_temporal_filter_smooths_step_change():
    filt = TemporalFilter()
    start = filt.update((1.0, 1.0, 1.0), 0.016)
    for _ in range(120):
        end = filt.update((1.0, 1.0, 1.0), 0.016)
    assert start[1] < end[1] < 1.0
    assert end[1] > start[1]


def test_engine_handles_none_and_keeps_state():
    values = iter([(0.8, -0.6, 0.4), None, None])
    engine = EmotionLavaLampEngine(vad_getter=lambda: next(values, None))

    p1 = engine.tick()
    p2 = engine.tick()
    p3 = engine.tick()

    assert p1.blob_count >= 3
    assert p2.blob_count == p3.blob_count
    assert engine.energy_model.energy >= 0.0


def test_arousal_changes_blob_count_and_turbulence():
    low = EmotionLavaLampEngine(vad_getter=lambda: (-0.2, -1.0, 0.1))
    high = EmotionLavaLampEngine(vad_getter=lambda: (-0.2, 1.0, 0.1))
    for _ in range(180):
        lp = low.tick()
        hp = high.tick()

    assert hp.blob_count > lp.blob_count
    assert hp.turbulence > lp.turbulence
    assert all(math.isfinite(v) for v in hp.rgb_primary)
