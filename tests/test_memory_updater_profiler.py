from starrygl.module.memorys import _MemoryUpdaterPhaseProfiler


def test_memory_updater_phase_profiler_records_and_resets(monkeypatch):
    monkeypatch.setattr('torch.cuda.is_available', lambda: False)
    profiler = _MemoryUpdaterPhaseProfiler(True, cuda_warmup=0, cuda_stride=1)

    profiler.begin_call()
    token = profiler.start('raw_memory_update')
    profiler.stop('raw_memory_update', token)

    stats = profiler.stats()
    assert stats['calls'] == 1
    assert stats['phases']['raw_memory_update']['host_calls'] == 1
    assert stats['phases']['raw_memory_update']['cuda_sample_calls'] == 0

    profiler.reset()
    assert profiler.stats()['calls'] == 0
