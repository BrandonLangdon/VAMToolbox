"""Hardware auto-detection and configuration recommendations."""
from vamtoolbox.util import hardware


def test_detect_system_shape():
    info = hardware.detect_system()
    for key in ("cpu_logical", "cpu_physical", "ram_total_gb", "ram_avail_gb", "gpus", "cuda"):
        assert key in info
    assert info["cpu_logical"] >= 1
    assert info["cpu_physical"] >= 1
    assert isinstance(info["gpus"], list)
    assert isinstance(info["cuda"], bool)


def test_recommend_cpu_small_box():
    info = {"cpu_logical": 4, "cpu_physical": 4, "ram_total_gb": 16,
            "ram_avail_gb": 8, "gpus": [], "cuda": False}
    rec = hardware.recommend_config(info)
    assert rec["use_cuda"] is False
    assert rec["rebin_jobs"] == -1                 # <=8 cores -> all cores
    assert rec["vram_budget_bytes"] is None
    assert rec["ram_budget_gb"] == 8


def test_recommend_leaves_cores_on_big_box():
    info = {"cpu_logical": 32, "cpu_physical": 16, "ram_total_gb": 64,
            "ram_avail_gb": 40, "gpus": [], "cuda": False}
    assert hardware.recommend_config(info)["rebin_jobs"] == 30   # 32 - 2


def test_recommend_gpu():
    info = {"cpu_logical": 16, "cpu_physical": 8, "ram_total_gb": 64, "ram_avail_gb": 40,
            "gpus": [{"name": "X", "vram_total_gb": 12, "vram_free_gb": 10}], "cuda": True}
    rec = hardware.recommend_config(info)
    assert rec["use_cuda"] is True
    assert rec["vram_budget_bytes"] is not None and rec["vram_budget_bytes"] > 0


def test_autoconfigure_no_apply_runs():
    info, rec = hardware.autoconfigure(apply=False, verbose=False)
    assert "use_cuda" in rec
    assert "rebin_jobs" in rec
    assert isinstance(hardware.summary(info, rec), str)
