"""
Example: how a GUI integrates with VAMToolbox via the high-level pipeline API.

The GUI never touches projectors, optimizers, or environment variables — it only
builds a PrintConfig (from form fields), runs a VAMPipeline on a worker thread, and
reacts to progress callbacks.  Run this file directly for a console demo:

    python examples/gui_integration_example.py path/to/model.stl
"""
import sys
import threading

import vamtoolbox
from vamtoolbox.pipeline import PrintConfig, VAMPipeline, PipelineCancelled


def main(stl_path):
    # 1) Detect hardware (show it in the GUI's "system" panel).
    info = vamtoolbox.pipeline.detect_hardware()
    print("Hardware:", info["cpu_logical"], "cores,",
          info["ram_avail_gb"], "GB free,",
          (info["gpus"][0]["name"] if info["gpus"] else "no GPU"))

    # 2) Build a config from "form fields" (this is what the GUI collects).
    form = {
        "stl_path": stl_path,
        "part_height_mm": 25.4,
        "voxel_pitch_um": 80.0,
        "method": "BCLP",          # OSMO | BCLP
        "n_iterations": 10,
        "absorption": True,
        "diffusion": True,         # BCLP only
        "resolution_scale": 1.0,
        "slab": "auto",            # auto | off | "<int>"
        "low_memory": False,
    }
    cfg = PrintConfig.from_dict(form)
    cfg.validate()                 # raises ValueError -> GUI shows the message

    pipe = VAMPipeline(cfg)
    pipe.apply_hardware()          # auto-tune GPU/RAM/CPU (fills use_cuda, rebin_jobs)

    # 3) Show an ETA before starting (drive a determinate progress bar).
    print("Estimated optimize time:", pipe.estimate_optimize_time()["pretty"])

    # 4) Progress callback -> update a progress bar / status label.
    #    Stages: hardware, voxelize, optimize (per-iteration), rebin, video, done.
    def on_progress(stage, frac, msg):
        print(f"  [{stage:9s}] {frac * 100:5.1f}%  {msg}")
    pipe.on_progress = on_progress

    # 5) Run on a worker thread so the GUI event loop stays responsive.
    #    Call pipe.cancel() from a Cancel button; run() raises PipelineCancelled.
    result_box = {}

    def worker():
        try:
            result_box["result"] = pipe.run()       # voxelize + optimize + rebin
        except PipelineCancelled:
            result_box["cancelled"] = True
        except ValueError as exc:                    # e.g. part wider than vial
            result_box["error"] = str(exc)

    t = threading.Thread(target=worker)
    t.start()
    t.join()

    # 6) Use the structured result.
    if "result" in result_box:
        r = result_box["result"]
        print("Done. timing:", {k: round(v, 1) for k, v in r.timing.items()})
        print("quality:", {k: round(v, 3) for k, v in r.quality.items()})
        pipe.save_video("print_sequence.mp4")        # printer-ready projection video
        print("Wrote print_sequence.mp4; sinogram arrays are in r.sinogram / r.rebinned_sinogram")
    elif result_box.get("cancelled"):
        print("Cancelled by user.")
    else:
        print("Error:", result_box.get("error"))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python examples/gui_integration_example.py <model.stl>")
        sys.exit(1)
    main(sys.argv[1])
