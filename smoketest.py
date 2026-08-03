"""smoketest.py — verify the backbone .tflite works in isolation
and measure pure invoke latency. No sensor / Flask / multiprocessing —
just the model.

Two things this catches:

1. Silent NPU corruption. A float32 .tflite loaded through libteflon.so
   doesn't error and doesn't warn — it returns bit-identical garbage for
   every input. Pushing zeros/ones/random and checking the outputs
   actually differ is the cheapest way to confirm the model is honest.

2. NPU vs CPU latency. We time N invokes and report min/mean/max, plus
   (optionally) a CPU run for comparison. Use this to ground-truth what
   "the NPU can do" before chasing inference throughput at the integration
   layer.

Usage:
    python smoketest.py                   # NPU (fall back to CPU)
    FORCE_CPU=1 python smoketest.py       # CPU only
    python smoketest.py --runs 500        # more samples for stable timing
    python smoketest.py --compare         # run both NPU and CPU, print both
"""

import argparse
import os
import sys
import time

import numpy as np


INT8_BACKBONE_PATH    = "output/vibration_backbone_int8.tflite"
FLOAT_BACKBONE_PATH   = "vibration_backbone.tflite"
DEFAULT_DELEGATE_PATH = "/usr/lib/libethosu_delegate.so"


def resolve_model_path():
    here = os.path.dirname(os.path.abspath(__file__))
    for candidate in (INT8_BACKBONE_PATH, FLOAT_BACKBONE_PATH):
        path = os.path.join(here, candidate)
        if os.path.exists(path):
            return path
    sys.exit(f"no backbone .tflite found in {here} "
             f"(looked for {INT8_BACKBONE_PATH} and {FLOAT_BACKBONE_PATH})")


def build_interpreter(model_path, use_npu):
    """Returns (interpreter, mode_str). mode_str is 'npu' or 'cpu'."""
    try:
        from tflite_runtime.interpreter import Interpreter, load_delegate
    except ImportError as e:
        sys.exit(f"tflite_runtime not available: {e}")

    if use_npu and os.path.exists(DEFAULT_DELEGATE_PATH):
        try:
            delegate = load_delegate(DEFAULT_DELEGATE_PATH)
            interp = Interpreter(model_path=model_path,
                                 experimental_delegates=[delegate])
            interp.allocate_tensors()
            return interp, "npu"
        except Exception as e:
            print(f"  NPU delegate failed ({e}); falling back to CPU")

    interp = Interpreter(model_path=model_path)
    interp.allocate_tensors()
    return interp, "cpu"


def quantize_input(window, inp_detail):
    """Float32 window → quantized tensor matching the model's expected dtype.
    Mirrors inference.py:embed() so the smoketest measures the same path."""
    if inp_detail["dtype"] == np.int8:
        scale, zero_point = inp_detail["quantization"]
        x = np.clip(np.round(window / scale + zero_point), -128, 127).astype(np.int8)
    elif inp_detail["dtype"] == np.uint8:
        scale, zero_point = inp_detail["quantization"]
        x = np.clip(np.round(window / scale + zero_point), 0, 255).astype(np.uint8)
    else:
        x = window.astype(inp_detail["dtype"])
    return x.reshape(inp_detail["shape"])


def dequantize_output(y_raw, out_detail):
    if out_detail["dtype"] in (np.int8, np.uint8):
        o_scale, o_zero_point = out_detail["quantization"]
        return (y_raw.astype(np.float32) - o_zero_point) * o_scale
    return y_raw.astype(np.float32)


def invoke_once(interp, inp, out, window_f32):
    """One full embed pass (quantize + set + invoke + get + dequantize)."""
    x = quantize_input(window_f32, inp)
    interp.set_tensor(inp["index"], x)
    interp.invoke()
    y_raw = interp.get_tensor(out["index"])
    return dequantize_output(y_raw, out)


def time_invokes(interp, inp, out, runs):
    """Returns a list of per-invoke seconds. Skips the first invoke as
    warm-up — first call on the NPU often eats setup cost."""
    feat_shape = (inp["shape"][1], inp["shape"][2])
    # Warm-up
    invoke_once(interp, inp, out, np.random.uniform(-1, 1, feat_shape).astype(np.float32))
    times = []
    for _ in range(runs):
        w = np.random.uniform(-1, 1, feat_shape).astype(np.float32)
        t0 = time.perf_counter()
        invoke_once(interp, inp, out, w)
        times.append(time.perf_counter() - t0)
    return times


def differ_check(interp, inp, out):
    """Push three very different inputs and confirm outputs actually change.
    Returns True if outputs differ pairwise; False if the model is returning
    identical embeddings for distinct inputs (silent failure)."""
    feat_shape = (inp["shape"][1], inp["shape"][2])
    zeros = np.zeros(feat_shape, dtype=np.float32)
    ones  = np.ones(feat_shape, dtype=np.float32)
    rand  = np.random.uniform(-1, 1, feat_shape).astype(np.float32)

    f_zeros = invoke_once(interp, inp, out, zeros)
    f_ones  = invoke_once(interp, inp, out, ones)
    f_rand  = invoke_once(interp, inp, out, rand)

    d_zo = float(np.linalg.norm(f_zeros - f_ones))
    d_zr = float(np.linalg.norm(f_zeros - f_rand))
    d_or = float(np.linalg.norm(f_ones  - f_rand))
    print(f"  ‖zeros - ones‖  = {d_zo:.4f}")
    print(f"  ‖zeros - random‖= {d_zr:.4f}")
    print(f"  ‖ones  - random‖= {d_or:.4f}")
    return d_zo > 1e-3 and d_zr > 1e-3 and d_or > 1e-3


def report(times, mode):
    a = np.array(times) * 1000.0    # ms
    print(f"  runs={len(a)}  min={a.min():.2f}ms  mean={a.mean():.2f}ms  "
          f"p50={np.percentile(a, 50):.2f}ms  p95={np.percentile(a, 95):.2f}ms  "
          f"max={a.max():.2f}ms")
    print(f"  throughput={1000.0 / a.mean():.1f} inferences/sec ({mode})")


def run_one(model_path, use_npu, runs):
    print(f"\n=== loading {os.path.basename(model_path)} "
          f"({'NPU' if use_npu else 'CPU'} requested) ===")
    interp, mode = build_interpreter(model_path, use_npu)
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    print(f"  mode={mode}  input shape={inp['shape']} dtype={inp['dtype'].__name__}  "
          f"output shape={out['shape']} dtype={out['dtype'].__name__}")
    print(f"  input quant={inp.get('quantization')}  output quant={out.get('quantization')}")

    print("\n  differ-check (zero / ones / random → outputs must differ):")
    if differ_check(interp, inp, out):
        print("  PASS: outputs differ across inputs")
    else:
        print("  FAIL: outputs are identical or nearly so — model is silent-failing")
        if mode == "npu":
            print("        most common cause: .tflite isn't int8-quantized end-to-end.")
            print("        re-run with FORCE_CPU=1 to confirm the model works on CPU.")

    print(f"\n  timing ({runs} invokes after 1 warm-up):")
    times = time_invokes(interp, inp, out, runs)
    report(times, mode)
    return mode, times


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=int, default=200,
                        help="number of timed invokes (default 200)")
    parser.add_argument("--compare", action="store_true",
                        help="run both NPU and CPU back-to-back and print both")
    args = parser.parse_args()

    model_path = resolve_model_path()
    force_cpu = os.environ.get("FORCE_CPU") == "1"

    if args.compare:
        run_one(model_path, use_npu=True,  runs=args.runs)
        run_one(model_path, use_npu=False, runs=args.runs)
    else:
        run_one(model_path, use_npu=(not force_cpu), runs=args.runs)


if __name__ == "__main__":
    main()
