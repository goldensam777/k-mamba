#!/usr/bin/env python3
"""
plot_paper.py — Generate all 8 figures for the k-mamba arXiv paper.

Usage:
    python scripts/plot_paper.py [--bench PATH] [--outdir figures/] [--fig N]

Runs bench_paper for each data point, collects JSON, plots PDF figures.
"""

import argparse
import json
import os
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Wong 2011 colorblind-safe palette ──────────────────────────────
WONG = {
    "black":      "#000000",
    "orange":     "#E69F00",
    "skyblue":    "#56B4E9",
    "green":      "#009E73",
    "yellow":     "#F0E442",
    "blue":       "#0072B2",
    "vermillion": "#D55E00",
    "purple":     "#CC79A7",
}
C_CPU     = WONG["blue"]
C_GPU     = WONG["vermillion"]
C_BLEL    = WONG["green"]
C_SEQ     = WONG["orange"]
C_SCALAR  = WONG["black"]
C_CAVX    = WONG["skyblue"]
C_ASM     = WONG["vermillion"]
C_REF     = WONG["purple"]

# ── Matplotlib defaults for paper ──────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.figsize": (3.5, 2.5),
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

# ── Helpers ────────────────────────────────────────────────────────

def run_bench(bench_path, fig, L=None, D=None, M=None,
              repeat=100, warmup=20):
    """Run bench_paper and return parsed JSON."""
    cmd = [bench_path, "--fig", str(fig),
           "--repeat", str(repeat), "--warmup", str(warmup)]
    if L is not None: cmd += ["--L", str(L)]
    if D is not None: cmd += ["--D", str(D)]
    if M is not None: cmd += ["--M", str(M)]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        print(f"  WARN: bench_paper failed: {result.stderr.strip()}", file=sys.stderr)
        return None
    try:
        return json.loads(result.stdout.strip())
    except json.JSONDecodeError:
        print(f"  WARN: bad JSON: {result.stdout[:200]}", file=sys.stderr)
        return None


def get_median(data, key):
    """Extract median_ms from results dict."""
    return data["results"][key]["median_ms"]


def get_stats(data, key):
    """Extract (median, p5, p95) from results dict."""
    r = data["results"][key]
    return r["median_ms"], r["p5_ms"], r["p95_ms"]


def save_fig(fig_obj, outdir, name):
    path = os.path.join(outdir, name)
    fig_obj.savefig(path)
    plt.close(fig_obj)
    print(f"  -> {path}")


# ═══════════════════════════════════════════════════════════════════
# Figure 1 — Scan1D: CPU ASM vs GPU
# ═══════════════════════════════════════════════════════════════════

def plot_fig1(bench, outdir):
    print("Fig 1: Scan1D CPU vs GPU")
    Ls = [64, 128, 256, 512, 1024, 2048, 4096]
    D, M = 64, 8

    cpu_med, gpu_med = [], []
    cpu_err, gpu_err = [], []
    valid_Ls = []

    for L in Ls:
        data = run_bench(bench, 1, L=L, D=D, M=M)
        if not data:
            continue
        valid_Ls.append(L)

        med, p5, p95 = get_stats(data, "cpu_asm")
        cpu_med.append(med)
        cpu_err.append([med - p5, p95 - med])

        if "gpu" in data["results"]:
            med, p5, p95 = get_stats(data, "gpu")
            gpu_med.append(med)
            gpu_err.append([med - p5, p95 - med])
        else:
            gpu_med.append(None)
            gpu_err.append([0, 0])

    fig, ax = plt.subplots()
    x = np.arange(len(valid_Ls))

    ax.errorbar(x, cpu_med, yerr=np.array(cpu_err).T,
                fmt="o-", color=C_CPU, label="CPU (ASM)", capsize=3, ms=4)

    if any(v is not None for v in gpu_med):
        gm = [v if v is not None else 0 for v in gpu_med]
        ge = [e if gpu_med[i] is not None else [0, 0]
              for i, e in enumerate(gpu_err)]
        ax.errorbar(x, gm, yerr=np.array(ge).T,
                    fmt="s-", color=C_GPU, label="GPU (Blelloch)", capsize=3, ms=4)

    ax.set_xticks(x)
    ax.set_xticklabels(valid_Ls)
    ax.set_xlabel("Sequence length L")
    ax.set_ylabel("Time (ms)")
    ax.set_title("Scan1D Forward: CPU vs GPU")
    ax.legend()
    ax.set_yscale("log")
    save_fig(fig, outdir, "fig1_scan1d_cpu_gpu.pdf")


# ═══════════════════════════════════════════════════════════════════
# Figure 2 — Blelloch vs Sequential GPU
# ═══════════════════════════════════════════════════════════════════

def plot_fig2(bench, outdir):
    print("Fig 2: Blelloch vs Sequential GPU")
    Ls = [64, 128, 256, 512, 1024]
    D, M = 64, 8

    seq_med, blel_med = [], []
    seq_err, blel_err = [], []
    valid_Ls = []

    for L in Ls:
        data = run_bench(bench, 2, L=L, D=D, M=M)
        if not data or "error" in data:
            continue
        valid_Ls.append(L)

        med, p5, p95 = get_stats(data, "sequential")
        seq_med.append(med)
        seq_err.append([med - p5, p95 - med])

        if "blelloch" in data["results"]:
            med, p5, p95 = get_stats(data, "blelloch")
            blel_med.append(med)
            blel_err.append([med - p5, p95 - med])

    if not valid_Ls:
        print("  SKIP: no CUDA data")
        return

    fig, ax = plt.subplots()
    x = np.arange(len(valid_Ls))

    ax.errorbar(x, seq_med, yerr=np.array(seq_err).T,
                fmt="o-", color=C_SEQ, label="Sequential", capsize=3, ms=4)
    if blel_med:
        ax.errorbar(x[:len(blel_med)], blel_med, yerr=np.array(blel_err).T,
                    fmt="s-", color=C_BLEL, label="Blelloch", capsize=3, ms=4)

    ax.set_xticks(x)
    ax.set_xticklabels(valid_Ls)
    ax.set_xlabel("Sequence length L")
    ax.set_ylabel("Time (ms)")
    ax.set_title("GPU: Blelloch vs Sequential Scan")
    ax.legend()
    save_fig(fig, outdir, "fig2_blelloch_vs_seq.pdf")


# ═══════════════════════════════════════════════════════════════════
# Figure 3 — Wavefront 2D diagonal widths
# ═══════════════════════════════════════════════════════════════════

def plot_fig3(bench, outdir):
    print("Fig 3: Wavefront 2D diagonals")
    configs = [(8, 8), (16, 16), (32, 32), (16, 32)]
    D, M = 16, 4

    fig, axes = plt.subplots(1, len(configs),
                             figsize=(3.5 * len(configs) / 2, 2.5))
    if len(configs) == 1:
        axes = [axes]

    for ax, (d1, d2) in zip(axes, configs):
        data = run_bench(bench, 3, L=d1, D=d2, M=M)
        if not data:
            ax.set_title(f"{d1}x{d2}: no data")
            continue

        widths = data["diag_widths"]
        ax.bar(range(len(widths)), widths, color=C_BLEL, alpha=0.8)
        ax.set_xlabel("Diagonal index")
        ax.set_ylabel("Width")
        ax.set_title(f"{d1}x{d2} (t={data['total_ms']['median_ms']:.2f}ms)")

    fig.suptitle("Wavefront 2D: Diagonal Parallelism", fontsize=10)
    fig.tight_layout()
    save_fig(fig, outdir, "fig3_wavefront_2d.pdf")


# ═══════════════════════════════════════════════════════════════════
# Figure 4 — GEMM Roofline
# ═══════════════════════════════════════════════════════════════════

def plot_fig4(bench, outdir):
    print("Fig 4: GEMM Roofline")
    # Sweep square matrix sizes
    sizes = [32, 64, 128, 256, 512, 1024]

    intensities = []
    gflops_list = []
    peak_bw = None

    for N in sizes:
        data = run_bench(bench, 4, L=N, D=N, M=N)
        if not data:
            continue
        r = data["results"]
        intensities.append(r["arith_intensity"])
        gflops_list.append(r["gflops"])
        if peak_bw is None:
            peak_bw = r["peak_bw_gbs"]

    if not intensities:
        print("  SKIP: no data")
        return

    fig, ax = plt.subplots()

    # Roofline envelope
    if peak_bw:
        x_roof = np.logspace(-1, 3, 200)
        # Estimate peak compute from largest measured
        peak_flops = max(gflops_list) * 1.1
        y_roof = np.minimum(peak_flops, peak_bw * x_roof)
        ax.plot(x_roof, y_roof, "k--", alpha=0.4, label="Roofline bound")

    ax.scatter(intensities, gflops_list, color=C_CPU, s=40, zorder=5)
    for i, N in enumerate(sizes[:len(intensities)]):
        ax.annotate(f"{N}", (intensities[i], gflops_list[i]),
                    textcoords="offset points", xytext=(4, 4), fontsize=7)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Arithmetic intensity (FLOP/byte)")
    ax.set_ylabel("GFLOP/s")
    ax.set_title("GEMM AVX2 Roofline")
    ax.legend(fontsize=7)
    save_fig(fig, outdir, "fig4_gemm_roofline.pdf")


# ═══════════════════════════════════════════════════════════════════
# Figure 5 — Backward speedup by D
# ═══════════════════════════════════════════════════════════════════

def plot_fig5(bench, outdir):
    print("Fig 5: Backward speedup")
    L = 512
    Ds = [8, 16, 32, 64, 128, 256]

    scl_med, cavx_med, asm_med = [], [], []
    valid_Ds = []

    for D in Ds:
        data = run_bench(bench, 5, L=L, D=D)
        if not data:
            continue
        valid_Ds.append(D)
        scl_med.append(get_median(data, "c_scalar"))
        cavx_med.append(get_median(data, "c_avx2"))
        asm_med.append(get_median(data, "asm_avx2"))

    if not valid_Ds:
        print("  SKIP: no data")
        return

    fig, ax = plt.subplots()
    x = np.arange(len(valid_Ds))
    w = 0.25

    ax.bar(x - w, scl_med,  w, color=C_SCALAR, label="C scalar")
    ax.bar(x,     cavx_med, w, color=C_CAVX,   label="C AVX2")
    ax.bar(x + w, asm_med,  w, color=C_ASM,    label="ASM AVX2")

    ax.set_xticks(x)
    ax.set_xticklabels(valid_Ds)
    ax.set_xlabel("State dimension D")
    ax.set_ylabel("Time (ms)")
    ax.set_title(f"Backward Pass (L={L})")
    ax.legend()
    save_fig(fig, outdir, "fig5_backward_speedup.pdf")


# ═══════════════════════════════════════════════════════════════════
# Figure 6 — Scaling throughput
# ═══════════════════════════════════════════════════════════════════

def plot_fig6(bench, outdir):
    print("Fig 6: Scaling throughput")
    Ls = [128, 256, 512, 1024, 2048, 4096]
    D, M = 64, 8

    throughputs = []
    medians = []
    valid_Ls = []

    for L in Ls:
        data = run_bench(bench, 6, L=L, D=D, M=M)
        if not data:
            continue
        valid_Ls.append(L)
        throughputs.append(data["results"]["throughput_gbs"])
        medians.append(get_median(data, "scan1d"))

    if not valid_Ls:
        print("  SKIP: no data")
        return

    fig, ax1 = plt.subplots()

    ax1.plot(valid_Ls, medians, "o-", color=C_CPU, label="Time (ms)", ms=4)
    ax1.set_xlabel("Sequence length L")
    ax1.set_ylabel("Time (ms)", color=C_CPU)
    ax1.tick_params(axis="y", labelcolor=C_CPU)

    ax2 = ax1.twinx()
    ax2.plot(valid_Ls, throughputs, "s--", color=C_GPU, label="Throughput", ms=4)
    ax2.set_ylabel("Throughput (GB/s)", color=C_GPU)
    ax2.tick_params(axis="y", labelcolor=C_GPU)

    ax1.set_title(f"Scan1D Scaling (D={D}, M={M})")
    fig.tight_layout()
    save_fig(fig, outdir, "fig6_scaling.pdf")


# ═══════════════════════════════════════════════════════════════════
# Figure 7 — Comparison vs other libraries
# ═══════════════════════════════════════════════════════════════════

def _try_pytorch_scan(L, D, M, repeat=100, warmup=20):
    """Run a PyTorch mamba-ssm benchmark if available."""
    try:
        import torch
        from mamba_ssm import Mamba
    except ImportError:
        return None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Mamba(d_model=D, d_state=M, d_conv=4, expand=1).to(device)
    x = torch.randn(1, L, D, device=device)

    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        if device == "cuda":
            torch.cuda.synchronize()

        times = []
        for _ in range(repeat):
            if device == "cuda":
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                model(x)
                end.record()
                torch.cuda.synchronize()
                times.append(start.elapsed_time(end))
            else:
                import time
                t0 = time.perf_counter()
                model(x)
                times.append((time.perf_counter() - t0) * 1e3)

    times.sort()
    n = len(times)
    return {
        "median_ms": times[n // 2],
        "p5_ms": times[int(n * 0.05)],
        "p95_ms": times[int(n * 0.95)],
    }


def plot_fig7(bench, outdir):
    print("Fig 7: Comparison vs other libraries")
    Ls = [128, 256, 512, 1024, 2048]
    D, M = 64, 16

    om_cpu, om_ref, om_gpu = [], [], []
    pt_times = []
    valid_Ls = []

    for L in Ls:
        data = run_bench(bench, 7, L=L, D=D, M=M)
        if not data:
            continue
        valid_Ls.append(L)

        om_cpu.append(get_median(data, "optimatrix_cpu_asm"))
        om_ref.append(get_median(data, "optimatrix_cpu_ref"))

        if "optimatrix_gpu" in data["results"]:
            om_gpu.append(get_median(data, "optimatrix_gpu"))
        else:
            om_gpu.append(None)

        pt = _try_pytorch_scan(L, D, M)
        pt_times.append(pt["median_ms"] if pt else None)

    if not valid_Ls:
        print("  SKIP: no data")
        return

    fig, ax = plt.subplots()
    x = np.arange(len(valid_Ls))

    ax.plot(x, om_cpu, "o-", color=C_CPU, label="optimatrix CPU (ASM)", ms=4)
    ax.plot(x, om_ref, "^--", color=C_REF, label="optimatrix CPU (ref)", ms=4)

    if any(v is not None for v in om_gpu):
        gvals = [v if v is not None else float("nan") for v in om_gpu]
        ax.plot(x, gvals, "s-", color=C_GPU, label="optimatrix GPU", ms=4)

    if any(v is not None for v in pt_times):
        pvals = [v if v is not None else float("nan") for v in pt_times]
        ax.plot(x, pvals, "D-", color=WONG["purple"], label="mamba-ssm (PyTorch)", ms=4)

    ax.set_xticks(x)
    ax.set_xticklabels(valid_Ls)
    ax.set_xlabel("Sequence length L")
    ax.set_ylabel("Time (ms)")
    ax.set_title(f"Scan1D Comparison (D={D}, M={M})")
    ax.legend(fontsize=7)
    ax.set_yscale("log")
    save_fig(fig, outdir, "fig7_comparison.pdf")


# ═══════════════════════════════════════════════════════════════════
# Figure 8 — MambaBlock end-to-end
# ═══════════════════════════════════════════════════════════════════

def plot_fig8(bench, outdir):
    print("Fig 8: MambaBlock end-to-end")
    dims = [32, 64, 128, 256, 512]

    medians = []
    errs = []
    valid_dims = []

    for dim in dims:
        data = run_bench(bench, 8, D=dim)
        if not data or "error" in data:
            continue
        valid_dims.append(dim)
        med, p5, p95 = get_stats(data, "kmamba_cpu")
        medians.append(med)
        errs.append([med - p5, p95 - med])

    if not valid_dims:
        print("  SKIP: no data")
        return

    fig, ax = plt.subplots()
    x = np.arange(len(valid_dims))

    ax.errorbar(x, medians, yerr=np.array(errs).T,
                fmt="o-", color=C_CPU, capsize=3, ms=4, label="k-mamba CPU")

    ax.set_xticks(x)
    ax.set_xticklabels(valid_dims)
    ax.set_xlabel("Model dimension")
    ax.set_ylabel("Time (ms)")
    ax.set_title("MambaBlock Forward (seq_len=128)")
    ax.legend()
    save_fig(fig, outdir, "fig8_mambablock.pdf")


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

PLOT_FNS = {
    1: plot_fig1,
    2: plot_fig2,
    3: plot_fig3,
    4: plot_fig4,
    5: plot_fig5,
    6: plot_fig6,
    7: plot_fig7,
    8: plot_fig8,
}


def main():
    parser = argparse.ArgumentParser(description="Generate k-mamba paper figures")
    parser.add_argument("--bench", default="build/bench/bench_paper",
                        help="Path to bench_paper binary")
    parser.add_argument("--outdir", default="figures",
                        help="Output directory for PDFs")
    parser.add_argument("--fig", type=int, default=0,
                        help="Generate only this figure (0 = all)")
    args = parser.parse_args()

    if not os.path.isfile(args.bench):
        # Try common build paths
        for p in ["build/bench/bench_paper",
                   "build-cuda/bench/bench_paper",
                   "cmake-build-release/bench/bench_paper"]:
            if os.path.isfile(p):
                args.bench = p
                break
        else:
            print(f"ERROR: bench_paper not found at '{args.bench}'", file=sys.stderr)
            print("Build with: cmake -B build -DKMAMBA_BUILD_BENCH=ON && cmake --build build",
                  file=sys.stderr)
            sys.exit(1)

    os.makedirs(args.outdir, exist_ok=True)
    print(f"Using bench: {args.bench}")
    print(f"Output dir:  {args.outdir}\n")

    if args.fig:
        if args.fig not in PLOT_FNS:
            print(f"ERROR: fig {args.fig} not in 1-8", file=sys.stderr)
            sys.exit(1)
        PLOT_FNS[args.fig](args.bench, args.outdir)
    else:
        for i in sorted(PLOT_FNS):
            PLOT_FNS[i](args.bench, args.outdir)
            print()

    print("Done.")


if __name__ == "__main__":
    main()
