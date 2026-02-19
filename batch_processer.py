#!/usr/bin/env python3
"""
Parallel runner for mlsharp_to_spatial_photo across multiple GPUs.

Usage:
	python run_parallel.py --frames-dir insidious/clip2/frames \
						   --plys-dir insidious/clip2/plys \
						   --num-processes 8 \
						   --num-gpus 2

Each process k handles frames where frame_number % N == k.
Each process gets its own Xvfb virtual display and is pinned to a GPU.

Progress is logged to stdout every --log-interval seconds.
"""

import argparse
import glob
import os
import re
import signal
import subprocess
import sys
import threading
import time
from ctypes import c_double, c_int
from datetime import timedelta
from multiprocessing import Array, Process, Value

class HiddenPrints:
	def __enter__(self):
		self._original_stdout = sys.stdout
		sys.stdout = open(os.devnull, 'w')

	def __exit__(self, exc_type, exc_val, exc_tb):
		sys.stdout.close()
		sys.stdout = self._original_stdout

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_frame_number(path: str) -> int:
	"""Extract the numeric part from a frame filename like frame_045.png."""
	match = re.search(r"(\d+)", os.path.basename(path))
	if not match:
		raise ValueError(f"Could not extract frame number from {path}")
	return int(match.group(1))


def start_xvfb(display_num: int) -> subprocess.Popen:
	"""Start an Xvfb virtual display and return the process handle."""
	cmd = ["Xvfb", f":{display_num}", "-screen", "0", "640x480x24"]
	proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
	time.sleep(0.5)  # Give Xvfb a moment to start
	return proc


def fmt_duration(seconds: float) -> str:
	"""Format seconds into a human-readable HH:MM:SS string."""
	if seconds < 0 or seconds != seconds:  # negative or NaN
		return "??:??:??"
	return str(timedelta(seconds=int(seconds)))


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def worker(
	worker_id: int,
	num_workers: int,
	gpu_id: int,
	display_num: int,
	frames_dir: str,
	plys_dir: str,
	out_stereo_ipds: list,
	# Shared-memory arrays (one slot per worker, indexed by worker_id)
	shared_done: Array,       # c_int    – frames completed by each worker
	shared_total: Array,      # c_int    – total frames assigned to each worker
	shared_start_ts: Value,   # c_double – wall-clock time when processing began
):
	"""
	Worker process: handles every frame where frame_number % num_workers == worker_id.
	Sets DISPLAY and CUDA_VISIBLE_DEVICES env vars before importing/running anything.
	"""
	# Set environment BEFORE any GPU/GL imports
	os.environ["DISPLAY"] = f":{display_num}.0"
	os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

	# Replace this import with your actual module
	from spatial_photos import mlsharp_to_spatial_photo  # noqa: E402

	# Collect and sort all frame paths, then pick this worker's subset
	frame_paths = sorted(
		glob.glob(os.path.join(frames_dir, "*.png")),
		key=get_frame_number,
	)
	my_frames = [p for p in frame_paths if get_frame_number(p) % num_workers == worker_id]

	shared_total[worker_id] = len(my_frames)
	shared_done[worker_id] = 0

	print(
		f"[Worker {worker_id:02d}] GPU={gpu_id}  DISPLAY=:{display_num}.0  "
		f"Frames assigned: {len(my_frames)}",
		flush=True,
	)

	for i, frame_path in enumerate(my_frames):
		frame_num = get_frame_number(frame_path)
		frame_stem = os.path.splitext(os.path.basename(frame_path))[0]

		ply_path = os.path.join(plys_dir, f"{frame_stem}.ply")
		if not os.path.exists(ply_path):
			print(
				f"[Worker {worker_id:02d}] WARNING: PLY not found for {frame_path}, skipping.",
				flush=True,
			)
			shared_done[worker_id] += 1
			continue

		out_stereo = [
			(ipd, f"{frame_stem}_ipd_{int(ipd * 1000):03d}.png")
			for ipd in out_stereo_ipds
		]

		t0 = time.time()
		try:
			mlsharp_to_spatial_photo(
				orgImagePath=frame_path,
				plyPath=ply_path,
				outGLB=None,
				outStereoImages=out_stereo,
			)
		except Exception as e:
			print(f"[Worker {worker_id:02d}] ERROR on frame {frame_num}: {e}", flush=True)

		frame_elapsed = time.time() - t0
		shared_done[worker_id] += 1

		completed = i + 1
		remaining = len(my_frames) - completed
		pct = 100 * completed / len(my_frames)

		# Per-worker ETA using this worker's own throughput so far
		worker_elapsed = time.time() - shared_start_ts.value
		fps_k = completed / worker_elapsed if worker_elapsed > 0 else 0
		eta_k = fmt_duration(remaining / fps_k) if fps_k > 0 else "??:??:??"

	print(f"[Worker {worker_id:02d}] ✓ Done.", flush=True)


# ---------------------------------------------------------------------------
# Progress monitor thread (runs in the main process)
# ---------------------------------------------------------------------------

def progress_monitor(shared_done, shared_total, shared_start_ts, num_workers, log_interval):
	"""Print a progress summary table every log_interval seconds."""
	# Wait briefly so workers have had a chance to set their totals
	time.sleep(2)

	while True:
		now = time.time()
		elapsed = now - shared_start_ts.value

		total_done   = sum(shared_done[k]  for k in range(num_workers))
		total_frames = sum(shared_total[k] for k in range(num_workers))
		remaining    = total_frames - total_done
		pct          = 100 * total_done / total_frames if total_frames > 0 else 0

		# Overall ETA based on aggregate throughput since start
		if elapsed > 0 and total_done > 0:
			fps_all = total_done / elapsed
			eta_all = fmt_duration(remaining / fps_all)
			fps_str = f"{fps_all:.2f} fr/s"
		else:
			eta_all = "??:??:??"
			fps_str = "—"

		sep = "─" * 72
		lines = [
			"",
			sep,
			f"  ▶ OVERALL  {total_done}/{total_frames} frames  ({pct:.1f}%)  "
			f"elapsed={fmt_duration(elapsed)}  ETA={eta_all}  throughput={fps_str}",
			f"  {'Worker':<8} {'Done':>6} {'Total':>6} {'%':>6}  {'Worker ETA':>12}",
			f"  {'------':<8} {'----':>6} {'-----':>6} {'---':>6}  {'----------':>12}",
		]

		for k in range(num_workers):
			done_k  = shared_done[k]
			total_k = shared_total[k]
			pct_k   = 100 * done_k / total_k if total_k > 0 else 0
			rem_k   = total_k - done_k

			# Per-worker ETA: use this worker's own rate
			if elapsed > 0 and done_k > 0:
				fps_k = done_k / elapsed
				eta_k = fmt_duration(rem_k / fps_k) if fps_k > 0 else "??:??:??"
			else:
				eta_k = "??:??:??"

			status = "✓ done" if done_k >= total_k > 0 else eta_k
			lines.append(
				f"  {k:<8} {done_k:>6} {total_k:>6} {pct_k:>5.1f}%  {status:>12}"
			)

		lines.append(sep)
		print("\n".join(lines), flush=True)

		if total_done >= total_frames > 0:
			break

		time.sleep(log_interval)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
	parser = argparse.ArgumentParser(description="Parallel mlsharp_to_spatial_photo runner")
	parser.add_argument("--frames-dir", required=True, help="Directory containing frame PNGs")
	parser.add_argument("--plys-dir",   required=True, help="Directory containing PLY files")
	parser.add_argument("--num-processes", "-n", type=int, required=True,
						help="Total number of worker processes (N)")
	parser.add_argument("--num-gpus", "-g", type=int, required=True,
						help="Number of GPUs (M)")
	parser.add_argument("--base-display", type=int, default=100,
						help="Starting Xvfb display number (default: 100). "
							 "Workers use :base, :base+1, ..., :base+N-1.")
	parser.add_argument("--stereo-ipds", nargs="+", type=float,
						default=[0.064, 0.032, 0.016, 0.008],
						help="IPD values for stereo output images (meters)")
	parser.add_argument("--log-interval", type=float, default=30.0,
						help="Seconds between progress summary logs (default: 30)")
	args = parser.parse_args()

	N = args.num_processes
	M = args.num_gpus

	print(f"Spawning {N} workers across {M} GPUs (~{N/M:.1f} workers per GPU)")
	print(f"Progress summary every {args.log_interval:.0f}s\n")

	# Shared memory — one slot per worker
	shared_done     = Array(c_int,    [0] * N)
	shared_total    = Array(c_int,    [0] * N)
	shared_start_ts = Value(c_double, 0.0)

	# Start one Xvfb per worker
	xvfb_procs = []
	for k in range(N):
		disp = args.base_display + k
		print(f"  Starting Xvfb :{disp} for worker {k} ...")
		xvfb_procs.append(start_xvfb(disp))

	# Record start time, then spawn workers
	shared_start_ts.value = time.time()

	workers = []
	for k in range(N):
		gpu_id      = k % M
		display_num = args.base_display + k
		p = Process(
			target=worker,
			args=(
				k, N, gpu_id, display_num,
				args.frames_dir, args.plys_dir, args.stereo_ipds,
				shared_done, shared_total, shared_start_ts,
			),
			daemon=True,
		)
		p.start()
		workers.append(p)
		print(f"  Worker {k:02d} started  PID={p.pid}  GPU={gpu_id}  DISPLAY=:{display_num}.0")

	print()

	# Progress monitor runs as a background thread in the main process
	monitor = threading.Thread(
		target=progress_monitor,
		args=(shared_done, shared_total, shared_start_ts, N, args.log_interval),
		daemon=True,
	)
	monitor.start()

	# Graceful shutdown on Ctrl-C / SIGTERM
	def shutdown(sig, frame):
		print("\nInterrupted — terminating workers and Xvfb instances...")
		for p in workers:
			p.terminate()
		for x in xvfb_procs:
			x.terminate()
		sys.exit(1)

	signal.signal(signal.SIGINT, shutdown)
	signal.signal(signal.SIGTERM, shutdown)

	for p in workers:
		p.join()

	# Clean up Xvfb
	for x in xvfb_procs:
		x.terminate()

	# Final summary
	total_done   = sum(shared_done[k]  for k in range(N))
	total_frames = sum(shared_total[k] for k in range(N))
	elapsed      = time.time() - shared_start_ts.value
	print(f"\n✓ All workers finished.  {total_done}/{total_frames} frames in {fmt_duration(elapsed)}.")


if __name__ == "__main__":
	main()