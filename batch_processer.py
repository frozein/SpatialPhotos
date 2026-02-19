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

# ------------------------------------------- #

def get_frame_number(path: str) -> int:
	match = re.search(r"(\d+)", os.path.basename(path))
	if not match:
		raise ValueError(f"Could not extract frame number from {path}")

	return int(match.group(1))


def start_xvfb(displayId: int) -> subprocess.Popen:
	cmd = ["Xvfb", f":{displayId}", "-screen", "0", "640x480x24"]
	proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
	time.sleep(0.5)

	return proc


def fmt_duration(seconds: float) -> str:
	if seconds < 0 or seconds != seconds:
		return "??:??:??"

	return str(timedelta(seconds=int(seconds)))

# ------------------------------------------- #

def worker(
	workerId: int,
	numWorkers: int,
	gpuId: int,
	displayId: int,
	framesDir: str,
	plysDir: str,
	outStereoIpds: list,
	outDir: str,
	
	sharedDone: Array,
	sharedTotal: Array,
	sharedStartTimes: Value,
):
	# set environment variables, import processing script:
	# ---------------
	os.environ["DISPLAY"] = f":{displayId}.0"
	os.environ["CUDA_VISIBLE_DEVICES"] = str(gpuId)

	from spatial_photos import mlsharp_to_spatial_photo

	# get frames to process:
	# ---------------
	framePaths = sorted(
		glob.glob(os.path.join(framesDir, "*.png")),
		key=get_frame_number,
	)
	myFrames = [p for p in framePaths if get_frame_number(p) % numWorkers == workerId]

	sharedTotal[workerId] = len(myFrames)
	sharedDone[workerId] = 0

	print(
		f"[Worker {workerId:03d}] GPU={gpuId}  DISPLAY=:{displayId}.0  "
		f"Frames assigned: {len(myFrames)}",
		flush=True,
	)

	# process each frame:
	# ---------------
	for i, framePath in enumerate(myFrames):
		frameNum = get_frame_number(framePath)
		frameStem = os.path.splitext(os.path.basename(framePath))[0]

		plyPath = os.path.join(plysDir, f"{frameStem}.ply")
		if not os.path.exists(plyPath):
			print(
				f"[Worker {workerId:03d}] WARNING: PLY not found for {framePath}, skipping.",
				flush=True,
			)

			sharedDone[workerId] += 1
			continue

		outStereo = [
			(ipd, os.path.join(outDir, f"ipd_{int(ipd * 1000):03d}", f"frame_{frameNum:06d}.png"))
			for ipd in outStereoIpds
		]

		t0 = time.time()
		try:
			mlsharp_to_spatial_photo(
				orgImagePath=framePath,
				plyPath=plyPath,
				outGLB=None,
				outStereoImages=outStereo,
			)
		except Exception as e:
			print(f"[Worker {workerId:03d}] ERROR on frame {frameNum}: {e}", flush=True)

		sharedDone[workerId] += 1

		completed = i + 1
		remaining = len(myFrames) - completed
		pct = 100 * completed / len(myFrames)

		workerElapsed = time.time() - sharedStartTimes.value
		fpsK = completed / workerElapsed if workerElapsed > 0 else 0
		etaK = fmt_duration(remaining / fpsK) if fpsK > 0 else "??:??:??"

	print(f"[Worker {workerId:03d}] Done.", flush=True)

# ------------------------------------------- #

def progress_monitor(sharedDone, sharedTotal, sharedStartTimes, numWorkers, log_interval):
	time.sleep(2)
	
	while True:
		now = time.time()
		elapsed = now - sharedStartTimes.value

		totalDone   = sum(sharedDone[k]  for k in range(numWorkers))
		totalFrames = sum(sharedTotal[k] for k in range(numWorkers))
		remaining    = totalFrames - totalDone
		pct          = 100 * totalDone / totalFrames if totalFrames > 0 else 0

		if elapsed > 0 and totalDone > 0:
			fpsAll = totalDone / elapsed
			etaAll = fmt_duration(remaining / fpsAll)
			fpsStr = f"{fpsAll:.2f} fr/s"
		else:
			etaAll = "??:??:??"
			fpsStr = "—"

		sep = "─" * 72
		lines = [
			"",
			sep,
			f"  ▶ OVERALL  {totalDone}/{totalFrames} frames  ({pct:.1f}%)  "
			f"elapsed={fmt_duration(elapsed)}  ETA={etaAll}  throughput={fpsStr}",
			f"  {'Worker':<8} {'Done':>6} {'Total':>6} {'%':>6}  {'Worker ETA':>12}",
			f"  {'------':<8} {'----':>6} {'-----':>6} {'---':>6}  {'----------':>12}",
		]

		for k in range(numWorkers):
			doneK  = sharedDone[k]
			totalK = sharedTotal[k]
			pctK   = 100 * doneK / totalK if totalK > 0 else 0
			remK   = totalK - doneK

			if elapsed > 0 and doneK > 0:
				fpsK = doneK / elapsed
				etaK = fmt_duration(remK / fpsK) if fpsK > 0 else "??:??:??"
			else:
				etaK = "??:??:??"

			status = "✓ done" if doneK >= totalK > 0 else etaK
			lines.append(
				f"  {k:<8} {doneK:>6} {totalK:>6} {pctK:>5.1f}%  {status:>12}"
			)

		lines.append(sep)
		print("\n".join(lines), flush=True)

		if totalDone >= totalFrames > 0:
			break

		time.sleep(log_interval)

# ------------------------------------------- #

def main():

	# setup argparse:
	# ---------------
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
	parser.add_argument("--out-dir", required=True,
						help="Output directory. Subfolders ipd_064, ipd_032, etc. will be created inside.")
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

	# create output folders:
	# ---------------
	for ipd in args.stereo_ipds:
		folder = os.path.join(args.out_dir, f"ipd_{int(ipd * 1000):03d}")
		os.makedirs(folder, exist_ok=True)
		print(f"  Output folder: {folder}")

	# create shared memory:
	# ---------------
	sharedDone     = Array(c_int,    [0] * N)
	sharedTotal    = Array(c_int,    [0] * N)
	sharedStartTimes = Value(c_double, 0.0)

	# start xvfb for each process:
	# ---------------
	xvfb_procs = []
	for k in range(N):
		disp = args.base_display + k
		print(f"  Starting Xvfb :{disp} for worker {k} ...")
		xvfb_procs.append(start_xvfb(disp))

	# start each worker:
	# ---------------
	sharedStartTimes.value = time.time()

	workers = []
	for k in range(N):
		gpuId      = k % M
		displayId = args.base_display + k
		p = Process(
			target=worker,
			args=(
				k, N, gpuId, displayId,
				args.frames_dir, args.plys_dir, args.stereo_ipds,
				args.out_dir,
				sharedDone, sharedTotal, sharedStartTimes,
			),
			daemon=True,
		)

		p.start()
		workers.append(p)
		print(f"  Worker {k:03d} started  PID={p.pid}  GPU={gpuId}  DISPLAY=:{displayId}.0")

	print()

	# start progress monitor:
	# ---------------
	monitor = threading.Thread(
		target=progress_monitor,
		args=(sharedDone, sharedTotal, sharedStartTimes, N, args.log_interval),
		daemon=True,
	)
	monitor.start()

	# graceful shutdown:
	# ---------------
	def shutdown(sig, frame):
		print("\nInterrupted — terminating workers and Xvfb instances...")
		for p in workers:
			p.terminate()
		for x in xvfb_procs:
			x.terminate()
		sys.exit(1)

	signal.signal(signal.SIGINT, shutdown)
	signal.signal(signal.SIGTERM, shutdown)

	# join with each worker, cleanup:
	# ---------------
	for p in workers:
		p.join()

	for x in xvfb_procs:
		x.terminate()

	# final summary:
	# ---------------
	totalDone   = sum(sharedDone[k]  for k in range(N))
	totalFrames = sum(sharedTotal[k] for k in range(N))
	elapsed      = time.time() - sharedStartTimes.value

	print(f"\n✓ All workers finished.  {totalDone}/{totalFrames} frames in {fmt_duration(elapsed)}.")


if __name__ == "__main__":
	main()