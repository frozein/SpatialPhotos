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
from multiprocessing import Array, Process, Value, Lock

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
	slotId: int,
	gpuId: int,
	displayId: int,
	frameBatch: list,       # list of (framePath, plyPath, outStereoImages) for this batch
	framesDir: str,
	plysDir: str,
	outStereoIpds: list,
	outDir: str,
	sharedDone: Array,      # one global counter
	sharedStartTime: Value,
):
	"""
	Processes exactly one batch of frames (up to K), then exits.
	slotId is just used for display/logging — it's the slot index (0..N-1),
	not a persistent worker identity.
	"""
	os.environ["DISPLAY"] = f":{displayId}.0"
	os.environ["CUDA_VISIBLE_DEVICES"] = str(gpuId)

	from spatial_photos import mlsharp_to_spatial_photo

	print(
		f"[Slot {slotId:02d}] GPU={gpuId}  DISPLAY=:{displayId}.0  "
		f"Batch of {len(frameBatch)} frames  "
		f"(frames {get_frame_number(frameBatch[0][0]):06d}–{get_frame_number(frameBatch[-1][0]):06d})",
		flush=True,
	)

	for framePath, plyPath, outStereoImages in frameBatch:
		frameNum = get_frame_number(framePath)

		t0 = time.time()
		try:
			import torch
			with torch.no_grad():
				mlsharp_to_spatial_photo(
					orgImagePath=framePath,
					plyPath=plyPath,
					outGLB=None,
					outStereoImages=outStereoImages,
				)
		except Exception as e:
			print(f"[Slot {slotId:02d}] ERROR on frame {frameNum}: {e}", flush=True)

		elapsed = time.time() - t0
		sharedDone[0] += 1
		done = sharedDone[0]

		workerElapsed = time.time() - sharedStartTime.value
		fps = done / workerElapsed if workerElapsed > 0 else 0
		print(
			f"[Slot {slotId:02d}] frame {frameNum:06d}  {elapsed:.1f}s",
			flush=True,
		)

	print(f"[Slot {slotId:02d}] Batch complete, exiting.", flush=True)

# ------------------------------------------- #

def progress_monitor(sharedDone, totalFrames, sharedStartTime, log_interval):
	time.sleep(2)

	while True:
		elapsed = time.time() - sharedStartTime.value
		done = sharedDone[0]
		remaining = totalFrames - done
		pct = 100 * done / totalFrames if totalFrames > 0 else 0

		if elapsed > 0 and done > 0:
			fps = done / elapsed
			eta = fmt_duration(remaining / fps)
			fpsStr = f"{fps:.2f} fr/s"
		else:
			eta = "??:??:??"
			fpsStr = "—"

		sep = "─" * 72
		print(
			f"\n{sep}\n"
			f"  ▶ OVERALL  {done}/{totalFrames} frames  ({pct:.1f}%)  "
			f"elapsed={fmt_duration(elapsed)}  ETA={eta}  throughput={fpsStr}"
			f"\n{sep}",
			flush=True,
		)

		if done >= totalFrames > 0:
			break

		time.sleep(log_interval)

# ------------------------------------------- #

def main():
	parser = argparse.ArgumentParser(description="Parallel mlsharp_to_spatial_photo runner")
	parser.add_argument("--frames-dir",    required=True)
	parser.add_argument("--plys-dir",      required=True)
	parser.add_argument("--out-dir",       required=True)
	parser.add_argument("--num-slots", "-n", type=int, required=True,
						help="Number of concurrent worker processes")
	parser.add_argument("--num-gpus",  "-g", type=int, required=True,
						help="Number of GPUs")
	parser.add_argument("--frames-per-worker", "-k", type=int, default=10,
						help="Max frames each worker processes before exiting and being replaced (default: 10)")
	parser.add_argument("--base-display", type=int, default=100)
	parser.add_argument("--stereo-ipds", nargs="+", type=float,
						default=[0.064, 0.032, 0.016, 0.008])
	parser.add_argument("--log-interval", type=float, default=30.0)
	args = parser.parse_args()

	N = args.num_slots
	M = args.num_gpus
	K = args.frames_per_worker

	# collect and sort all frames:
	# ---------------
	allFramePaths = sorted(
		glob.glob(os.path.join(args.frames_dir, "*.png")),
		key=get_frame_number,
	)

	# build (framePath, plyPath, outStereoImages) tuples, skip missing plys:
	# ---------------
	allTasks = []
	for framePath in allFramePaths:
		frameNum  = get_frame_number(framePath)
		frameStem = os.path.splitext(os.path.basename(framePath))[0]
		plyPath   = os.path.join(args.plys_dir, f"{frameStem}.ply")

		if not os.path.exists(plyPath):
			print(f"WARNING: PLY not found for {framePath}, skipping.")
			continue

		outStereoImages = [
			(ipd, os.path.join(args.out_dir, f"ipd_{int(ipd * 1000):03d}", f"frame_{frameNum:06d}.png"))
			for ipd in args.stereo_ipds
		]

		if all(os.path.exists(path) for _, path in outStereoImages):
			continue

		allTasks.append((framePath, plyPath, outStereoImages))

	totalFrames = len(allTasks)

	# split tasks into groups of K:
	# ---------------
	groups = [allTasks[i:i + K] for i in range(0, totalFrames, K)]
	numGroups = len(groups)

	print(f"Total frames: {totalFrames}  |  Groups of {K}: {numGroups}  |  Slots: {N}  |  GPUs: {M}")
	print(f"Progress summary every {args.log_interval:.0f}s\n")

	# create output folders:
	# ---------------
	for ipd in args.stereo_ipds:
		folder = os.path.join(args.out_dir, f"ipd_{int(ipd * 1000):03d}")
		os.makedirs(folder, exist_ok=True)
		print(f"  Output folder: {folder}")

	# start xvfb for each slot:
	# ---------------
	xvfbProcs = []
	for slotId in range(N):
		disp = args.base_display + slotId
		print(f"  Starting Xvfb :{disp} for slot {slotId} ...")
		xvfbProcs.append(start_xvfb(disp))

	# shared state:
	# ---------------
	sharedDone      = Array(c_int,    [0])
	sharedStartTime = Value(c_double, 0.0)
	nextGroup       = Value(c_int,    0)
	nextGroupLock   = Lock()

	def claim_next_group():
		"""Returns the next group index to process, or None if all done."""
		with nextGroupLock:
			idx = nextGroup.value
			if idx >= numGroups:
				return None, None
			nextGroup.value += 1
		return idx, groups[idx]

	# graceful shutdown:
	# ---------------
	activeWorkers = {}   # slotId -> Process
	shutdown_event = threading.Event()

	def shutdown(sig, frame):
		print("\nInterrupted — terminating all workers and Xvfb instances...")
		shutdown_event.set()
		for p in activeWorkers.values():
			p.terminate()
		for x in xvfbProcs:
			x.terminate()
		sys.exit(1)

	signal.signal(signal.SIGINT,  shutdown)
	signal.signal(signal.SIGTERM, shutdown)

	# start progress monitor:
	# ---------------
	sharedStartTime.value = time.time()

	monitor = threading.Thread(
		target=progress_monitor,
		args=(sharedDone, totalFrames, sharedStartTime, args.log_interval),
		daemon=True,
	)
	monitor.start()

	# seed all slots with their first group:
	# ---------------
	def spawn_worker(slotId):
		groupIdx, batch = claim_next_group()
		if batch is None:
			return False

		gpuId     = slotId % M
		displayId = args.base_display + slotId

		p = Process(
			target=worker,
			args=(
				slotId, gpuId, displayId,
				batch,
				args.frames_dir, args.plys_dir, args.stereo_ipds, args.out_dir,
				sharedDone, sharedStartTime,
			),
			daemon=True,
		)
		p.start()
		activeWorkers[slotId] = p
		print(
			f"  → Slot {slotId:02d}  PID={p.pid}  GPU={gpuId}  group {groupIdx}/{numGroups - 1}  "
			f"({len(batch)} frames)",
			flush=True,
		)
		return True

	for slotId in range(N):
		spawn_worker(slotId)

	print()

	# main loop: replace workers as they finish:
	# ---------------
	while not shutdown_event.is_set():
		allDone = True

		for slotId in list(activeWorkers.keys()):
			p = activeWorkers[slotId]

			if p.is_alive():
				allDone = False
				continue

			p.join()
			del activeWorkers[slotId]

			if not spawn_worker(slotId):
				pass
			else:
				allDone = False

		# check if everything is truly finished
		if not activeWorkers and nextGroup.value >= numGroups:
			break

		time.sleep(1)

	# final summary:
	# ---------------
	elapsed = time.time() - sharedStartTime.value
	done    = sharedDone[0]
	print(f"\n✓ All done.  {done}/{totalFrames} frames in {fmt_duration(elapsed)}.")

	for x in xvfbProcs:
		x.terminate()


if __name__ == "__main__":
	main()