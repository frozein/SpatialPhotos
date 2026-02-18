import os
import io
import math
import ddgs
import numpy as np
import torch
import base64
import rectpack
import imageio.v2 as imageio

from PIL import Image
from tqdm import tqdm
from plyfile import PlyData

from renderer import render_headless
from exporter import export_glb

# ------------------------------------------- #

OUTFILL_AMOUNT = 0.0

DEPTH_MIN_QUANTILE = 0.0
DEPTH_MAX_QUANTILE = 0.8
NUM_SLICES = 30

BLOCK_SIZE = 64

ALPHA_REPLACE_THRESHOLD = 0.1
ALPHA_TEST_THRESHOLD = 0.5

DEPTH_INFILL_CUTOFF = 0.1
DEPTH_INFILL_OUTLIER_STD = 2.0

ATLAS_MIN_SIZE = 64
ATLAS_MAX_SIZE = 8192

UV_PADDING = 0.0

IPD = 0.064

# ------------------------------------------- #

def look_at(eye, target, up):
	f = (target - eye)
	f = f / torch.norm(f)
	u = up / torch.norm(up)
	s = torch.cross(f, u, dim=0)
	s = s / torch.norm(s)
	u = torch.cross(s, f, dim=0)

	m = torch.eye(4, dtype=torch.float32)
	m[0, :3] = s
	m[1, :3] = u
	m[2, :3] = -f
	m[0, 3] = -torch.dot(s, eye)
	m[1, 3] = -torch.dot(u, eye)
	m[2, 3] = torch.dot(f, eye)

	return m

def perspective(fovy, aspect, znear, zfar):
	tan_half_fovy = math.tan(fovy / 2)

	m = torch.zeros((4, 4), dtype=torch.float32)
	m[0, 0] = 1 / (aspect * tan_half_fovy)
	m[1, 1] = 1 / tan_half_fovy
	m[2, 2] = -(zfar + znear) / (zfar - znear)
	m[2, 3] = -(2 * zfar * znear) / (zfar - znear)
	m[3, 2] = -1.0

	return m

def load_ply(path, device='cuda'):
	data = PlyData.read(path)
	vertex = data['vertex'].data

	def np_to_torch(name, dim=1):
		arr = np.stack([vertex[n] for n in name], axis=-1) if isinstance(name, (list, tuple)) else vertex[name]
		return torch.tensor(arr, dtype=torch.float32, device=device)

	means = np_to_torch(['x', 'y', 'z'])
	colors = 0.5 + np_to_torch(['f_dc_0', 'f_dc_1', 'f_dc_2']) * 0.28209479177387814
	opacities = torch.sigmoid(np_to_torch('opacity').unsqueeze(1))
	scales = torch.exp(np_to_torch(['scale_0', 'scale_1', 'scale_2']))
	rotations = np_to_torch(['rot_1', 'rot_2', 'rot_3', 'rot_0'])

	numGaussians = means.shape[0]
	colors = colors.reshape((numGaussians, 1, 3))

	gaussians = (means, scales, rotations, opacities, colors)
	focalY = data['intrinsic'].data['intrinsic'][0]

	return gaussians, focalY

# ------------------------------------------- #

def slice_t(numSlices, idx):
	return (idx / numSlices) * (idx / numSlices)

def get_slice(gaussians, zMin, zMax, numSlices, idx, includeBehind=False):
	means, scales, rotations, opacities, colors = gaussians

	tMin = slice_t(numSlices, idx)
	tMax = slice_t(numSlices, idx + 1)

	zMinSlice = zMin + tMin * (zMax - zMin)
	zMaxSlice = zMin + tMax * (zMax - zMin)

	if (idx == numSlices - 1) or includeBehind:
		where = means[:, 2] >= zMinSlice
	elif idx == 0:
		where = means[:, 2] < zMaxSlice
	else:
		where = (means[:, 2] >= zMinSlice) & (means[:, 2] < zMaxSlice)

	return means[where], scales[where], rotations[where], opacities[where], colors[where]

# ------------------------------------------- #
# Block extraction — keeps everything on CUDA.
# Returns:
#   present_cpu: (GH, GW) bool numpy array  [tiny, needed for greedy_mesh]
#   img_t:       (H, W, 4) uint8 CUDA tensor [stays on GPU]
# ------------------------------------------- #

def extract_block_presence(img_t: torch.Tensor, blockSize: int):
	"""
	img_t: (H, W, 4) uint8 CUDA tensor
	Returns present_cpu as a (GH, GW) bool numpy array (small, safe to download).
	The full image tensor is NOT downloaded.
	"""
	H, W, _ = img_t.shape
	GH, GW = H // blockSize, W // blockSize

	# reshape: (GH, blockSize, GW, blockSize) then check alpha
	alpha = img_t[:GH*blockSize, :GW*blockSize, 3]          # (GH*bs, GW*bs)
	alpha = alpha.reshape(GH, blockSize, GW, blockSize)      # (GH, bs, GW, bs)
	present = (alpha >= int(ALPHA_TEST_THRESHOLD * 255)).any(dim=1).any(dim=2)  # (GH, GW)

	return present.cpu().numpy()  # tiny boolean grid, cheap to download


def maximal_rectangles(mask):
	h, w = mask.shape
	heights = np.zeros(w, dtype=int)
	rects = []

	for y in range(h):
		for x in range(w):
			heights[x] = heights[x] + 1 if mask[y, x] else 0

		stack = []
		x = 0
		while x <= w:
			cur = heights[x] if x < w else 0
			if not stack or cur >= heights[stack[-1]]:
				stack.append(x)
				x += 1
			else:
				top = stack.pop()
				width = x if not stack else x - stack[-1] - 1
				height = heights[top]
				if width > 0 and height > 0:
					rects.append((x - width, y - height + 1, width, height))

	return rects

def greedy_mesh(mask):
	mask = mask.copy()
	rectsOut = []

	while np.any(mask):
		rects = maximal_rectangles(mask)
		x, y, w, h = max(rects, key=lambda r: r[2] * r[3])
		rectsOut.append((x, y, w, h))
		mask[y:y+h, x:x+w] = False

	return rectsOut

def pack_blocks(mergedBlockDims):
	"""
	mergedBlockDims: list of (pw, ph) — only sizes needed for packing, no image data
	"""
	def fits(size):
		packer = rectpack.newPacker(rotation=False)
		for i, (w, h) in enumerate(mergedBlockDims):
			packer.add_rect(w, h, i)
		packer.add_bin(size, size)
		packer.pack()
		return len(packer.rect_list()) == len(mergedBlockDims)

	low, high = ATLAS_MIN_SIZE, ATLAS_MAX_SIZE
	bestSize = high

	while (high - low) > ATLAS_MIN_SIZE:
		mid = (low + high) // 2
		if fits(mid):
			bestSize = mid
			high = mid - 1
		else:
			low = mid + 1

	finalPacker = rectpack.newPacker(rotation=False)
	for i, (w, h) in enumerate(mergedBlockDims):
		finalPacker.add_rect(w, h, i)
	finalPacker.add_bin(bestSize, bestSize)
	finalPacker.pack()

	return finalPacker


def generate_block_atlas_cuda(slices_cuda, blockSize):
	"""
	slices_cuda: list of (img_t, depth_t)
		img_t:   (H, W, 4) uint8 CUDA tensor
		depth_t: (H, W, 1) float32 CUDA tensor

	Returns:
		atlas_t:    (atlasH, atlasW, 4) uint8 CUDA tensor
		placements: list of (sliceIdx, bx, by, u0, v0, u1, v1)
	"""

	# --- Step 1: greedy mesh on CPU (alpha presence grid only, tiny download) ---
	mergedMeta = []    # (sliceIdx, px, py, pw, ph, blockCoords_list)
	mergedDims = []    # (pw, ph) for rectpack

	for idx, (img_t, _) in enumerate(tqdm(slices_cuda, desc="Greedy meshing slices", unit="slice")):
		present_cpu = extract_block_presence(img_t, blockSize)
		if not np.any(present_cpu):
			continue

		rects = greedy_mesh(present_cpu)

		for (gx, gy, gw, gh) in rects:
			px, py = gx * blockSize, gy * blockSize
			pw, ph = gw * blockSize, gh * blockSize

			blockCoords = [(gx + dx, gy + dy) for dy in range(gh) for dx in range(gw)]
			mergedMeta.append((idx, px, py, pw, ph, blockCoords))
			mergedDims.append((pw, ph))

	# --- Step 2: pack rects (CPU, rectpack) ---
	print("Packing slices into atlas... ", end='', flush=True)

	packer = pack_blocks(mergedDims)
	bin0 = packer.bin_list()[0]
	atlasW, atlasH = bin0

	# --- Step 3: blit merged rects into atlas fully on CUDA ---
	atlas_t = torch.zeros((atlasH, atlasW, 4), dtype=torch.uint8, device='cuda')
	placements = []

	for rect in packer.rect_list():
		_, ax, ay, aw, ah, i = rect

		sliceIdx, src_px, src_py, pw, ph, blockCoords = mergedMeta[i]
		img_t, _ = slices_cuda[sliceIdx]

		# Copy the merged rect from the source slice image directly on GPU
		src_patch = img_t[src_py:src_py+ph, src_px:src_px+pw]   # (ph, pw, 4) — no copy yet
		atlas_t[ay:ay+ah, ax:ax+aw] = src_patch                  # GPU-to-GPU

		for (gx, gy) in blockCoords:
			bx = gx * blockSize
			by = gy * blockSize

			ox = bx - src_px
			oy = by - src_py

			u0 = (ax + ox + UV_PADDING) / atlasW
			v0 = (ay + oy + UV_PADDING) / atlasH
			u1 = (ax + ox + blockSize - UV_PADDING) / atlasW
			v1 = (ay + oy + blockSize - UV_PADDING) / atlasH

			placements.append((sliceIdx, bx, by, u0, v1, u1, v0))

	# Clear sub-threshold alpha in-place on GPU
	alpha = atlas_t[..., 3]
	transparent = alpha < int(ALPHA_TEST_THRESHOLD * 255)
	atlas_t[transparent] = 0
	atlas_t[..., 3][~transparent] = 255

	print('done')
	return atlas_t, placements


# ------------------------------------------- #
# Depth infill — fully on CUDA
# ------------------------------------------- #

def fill_block_depth_cuda(depth_block: torch.Tensor) -> torch.Tensor:
	"""
	depth_block: (blockSize, blockSize) float32 CUDA tensor
	Returns filled tensor, same shape and device.
	"""
	valid = depth_block > 0
	if not valid.any():
		return depth_block

	ys, xs = torch.where(valid)
	zs = depth_block[ys, xs]

	mean = zs.mean()
	std = zs.std() if zs.shape[0] >= 2 else torch.tensor(0.0, device=zs.device)
	inlier = (zs - mean).abs() <= DEPTH_INFILL_OUTLIER_STD * std

	if not inlier.any():
		filled = depth_block.clone()
		filled[~valid] = mean
		return filled

	xs_in = xs[inlier].float()
	ys_in = ys[inlier].float()
	zs_in = zs[inlier]

	# Plane fit via least squares on CUDA
	A = torch.stack([xs_in, ys_in, torch.ones_like(xs_in)], dim=1)  # (N, 3)
	sol = torch.linalg.lstsq(A, zs_in.unsqueeze(1)).solution         # (3, 1)
	a, b, c = sol[0, 0], sol[1, 0], sol[2, 0]

	H, W = depth_block.shape
	yy = torch.arange(H, device=depth_block.device, dtype=torch.float32)
	xx = torch.arange(W, device=depth_block.device, dtype=torch.float32)
	yy, xx = torch.meshgrid(yy, xx, indexing='ij')
	z_est = a * xx + b * yy + c

	filled = depth_block.clone()
	invalid = ~valid
	if valid.float().mean() < DEPTH_INFILL_CUTOFF:
		filled[invalid] = mean
	else:
		filled[invalid] = z_est[invalid]

	return filled

def fill_all_block_depths_cuda(placements, slices_cuda, blockSize):
    """
    Batched replacement for the blockDepths loop.
    Returns: dict (sliceIdx, px, py) -> (bs, bs) float32 CUDA tensor
    """
    device = 'cuda'
    B = blockSize

    # --- Collect unique blocks ---
    seen = {}
    keys = []
    for (sliceIdx, px, py, *_) in placements:
        key = (sliceIdx, px, py)
        if key not in seen:
            seen[key] = len(keys)
            keys.append(key)

    N = len(keys)

    # --- Extract all blocks into a single tensor (N, B, B) ---
    # One slice index + pixel offset per block, gathered in batch
    blocks = torch.zeros((N, B, B), dtype=torch.float32, device=device)
    for i, (sliceIdx, px, py) in enumerate(keys):
        _, depth_t = slices_cuda[sliceIdx]
        blocks[i] = depth_t[py:py+B, px:px+B, 0]

    # --- Valid mask (N, B, B) ---
    valid = blocks > 0                          # (N, B, B)
    valid_count = valid.sum(dim=(1, 2)).float() # (N,)

    # --- Flatten pixels: (N, B*B) ---
    flat   = blocks.reshape(N, B * B)           # (N, P)  P = B*B
    vflat  = valid.reshape(N, B * B)            # (N, P)

    # --- Per-block mean and std (ignoring zeros) ---
    # Use masked mean: sum(z * valid) / count
    sum_z  = (flat * vflat.float()).sum(dim=1)  # (N,)
    mean_z = sum_z / valid_count.clamp(min=1)   # (N,)

    # Masked variance: E[(z-mu)^2] over valid pixels
    diff_sq = ((flat - mean_z.unsqueeze(1)) ** 2) * vflat.float()
    var_z   = diff_sq.sum(dim=1) / (valid_count - 1).clamp(min=1)
    std_z   = var_z.sqrt()
    std_z[valid_count < 2] = 0.0

    # --- Inlier mask: within DEPTH_INFILL_OUTLIER_STD sigmas of mean ---
    inlier = vflat & (
        (flat - mean_z.unsqueeze(1)).abs() <= DEPTH_INFILL_OUTLIER_STD * std_z.unsqueeze(1)
    )  # (N, P)
    inlier_count = inlier.sum(dim=1).float()    # (N,)

    # --- Plane fit via batched lstsq ---
    # Build coordinate grids
    gy = torch.arange(B, device=device, dtype=torch.float32)
    gx = torch.arange(B, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(gy, gx, indexing='ij')  # (B, B)
    xx_flat = xx.reshape(B * B)                      # (P,)
    yy_flat = yy.reshape(B * B)                      # (P,)
    ones    = torch.ones(B * B, device=device)

    # A_full: (P, 3) — same for all blocks (coords don't change)
    A_full = torch.stack([xx_flat, yy_flat, ones], dim=1)  # (P, 3)

    # For batched lstsq we need (N, P, 3) and (N, P, 1)
    # but we only want inlier rows per block — pad to fixed size P
    # Trick: zero out non-inlier rows in A and b, lstsq still works
    # (zeros in both A and b contribute zero to AtA and Atb)
    inlier_f = inlier.float().unsqueeze(2)                    # (N, P, 1)
    A_batch  = A_full.unsqueeze(0).expand(N, -1, -1) * inlier_f   # (N, P, 3)
    b_batch  = (flat * inlier.float()).unsqueeze(2)           # (N, P, 1)

    # Single batched lstsq call — this is the key win
    sol = torch.linalg.lstsq(A_batch, b_batch).solution      # (N, 3, 1)
    a   = sol[:, 0, 0]   # (N,)
    b   = sol[:, 1, 0]   # (N,)
    c   = sol[:, 2, 0]   # (N,)

    # --- Compute z_est for all blocks at once ---
    # z_est[i, p] = a[i]*xx[p] + b[i]*yy[p] + c[i]
    z_est_flat = (
        a.unsqueeze(1) * xx_flat.unsqueeze(0) +
        b.unsqueeze(1) * yy_flat.unsqueeze(0) +
        c.unsqueeze(1)
    )  # (N, P)

    # --- Infill ---
    filled = flat.clone()
    invalid_flat = ~vflat                                  # (N, P)

    # Blocks where valid fraction < DEPTH_INFILL_CUTOFF → fill with mean
    sparse = (valid_count / (B * B)) < DEPTH_INFILL_CUTOFF  # (N,) bool
    use_mean  = sparse.unsqueeze(1) & invalid_flat          # (N, P)
    use_plane = (~sparse).unsqueeze(1) & invalid_flat       # (N, P)

    filled[use_mean]  = mean_z.unsqueeze(1).expand(N, B*B)[use_mean]
    filled[use_plane] = z_est_flat[use_plane]

    # Blocks with zero valid pixels: leave as zero (passthrough)
    no_data = (valid_count == 0).unsqueeze(1).expand(N, B*B)
    filled[no_data] = flat[no_data]

    # --- Blocks where no inliers: fill entirely with mean ---
    no_inlier = (inlier_count == 0).unsqueeze(1).expand(N, B*B)
    filled[no_inlier & invalid_flat] = mean_z.unsqueeze(1).expand(N, B*B)[no_inlier & invalid_flat]

    filled = filled.reshape(N, B, B)

    # --- Return as dict ---
    return {keys[i]: filled[i] for i in range(N)}

# ------------------------------------------- #
# Geometry — fully on CUDA, vectorised
# ------------------------------------------- #

def build_geometry_cuda(placements, slices_cuda, width, height, blockSize, focal, aspect):
    device = 'cuda'
    placements = sorted(placements, key=lambda x: x[0])
    N = len(placements)

    # --- Step 1: fill all block depths (same as before, already fast) ---
    # blockDepths = {}
    # for (sliceIdx, px, py, *_) in tqdm(placements, desc="Filling block depths", unit="block"):
    #     key = (sliceIdx, px, py)
    #     if key in blockDepths:
    #         continue
    #     _, depth_t = slices_cuda[sliceIdx]
    #     db = depth_t[py:py+blockSize, px:px+blockSize, 0]
    #     blockDepths[key] = fill_block_depth_cuda(db)
    print("Filling block depths... ", end='', flush=True)
    blockDepths = fill_all_block_depths_cuda(placements, slices_cuda, blockSize)
    print('done')
    # --- Step 2: build a dense Z grid per slice, shape (S, GH+1, GW+1) ---
    # Each vertex sits at a block corner (multiples of blockSize).
    # We compute the averaged depth at each corner from up to 4 neighbouring blocks.
    # Then we take a cumulative max over the slice axis to enforce monotonicity.

    S  = NUM_SLICES
    GW = width  // blockSize
    GH = height // blockSize

    # Vertex grid: (S, GH+1, GW+1) — index [s, gy, gx] = corner at (gx*bs, gy*bs) in slice s
    # Initialise to 0
    z_grid = torch.zeros((S, GH + 1, GW + 1), dtype=torch.float32, device=device)
    count_grid = torch.zeros((S, GH + 1, GW + 1), dtype=torch.float32, device=device)

    # Accumulate depth contributions from each block into its 4 corners
    # Block (gx, gy) in slice s contributes to corners (gx,gy), (gx+1,gy), (gx,gy+1), (gx+1,gy+1)
    # The corner pixel sampled from the block matches the original logic:
    #   corner (gx,   gy  ) <- block pixel [0,       0      ]  (top-left)
    #   corner (gx+1, gy  ) <- block pixel [0,       bs-1   ]  (top-right)
    #   corner (gx,   gy+1) <- block pixel [bs-1,    0      ]  (bottom-left)
    #   corner (gx+1, gy+1) <- block pixel [bs-1,    bs-1   ]  (bottom-right)
    CORNER_OFFSETS = [
        (0, 0, 0,           0          ),   # (dgx, dgy, row, col)
        (1, 0, 0,           blockSize-1),
        (0, 1, blockSize-1, 0          ),
        (1, 1, blockSize-1, blockSize-1),
    ]

    for (sliceIdx, px, py, *_) in tqdm(placements, desc="Building Z grid", unit="block"):
        key = (sliceIdx, px, py)
        if key not in blockDepths:
            continue
        block = blockDepths[key]   # (bs, bs) CUDA
        gx = px // blockSize
        gy = py // blockSize

        for (dgx, dgy, row, col) in CORNER_OFFSETS:
            cvx = gx + dgx
            cvy = gy + dgy
            if cvx > GW or cvy > GH:
                continue
            val = block[row, col]
            if val > 0:                               # ← only accumulate real depths
                z_grid[sliceIdx, cvy, cvx]     += val
                count_grid[sliceIdx, cvy, cvx] += 1.0


    # Average
    has_data = count_grid > 0
    z_grid[has_data] = z_grid[has_data] / count_grid[has_data]

    for s in range(1, S):
        no_data_this_slice = ~has_data[s]             # (GH+1, GW+1)
        had_data_before    = z_grid[s-1] > 0          # propagated value exists
        should_fill        = no_data_this_slice & had_data_before
        z_grid[s][should_fill] = z_grid[s-1][should_fill]   # carry forward

    # NOW cummax is safe: gaps are filled, so it only ever increases real values
    z_grid_mono, _ = z_grid.cummax(dim=0)

    # --- Step 3: compute world-space XYZ for every vertex in the grid ---
    # x = (gx*bs - width/2)  * z / focal
    # y = (height/2 - gy*bs) * z / focal

    gx_coords = torch.arange(GW + 1, device=device, dtype=torch.float32) * blockSize  # (GW+1,)
    gy_coords = torch.arange(GH + 1, device=device, dtype=torch.float32) * blockSize  # (GH+1,)

    x_offset = gx_coords - width  * 0.5   # (GW+1,)
    y_offset = height * 0.5 - gy_coords   # (GH+1,)

    # Broadcast to (S, GH+1, GW+1)
    x_offset = x_offset.unsqueeze(0).unsqueeze(0).expand(S, GH+1, GW+1)
    y_offset = y_offset.unsqueeze(0).unsqueeze(2).expand(S, GH+1, GW+1)

    x_world = x_offset * z_grid_mono / focal   # (S, GH+1, GW+1)
    y_world = y_offset * z_grid_mono / focal   # (S, GH+1, GW+1)
    # z_world is just z_grid_mono

    # --- Step 4: gather per-placement quad corners from the grid ---
    # Each placement → 4 corners → positions + uvs + indices
    # Do this with index tensors, no Python loop over quads

    pl_s  = torch.tensor([p[0] for p in placements], dtype=torch.long,  device=device)  # (N,)
    pl_px = torch.tensor([p[1] for p in placements], dtype=torch.long,  device=device)  # (N,)
    pl_py = torch.tensor([p[2] for p in placements], dtype=torch.long,  device=device)  # (N,)
    pl_u0 = torch.tensor([p[3] for p in placements], dtype=torch.float32, device=device)
    pl_v0 = torch.tensor([p[4] for p in placements], dtype=torch.float32, device=device)
    pl_u1 = torch.tensor([p[5] for p in placements], dtype=torch.float32, device=device)
    pl_v1 = torch.tensor([p[6] for p in placements], dtype=torch.float32, device=device)

    gx0 = pl_px // blockSize        # (N,) left column index
    gy0 = pl_py // blockSize        # (N,) top row index
    gx1 = gx0 + 1
    gy1 = gy0 + 1

    # 4 corners per quad (matching original winding):
    # 0: (px,    py+bs) → grid (gx0, gy1)
    # 1: (px+bs, py+bs) → grid (gx1, gy1)
    # 2: (px+bs, py   ) → grid (gx1, gy0)
    # 3: (px,    py   ) → grid (gx0, gy0)
    corner_gx = torch.stack([gx0, gx1, gx1, gx0], dim=1)   # (N, 4)
    corner_gy = torch.stack([gy1, gy1, gy0, gy0], dim=1)   # (N, 4)
    corner_s  = pl_s.unsqueeze(1).expand(N, 4)              # (N, 4)

    # Gather world positions: index into (S, GH+1, GW+1) grids
    cx = x_world[corner_s, corner_gy, corner_gx]   # (N, 4)
    cy = y_world[corner_s, corner_gy, corner_gx]   # (N, 4)
    cz = z_grid_mono[corner_s, corner_gy, corner_gx]  # (N, 4)

    positions_t = torch.stack([cx, cy, cz], dim=2).reshape(N * 4, 3)  # (N*4, 3)

    # UVs
    uv_corners = torch.stack([
        torch.stack([pl_u0, pl_v0], dim=1),   # corner 0
        torch.stack([pl_u1, pl_v0], dim=1),   # corner 1
        torch.stack([pl_u1, pl_v1], dim=1),   # corner 2
        torch.stack([pl_u0, pl_v1], dim=1),   # corner 3
    ], dim=1)  # (N, 4, 2)
    uvs_t = uv_corners.reshape(N * 4, 2)   # (N*4, 2)

    # Indices
    base = torch.arange(N, device=device, dtype=torch.int32) * 4   # (N,)
    tri0 = torch.stack([base, base+1, base+2], dim=1)   # (N, 3)
    tri1 = torch.stack([base, base+2, base+3], dim=1)   # (N, 3)
    indices_t = torch.cat([tri0, tri1], dim=1).reshape(N * 2, 3)   # (N*2, 3)

    # Single download
    positions = positions_t.cpu().numpy().reshape(-1).astype(np.float32)
    uvs       = uvs_t.cpu().numpy().reshape(-1).astype(np.float32)
    indices   = indices_t.cpu().numpy().reshape(-1).astype(np.uint32)

    return positions, uvs, indices


# ------------------------------------------- #
# GT color replacement — fully on CUDA
# ------------------------------------------- #

def replace_gt_color_cuda(slices_cuda, orgImage, outfilledWidth, outfilledHeight, orgWidth, orgHeight):
	"""
	slices_cuda: list of (img_t, depth_t) — img_t is (H, W, 4) uint8 CUDA, MODIFIED IN PLACE.
	"""
	device = 'cuda'

	orgRGB = torch.tensor(
		np.flip(np.array(orgImage.convert("RGB"), dtype=np.uint8), axis=1).copy(),
		device=device
	)  # (orgH, orgW, 3) uint8

	S = len(slices_cuda)

	# Stack alpha channels: (S, H, W)
	alpha_stack = torch.stack([slices_cuda[i][0][..., 3].to(torch.int32) for i in range(S)], dim=0)

	threshold = int(ALPHA_REPLACE_THRESHOLD * 255)
	hit_mask  = alpha_stack > threshold          # (S, H, W) bool
	hit_any   = hit_mask.any(dim=0)              # (H, W)
	first_hit = hit_mask.to(torch.int32).argmax(dim=0)  # (S, H, W) -> (H, W)

	offX = (outfilledWidth  - orgWidth)  // 2
	offY = (outfilledHeight - orgHeight) // 2
	H, W = outfilledHeight, outfilledWidth

	yy = torch.arange(H, device=device).unsqueeze(1).expand(H, W)
	xx = torch.arange(W, device=device).unsqueeze(0).expand(H, W)

	inside_gt = (xx >= offX) & (xx < offX + orgWidth) & (yy >= offY) & (yy < offY + orgHeight)

	for s in range(S):
		mask = (first_hit == s) & hit_any & inside_gt   # (H, W)
		if not mask.any():
			continue

		# Gather org pixel coords
		flat_idx = mask.nonzero(as_tuple=False)          # (K, 2): rows=yy, cols=xx
		fy = flat_idx[:, 0]
		fx = flat_idx[:, 1]

		org_x = fx - offX
		org_y = fy - offY

		slices_cuda[s][0][fy, fx, :3] = orgRGB[org_y, org_x]

	return slices_cuda


# ------------------------------------------- #
# Main pipeline
# ------------------------------------------- #

def mlsharp_to_spatial_photo(orgImagePath, plyPath, outGLB, outStereoImage):
	torch.set_default_device('cuda')

	# load original image
	print('Reading original image... ', end='', flush=True)
	orgImage    = Image.open(orgImagePath)
	orgWidth    = orgImage.width
	orgHeight   = orgImage.height

	outfilledWidth  = math.floor((1 + OUTFILL_AMOUNT) * orgWidth)
	outfilledHeight = math.floor((1 + OUTFILL_AMOUNT) * orgHeight)
	outfilledWidth  = (outfilledWidth  // BLOCK_SIZE) * BLOCK_SIZE
	outfilledHeight = (outfilledHeight // BLOCK_SIZE) * BLOCK_SIZE
	aspect = outfilledWidth / outfilledHeight
	print('done')

	# load ply
	print('Reading gaussians... ', end='', flush=True)
	gaussians, focalY = load_ply(plyPath)
	fov = 2 * math.atan(outfilledHeight / (2 * focalY))
	print('done')

	# render settings
	eye    = torch.tensor([0.0, 0.0, 0.0])
	target = torch.tensor([0.0, 0.0, 1.0])
	up     = torch.tensor([0.0, 1.0, 0.0])
	view   = look_at(eye, target, up)
	proj   = perspective(fov, aspect, 0.1, 1000.0)
	focalX = focalY

	settings = ddgs.Settings(
		width=outfilledWidth, height=outfilledHeight,
		view=view, proj=proj,
		focalX=focalX, focalY=focalY,
		outputs=ddgs.RenderOutputs.COLOR | ddgs.RenderOutputs.ALPHA | ddgs.RenderOutputs.DEPTH,
		debug=False
	)

	# render slices — keep everything on CUDA
	means = gaussians[0]
	zMin  = torch.quantile(means[:, 2], DEPTH_MIN_QUANTILE).item()
	zMax  = torch.quantile(means[:, 2], DEPTH_MAX_QUANTILE).item()

	slices_cuda = []  # list of [img_t, depth_t]  (mutable lists so GT replace works in-place)

	for i in tqdm(range(NUM_SLICES), desc='Rendering slices', unit='slice'):
		render       = ddgs.render(settings, *get_slice(gaussians, zMin, zMax, NUM_SLICES, i))
		renderBehind = ddgs.render(settings, *get_slice(gaussians, zMin, zMax, NUM_SLICES, i, includeBehind=True))

		# All ops stay on CUDA
		color_t = renderBehind.color.detach()   # (H, W, 3) float32 CUDA
		alpha_t = render.alpha.detach()          # (H, W, 1) float32 CUDA
		depth_t = render.depth.detach()          # (H, W, 1) float32 CUDA

		img_t = torch.cat([color_t, alpha_t], dim=-1)     # (H, W, 4)
		img_t = (img_t * 255).to(torch.uint8)             # quantise on GPU

		# Clamp depth on GPU
		depth_t = depth_t.clone()
		depth_t[depth_t > zMax] = zMax
		depth_t[(depth_t < zMin) & (depth_t > 0)] = zMin

		slices_cuda.append([img_t, depth_t])

	# GT color replacement — fully on CUDA
	print('Replacing renders with GT color... ', end='', flush=True)
	replace_gt_color_cuda(slices_cuda, orgImage, outfilledWidth, outfilledHeight, orgWidth, orgHeight)
	print('done')

	# Atlas generation — greedy mesh downloads only the bool grid, blitting is GPU-to-GPU
	atlas_t, placements = generate_block_atlas_cuda(slices_cuda, BLOCK_SIZE)

	# Geometry — all CUDA, single download at the end
	positions, uvs, indices = build_geometry_cuda(
		placements, slices_cuda,
		outfilledWidth, outfilledHeight,
		BLOCK_SIZE, focalY, aspect
	)

	# Single atlas download (needed for GLB export / ModernGL unless using CUDA-GL interop)
	atlas_np = atlas_t.cpu().numpy()

	# save GLB
	if outGLB is not None:
		print('Writing GLB... ', end='', flush=True)
		export_glb(atlas_np, positions, uvs, indices, outGLB)
		print('done')

	# render stereo
	if outStereoImage is not None:
		print('Rendering stereo image... ', end='', flush=True)

		eyeLeft    = torch.tensor([ IPD / 2, 0.0, 0.0])
		targetLeft = torch.tensor([ IPD / 2, 0.0, 1.0])
		viewLeft   = look_at(eyeLeft, targetLeft, up)

		eyeRight    = torch.tensor([-IPD / 2, 0.0, 0.0])
		targetRight = torch.tensor([-IPD / 2, 0.0, 1.0])
		viewRight   = look_at(eyeRight, targetRight, up)

		imgLeft  = render_headless(positions, uvs, indices, atlas_np,
								   (orgWidth, orgHeight),
								   viewLeft.cpu().numpy(), proj.cpu().numpy())
		imgRight = render_headless(positions, uvs, indices, atlas_np,
								   (orgWidth, orgHeight),
								   viewRight.cpu().numpy(), proj.cpu().numpy())

		stereo = Image.new(imgLeft.mode, (orgWidth * 2, orgHeight))
		stereo.paste(imgLeft,  (0, 0))
		stereo.paste(imgRight, (orgWidth, 0))
		stereo.save(outStereoImage)

		print('done')


# ------------------------------------------- #

import re
import argparse
from pathlib import Path

def main():
	parser = argparse.ArgumentParser(description="Batch process frame_N.png and frame_N.ply pairs.")
	parser.add_argument("--image_dir", required=True)
	parser.add_argument("--ply_dir",   required=True)
	parser.add_argument("--out_glb_dir", default=None)
	parser.add_argument("--out_png_dir", default=None)
	args = parser.parse_args()

	image_dir   = Path(args.image_dir)
	ply_dir     = Path(args.ply_dir)
	out_glb_dir = Path(args.out_glb_dir) if args.out_glb_dir else None
	out_png_dir = Path(args.out_png_dir) if args.out_png_dir else None

	if out_glb_dir: out_glb_dir.mkdir(parents=True, exist_ok=True)
	if out_png_dir: out_png_dir.mkdir(parents=True, exist_ok=True)

	pattern    = re.compile(r"frame_?(\d+)\.png$")
	image_files = sorted(image_dir.glob("*.png"))
	pairs = []

	for img_path in image_files:
		match = pattern.search(img_path.name)
		if not match:
			continue
		idx = match.group(1)
		ply_path = ply_dir / f"frame_{idx}.ply"
		if not ply_path.exists():
			ply_path = ply_dir / f"frame{idx}.ply"
		if ply_path.exists():
			pairs.append((idx, img_path, ply_path))

	total = len(pairs)
	print(f"Found {total} matching frame pairs\n")

	for i, (idx, img_path, ply_path) in enumerate(pairs, 1):
		out_glb = out_glb_dir / f"frame_{idx}.glb" if out_glb_dir else None
		out_png = out_png_dir / f"frame_{idx}.png" if out_png_dir else None

		mlsharp_to_spatial_photo(
			orgImagePath=str(img_path),
			plyPath=str(ply_path),
			outGLB=str(out_glb)  if out_glb else None,
			outStereoImage=str(out_png) if out_png else None
		)
		print(f"FINISHED {i}/{total} (frame_{idx})\n")


if __name__ == "__main__":
	# mlsharp_to_spatial_photo(
	# 	orgImagePath="insidious/clip2/frames/frame_044.png",
	# 	plyPath     ="insidious/clip2/plys/frame_044.ply",
	# 	outGLB      ="insidious/clip2/glbs/frame_044.glb",
	# 	outStereoImage="insidious/clip2/stereo/frame_044.png",
	# )
	main()