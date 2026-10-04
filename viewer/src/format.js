const HEADER_BYTES = 60;
const SPATIAL_MAGIC = 0x00415053;

// ------------------------------------------- //

function readMask(view, offset, numSlots, count) 
{
	const maskBytes = Math.ceil(numSlots / 8);
	const mask = new Uint8Array(view.buffer, view.byteOffset + offset, maskBytes);
	const indices = new Uint32Array(count);
	let index = 0;

	for(let byteIdx = 0; byteIdx < maskBytes; byteIdx++) 
	{
		let bits = mask[byteIdx];
		while(bits) 
		{
			const bit = 31 - Math.clz32(bits & -bits);
			indices[index++] = byteIdx * 8 + bit;
			bits &= bits - 1;
		}
	}

	return { indices, offset: offset + maskBytes };
}

export function parseSpatial(data) 
{
	const bytes = data instanceof ArrayBuffer ? new Uint8Array(data) : data;
	const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
	if(view.getUint32(0, true) !== SPATIAL_MAGIC)
		throw new Error('This photo is not in the Spatial format.');

	//read header:
	//---------------
	const header = {
		imageWidth: view.getUint32(4, true),
		imageHeight: view.getUint32(8, true),
		numSlices: view.getUint32(12, true),
		blockSize: view.getUint32(16, true),
		focal: view.getFloat32(20, true),
		atlasWidth: view.getUint32(24, true),
		atlasHeight: view.getUint32(28, true),
		opaqueOnly: Boolean(view.getUint32(32, true) & 1),
		uvPadding: view.getFloat32(36, true),
		totalCorners: view.getUint32(40, true),
		totalBlocks: view.getUint32(44, true),
		geometryBytes: view.getUint32(48, true),
		colorBytes: view.getUint32(52, true),
		alphaBytes: view.getUint32(56, true),
	};
	header.gridWidth = header.imageWidth / header.blockSize;
	header.gridHeight = header.imageHeight / header.blockSize;

	//read slices:
	//---------------
	const numCornerSlots = (header.gridWidth + 1) * (header.gridHeight + 1);
	const numBlockSlots = header.gridWidth * header.gridHeight;
	const slices = [];
	let offset = HEADER_BYTES;

	for(let sliceIdx = 0; sliceIdx < header.numSlices; sliceIdx++) 
	{
		const cornerCount = view.getUint32(offset, true);
		const blockCount = view.getUint32(offset + 4, true);
		offset += 8;

		const corners = readMask(view, offset, numCornerSlots, cornerCount);
		offset = corners.offset;
		const depths = new Float32Array(numCornerSlots);
		for(const cornerIdx of corners.indices) 
		{
			depths[cornerIdx] = view.getFloat32(offset, true);
			offset += 4;
		}

		const blocks = readMask(view, offset, numBlockSlots, blockCount);
		offset = blocks.offset;
		const atlasX = new Int16Array(numBlockSlots).fill(-1);
		const atlasY = new Int16Array(numBlockSlots).fill(-1);
		for(const blockIdx of blocks.indices) 
		{
			atlasX[blockIdx] = bytes[offset++];
			atlasY[blockIdx] = bytes[offset++];
		}

		slices.push({ depths, blocks: blocks.indices, atlasX, atlasY });
	}

	const colorOffset = HEADER_BYTES + header.geometryBytes;
	const alphaOffset = colorOffset + header.colorBytes;
	return {
		header,
		slices,
		color: bytes.subarray(colorOffset, alphaOffset),
		alpha: bytes.subarray(alphaOffset, alphaOffset + header.alphaBytes),
	};
}

export function buildRenderBuffers(photo) 
{
	const { header, slices } = photo;
	const positions = new Float32Array(header.totalBlocks * 12);
	const uvs = new Float32Array(header.totalBlocks * 8);
	const indices = new Uint32Array(header.totalBlocks * 6);
	let blockIdx = 0;

	//define projection function:
	//---------------
	const project = (pixel, center, depth) => Math.fround(
		Math.fround(Math.fround(pixel - Math.fround(center)) * depth) / header.focal
	);

	//process each block:
	//---------------
	for(let sliceIdx = slices.length - 1; sliceIdx >= 0; sliceIdx--) 
	{
		const { depths, blocks, atlasX, atlasY } = slices[sliceIdx];
		const isAdjacent = (index, x, y) => atlasX[index] === x && atlasY[index] === y;

		for(const index of blocks) 
		{
			const gridX = index % header.gridWidth;
			const gridY = Math.floor(index / header.gridWidth);
			const atlasBlockX = atlasX[index];
			const atlasBlockY = atlasY[index];
			let left = atlasBlockX * header.blockSize;
			let right = (atlasBlockX + 1) * header.blockSize;
			let top = atlasBlockY * header.blockSize;
			let bottom = (atlasBlockY + 1) * header.blockSize;

			//inset blocks with no neighbor:
			//---------------
			if(!(gridX > 0 && isAdjacent(index - 1, atlasBlockX - 1, atlasBlockY)))
				left = Math.fround(left + header.uvPadding);
			if(!(gridX + 1 < header.gridWidth && isAdjacent(index + 1, atlasBlockX + 1, atlasBlockY)))
				right = Math.fround(right - header.uvPadding);
			if(!(gridY > 0 && isAdjacent(index - header.gridWidth, atlasBlockX, atlasBlockY - 1)))
				top = Math.fround(top + header.uvPadding);
			if(!(gridY + 1 < header.gridHeight && isAdjacent(index + header.gridWidth, atlasBlockX, atlasBlockY + 1)))
				bottom = Math.fround(bottom - header.uvPadding);

			//build vertices:
			//---------------
			const cornerX = [gridX, gridX + 1, gridX + 1, gridX];
			const cornerY = [gridY + 1, gridY + 1, gridY, gridY];
			const textureX = [left, right, right, left];
			const textureY = [bottom, bottom, top, top];
			const base = blockIdx * 4;

			for(let cornerIdx = 0; cornerIdx < 4; cornerIdx++) 
			{
				const depth = depths[cornerY[cornerIdx] * (header.gridWidth + 1) + cornerX[cornerIdx]];
				const vertexIdx = base + cornerIdx;
				
				positions[vertexIdx * 3] = project(cornerX[cornerIdx] * header.blockSize, header.imageWidth * 0.5, depth);
				positions[vertexIdx * 3 + 1] = project(header.imageHeight * 0.5, cornerY[cornerIdx] * header.blockSize, depth);
				positions[vertexIdx * 3 + 2] = depth;
				
				uvs[vertexIdx * 2] = textureX[cornerIdx] / header.atlasWidth;
				uvs[vertexIdx * 2 + 1] = textureY[cornerIdx] / header.atlasHeight;
			}

			//add quad indices:
			//---------------
			indices.set([base, base + 1, base + 2, base, base + 2, base + 3], blockIdx * 6);
			blockIdx++;
		}
	}

	return { positions, uvs, indices };
}
