import * as THREE from 'three';
import { parseSPM, buildRenderBuffers } from './spm.js';

// ------------------------------------------- //

function decodeJPEG(bytes, signal)
{
	return new Promise((resolve, reject) => {
		const image = new Image();
		const url = URL.createObjectURL(new Blob([bytes], { type: 'image/jpeg' }));

		const cleanup = () => {
			URL.revokeObjectURL(url);
			signal.removeEventListener('abort', abort);
			image.onload = image.onerror = null;
		};

		const abort = () => {
			cleanup();
			image.src = '';
			reject(new DOMException('Photo loading was canceled.', 'AbortError'));
		};

		image.onload = () => { cleanup(); resolve(image); };
		image.onerror = () => { cleanup(); reject(new Error('Could not decode the photo images.')); };

		signal.addEventListener('abort', abort, { once: true });
		if(signal.aborted)
			abort();
		else
			image.src = url;
	});
}

function makeTexture(image, colorSpace)
{
	const texture = new THREE.Texture(image);
	texture.flipY = false;
	texture.premultiplyAlpha = false;
	texture.colorSpace = colorSpace;
	texture.generateMipmaps = false;
	texture.minFilter = texture.magFilter = THREE.NearestFilter;
	texture.needsUpdate = true;

	return texture;
}

// ------------------------------------------- //

const ElementBase = globalThis.HTMLElement || class {};

export class SpatialPhotoElement extends ElementBase 
{
	static observedAttributes = ['src'];

	constructor() 
	{
		super();

		//clear member vars:
		//---------------
		this.source = null;
		this.loadId = 0;
		this.loadController = null;

		this.renderer = null;
		this.scene = null;
		this.camera = null;

		this.targetPosition = new THREE.Vector3();
		this.pointerX = 0;
		this.pointerY = 0;
		this.hovered = false;
		this.lastFrame = 0;

		this.mesh = null;
		this.header = null;
		this.photoInfo = null;

		this.isLoading = false;
		this.loadError = null;

		//create the HTML:
		//---------------
		this.attachShadow({ mode: 'open' });
		this.shadowRoot.innerHTML = `
			<style>
				:host {
					display: block;
					position: relative;
					aspect-ratio: 4 / 3;
					overflow: hidden;
					background: var(--spatial-photo-background, #07090c);
				}
				:host([hidden]) { display: none; }
				canvas {
					position: absolute;
					inset: 0;
					width: 100%;
					height: 100%;
					display: block;
				}
				.status {
					position: absolute;
					left: 16px;
					right: 16px;
					bottom: 24px;
					padding: 10px 14px;
					border-radius: 8px;
					background: #111722dd;
					color: #eef2f7;
					font: 13px / 1.5 system-ui, sans-serif;
					pointer-events: none;
				}
				progress {
					position: absolute;
					left: 16px;
					bottom: 12px;
					width: calc(100% - 32px);
					height: 4px;
					accent-color: #91baff;
				}
				[hidden] { display: none; }
			</style>
			<div class="status" part="status" role="status" hidden></div>
			<progress part="progress" max="1" hidden></progress>
		`;
		this.statusElement = this.shadowRoot.querySelector('.status');
		this.progressElement = this.shadowRoot.querySelector('progress');

		//follow the pointer, return to centre when the mouse leaves:
		//---------------
		this.addEventListener('pointerenter', event => this.aim(event));
		this.addEventListener('pointermove', event => this.aim(event));
		this.addEventListener('pointerdown', event => this.aim(event));
		this.addEventListener('pointerleave', event => {
			if(event.pointerType !== 'touch')
				this.hovered = false;
		});
	}

	connectedCallback() 
	{
		const loadId = this.loadId;
		queueMicrotask(() => {
			if(!this.isConnected || loadId !== this.loadId) 
				return;

			try 
			{
				this.init();
				const source = this.source || this.src;
				if(source) 
					this.load(source).catch(() => {});
			} 
			catch (error) 
			{
				this.setState(false, error);
				this.emit('error', error);
			}
		});
	}

	disconnectedCallback() 
	{
		this.loadId++;

		this.loadController?.abort();
		this.resizeObserver?.disconnect();

		this.clearPhoto();

		if(this.renderer) 
		{
			this.renderer.setAnimationLoop(null);
			this.renderer.dispose();
			this.renderer.forceContextLoss();
			this.renderer.domElement.remove();
		}

		this.renderer = this.scene = this.camera = null;
		this.targetPosition.set(0, 0, 0);
		this.pointerX = this.pointerY = 0;
		this.hovered = false;
		this.setState(false);
	}

	attributeChangedCallback(name, oldValue, value) 
	{
		if(oldValue === value) 
			return;
		
		this.source = value;
		if(this.isConnected) 
			this.load(value).catch(() => {});
	}

	get src() 
	{ 
		return this.getAttribute('src') || ''; 
	}

	set src(value) 
	{
		if(value) 
			this.setAttribute('src', String(value));
		else 
			this.removeAttribute('src');
	}

	get loading() { return this.isLoading; }
	get error() { return this.loadError; }
	get info() { return this.photoInfo; }

	get sensitivity() { return Number(this.getAttribute('sensitivity') ?? 0.075); }
	set sensitivity(value) { this.setAttribute('sensitivity', String(value)); }
	get snappiness() { return Number(this.getAttribute('snappiness') ?? 0.1); }
	set snappiness(value) { this.setAttribute('snappiness', String(value)); }

	// ------------------------------------------- //

	init() 
	{
		if(this.renderer) 
			return;

		//create canvas + renderer:
		//---------------
		const canvas = this.ownerDocument.createElement('canvas');
		canvas.setAttribute('part', 'canvas');

		this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
		this.renderer.outputColorSpace = THREE.SRGBColorSpace;
		this.renderer.toneMapping = THREE.NoToneMapping;

		this.shadowRoot.prepend(canvas);

		//create objects needed for rendering:
		//---------------
		this.scene = new THREE.Scene();

		this.camera = new THREE.PerspectiveCamera(45, 1, 0.001, 1000);

		//watch for resize:
		//---------------
		this.resizeObserver = new ResizeObserver(() => this.resize());
		this.resizeObserver.observe(this);
		this.resize();

		this.lastFrame = performance.now();
		this.renderer.setAnimationLoop(time => {
			const dt = Math.min(time - this.lastFrame, 100);
			this.lastFrame = time;
			this.updateCamera(dt);
			this.renderer.render(this.scene, this.camera);
		});
	}

	aim(event)
	{
		const rect = this.getBoundingClientRect();
		if(!rect.width || !rect.height)
			return;

		this.pointerX = THREE.MathUtils.clamp((event.clientX - rect.left) / rect.width * 2 - 1, -1, 1);
		this.pointerY = THREE.MathUtils.clamp(1 - (event.clientY - rect.top) / rect.height * 2, -1, 1);
		this.hovered = true;
	}

	updateCamera(dt)
	{
		//set target position:
		//---------------
		const sensitivity = this.sensitivity;
		this.targetPosition.set(
			this.hovered ? this.pointerX * sensitivity : 0,
			this.hovered ? this.pointerY * sensitivity : 0,
			0,
		);

		//apply easing:
		//---------------
		const snappiness = THREE.MathUtils.clamp(this.snappiness, 0, 1);
		const amount = 1 - Math.pow(1 - snappiness, dt / (1000 / 60));
		this.camera.position.lerp(this.targetPosition, amount);
		this.camera.position.z = 0;
		this.camera.rotation.set(0, 0, 0);
	}

	resize() 
	{
		if(!this.renderer) 
			return;
		
		const width = Math.max(1, this.clientWidth);
		const height = Math.max(1, this.clientHeight);
		
		this.renderer.setPixelRatio(Math.min(globalThis.devicePixelRatio || 1, 2));
		this.renderer.setSize(width, height, false);
		
		this.camera.aspect = width / height;
		
		if(this.header) 
		{
			const sourceAspect = this.header.imageWidth / this.header.imageHeight;
			this.camera.fov = THREE.MathUtils.radToDeg(2 * Math.atan(
				this.header.imageHeight / (2 * this.header.focal) * Math.max(1, sourceAspect / this.camera.aspect)
			));
		}
		this.camera.updateProjectionMatrix();
		
		if(this.mesh) 
			this.renderer.render(this.scene, this.camera);
	}

	// ------------------------------------------- //

	async readPhoto(source, signal, onProgress)
	{
		//just return data if not a string/URL:
		//---------------
		if(typeof source !== 'string' && !(source instanceof URL))
		{
			const data = source instanceof Blob ? await source.arrayBuffer() : source;
			onProgress({ loaded: data.byteLength, total: data.byteLength, progress: 1 });
			return data;
		}

		//start fetch:
		//---------------
		const response = await fetch(source, { signal });
		if(!response.ok)
			throw new Error(`Could not load photo (HTTP ${response.status}).`);

		const total = Number(response.headers.get('content-length')) || null;
		if(!response.body)
		{
			const data = await response.arrayBuffer();
			onProgress({ loaded: data.byteLength, total: data.byteLength, progress: 1 });
			return data;
		}

		//update progress:
		//---------------
		const reader = response.body.getReader();
		const chunks = [];
		let loaded = 0;
		while(true)
		{
			const { value, done } = await reader.read();
			if(done)
				break;
			chunks.push(value);
			loaded += value.byteLength;
			onProgress({ loaded, total, progress: total ? Math.min(loaded / total, 1) : null });
		}

		//return data:
		//---------------
		const data = new Uint8Array(loaded);
		let offset = 0;
		for(const chunk of chunks)
		{
			data.set(chunk, offset);
			offset += chunk.byteLength;
		}
		onProgress({ loaded, total: loaded, progress: 1 });

		return data;
	}

	async createPhoto(data, signal)
	{
		//read SPM:
		//---------------
		const photo = parseSPM(data);
		if(!photo.header.totalBlocks)
			throw new Error('This photo has no visible blocks.');

		const { positions, uvs, indices } = buildRenderBuffers(photo);
		const [colorImage, alphaImage] = await Promise.all([
			decodeJPEG(photo.color, signal),
			decodeJPEG(photo.alpha, signal),
		]);
		if(signal.aborted)
			throw new DOMException('Photo loading was canceled.', 'AbortError');

		//create mesh:
		//---------------
		const opaqueOnly = photo.header.opaqueOnly;
		const material = new THREE.MeshBasicMaterial({
			map: makeTexture(colorImage, THREE.SRGBColorSpace),
			alphaMap: makeTexture(alphaImage, THREE.NoColorSpace),
			side: THREE.BackSide,
			transparent: !opaqueOnly,
			premultipliedAlpha: !opaqueOnly,
			depthTest: opaqueOnly,
			depthWrite: opaqueOnly,
			alphaTest: opaqueOnly ? 0.5 : 0,
			blending: opaqueOnly ? THREE.NormalBlending : THREE.CustomBlending,
			blendSrc: THREE.OneFactor,
			blendDst: THREE.OneMinusSrcAlphaFactor,
			blendEquation: THREE.AddEquation,
		});
		const geometry = new THREE.BufferGeometry();
		geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
		geometry.setAttribute('uv', new THREE.BufferAttribute(uvs, 2));
		geometry.setIndex(new THREE.BufferAttribute(indices, 1));
		geometry.computeBoundingBox();

		const mesh = new THREE.Mesh(geometry, material);
		mesh.rotation.y = Math.PI;
		mesh.frustumCulled = false;
		return { mesh, header: photo.header };
	}

	disposePhoto(mesh)
	{
		if(!mesh)
			return;
		mesh.removeFromParent();
		mesh.geometry.dispose();
		mesh.material.map.dispose();
		mesh.material.alphaMap.dispose();
		mesh.material.dispose();
	}

	clearPhoto() 
	{
		this.disposePhoto(this.mesh);
		this.mesh = this.header = this.photoInfo = null;
		if(this.renderer && this.scene) 
			this.renderer.render(this.scene, this.camera);
	}

	// ------------------------------------------- //

	async load(source) 
	{
		//clear existing:
		//---------------
		this.source = source;
		this.loadController?.abort();
		const loadId = ++this.loadId;
		this.clearPhoto();
		if(!source || !this.isConnected) 
		{
			this.setState(false);
			return null;
		}

		//create abortcontroller, flag loading:
		//---------------
		const controller = new AbortController();
		this.loadController = controller;
		this.setState(true);
		this.progressElement.removeAttribute('value');

		try 
		{
			this.init();

			//fetch file:
			//---------------
			const data = await this.readPhoto(source, controller.signal, progress => {
				if(loadId !== this.loadId) 
					return;
				if(progress.progress === null) 
					this.progressElement.removeAttribute('value');
				else 
					this.progressElement.value = progress.progress;
				
				this.emit('progress', progress);
			});
			if(loadId !== this.loadId) 
				return null;

			//parse:
			//---------------
			const photo = await this.createPhoto(data, controller.signal);
			if(loadId !== this.loadId) 
			{
				this.disposePhoto(photo.mesh);
				return null;
			}

			//set rendering fields:
			//---------------
			this.mesh = photo.mesh;
			this.header = photo.header;
			this.scene.add(photo.mesh);
			this.photoInfo = {
				width: photo.header.imageWidth,
				height: photo.header.imageHeight,
				slices: photo.header.numSlices,
				blocks: photo.header.totalBlocks,
				bytes: data.byteLength,
			};

			//start at the capture pose:
			//---------------
			const bounds = this.mesh.geometry.boundingBox;
			this.camera.near = Math.max(0.0001, bounds.min.z * 0.001);
			this.camera.far = Math.max(100, bounds.max.z * 100);
			this.camera.position.set(0, 0, 0);
			this.camera.rotation.set(0, 0, 0);
			this.targetPosition.set(0, 0, 0);
			this.pointerX = this.pointerY = 0;
			this.hovered = false;
			this.resize();

			this.setState(false);
			this.emit('load', this.photoInfo);

			return this.photoInfo;
		} 
		catch (error) 
		{
			if(controller.signal.aborted || loadId !== this.loadId) 
				return null;

			controller.abort();
			this.setState(false, error);
			this.emit('error', error);
			throw error;
		}
	}

	setState(loading, error = null) 
	{
		this.isLoading = loading;
		this.loadError = error;
		this.toggleAttribute('loading', loading);
		this.toggleAttribute('error', Boolean(error));
		this.setAttribute('aria-busy', String(loading));
		this.statusElement.hidden = !loading && !error;
		this.statusElement.textContent = loading ? 'Loading photo…' : error?.message || '';
		this.progressElement.hidden = !loading;
	}

	emit(type, detail) 
	{
		this.dispatchEvent(new CustomEvent(type, { detail, bubbles: true, composed: true }));
	}
}
