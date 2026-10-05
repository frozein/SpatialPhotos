export interface SpatialPhotoInfo 
{
	width: number;
	height: number;
	slices: number;
	blocks: number;
	bytes: number;
}

export interface SpatialPhotoProgress 
{
	loaded: number;
	total: number | null;
	progress: number | null;
}

export type SpatialPhotoSource = string | URL | Blob | ArrayBuffer | Uint8Array;

export interface SpatialPhotoEventMap 
{
	load: CustomEvent<SpatialPhotoInfo>;
	error: CustomEvent<Error>;
	progress: CustomEvent<SpatialPhotoProgress>;
}

export declare class SpatialPhotoElement extends HTMLElement 
{
	src: string;
	fit: 'contain' | 'cover';
	sensitivity: number;
	snappiness: number;
	readonly loading: boolean;
	readonly error: Error | null;
	readonly info: SpatialPhotoInfo | null;

	load(source: SpatialPhotoSource | null): Promise<SpatialPhotoInfo | null>;
	
	addEventListener<K extends keyof SpatialPhotoEventMap>(type: K, listener: (this: SpatialPhotoElement, event: SpatialPhotoEventMap[K]) => void, options?: boolean | AddEventListenerOptions): void;
	addEventListener<K extends keyof HTMLElementEventMap>(type: K, listener: (this: HTMLElement, event: HTMLElementEventMap[K]) => void, options?: boolean | AddEventListenerOptions): void;
	addEventListener(type: string, listener: EventListenerOrEventListenerObject | null, options?: boolean | AddEventListenerOptions): void;
	removeEventListener<K extends keyof SpatialPhotoEventMap>(type: K, listener: (this: SpatialPhotoElement, event: SpatialPhotoEventMap[K]) => void, options?: boolean | EventListenerOptions): void;
	removeEventListener<K extends keyof HTMLElementEventMap>(type: K, listener: (this: HTMLElement, event: HTMLElementEventMap[K]) => void, options?: boolean | EventListenerOptions): void;
	removeEventListener(type: string, listener: EventListenerOrEventListenerObject | null, options?: boolean | EventListenerOptions): void;
}

declare global 
{
	interface HTMLElementTagNameMap 
	{
		'spatial-photo': SpatialPhotoElement;
	}
}
