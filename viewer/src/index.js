import { SpatialPhotoElement } from './spatial-photo.js';

// ------------------------------------------- //

if(globalThis.customElements && !customElements.get('spatial-photo'))
	customElements.define('spatial-photo', SpatialPhotoElement);

export { SpatialPhotoElement };
