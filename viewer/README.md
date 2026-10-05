# spatial-photos

A web component for displaying `.spatial` photos with pointer-driven parallax.
Three.js and TypeScript declarations are included.

## Quickstart

```sh
npm install spatial-photos
```

Import the component in your app:

```js
import 'spatial-photos';
```

```html
<spatial-photo
	src="/photo.spatial"
	sensitivity="0.075"
	snappiness="5.5"
	style="width: 100%; height: 480px"
></spatial-photo>
```

On desktop, move the mouse to pan around; leaving the photo returns the camera to center. On mobile, tap or drag to pan around.
Set the size with CSS. Without an explicit height, the component uses a 4:3 aspect ratio.

## Documentation

| Attribute / property | Description |
| --- | --- |
| `src` | Photo URL. Changing it loads a new photo, removing it clears the view. |
| `sensitivity` | Maximum X/Y offset in world units. Default: `0.075`. |
| `snappiness` | Logarithmic catch-up speed from `1` to `10`. `10` is instant. Default: `5.5`. |
| `loading` | Read-only loading state. |
| `error` | Read-only last `Error`, or `null`. |
| `info` | Read-only `{ width, height, slices, blocks, bytes }`, or `null`. |

`load(source)` accepts a URL, `File`, `Blob`, `ArrayBuffer`, or `Uint8Array`.
It resolves to photo info, rejects on errors, and returns `null` when canceled
or queued before attachment.

```js
const photo = document.querySelector('spatial-photo');

photo.addEventListener('load', event => console.log(event.detail));
photo.addEventListener('error', event => console.error(event.detail));
await photo.load(file);
```

The `load`, `error`, and `progress` events carry data in `event.detail`.
Progress contains `{ loaded, total, progress }`. Unknown totals are `null`.
Removing the element cancels loading and releases its GPU resources.

Style the host with CSS. The `canvas`, `status`, and `progress` CSS parts and
`--spatial-photo-background` variable are available for customization.

## Development

```sh
cd viewer
npm install
npm run dev
```

Open `/example/` on the local server. Use the file picker or drop a `.spatial`
file onto the page.

```sh
npm run build
npm pack
```

Build output is in `dist/`. Packing builds the module automatically and includes
the README, types, and third-party license; examples are excluded.
