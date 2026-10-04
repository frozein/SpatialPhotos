import { build } from 'esbuild';
import { rm, mkdir, copyFile } from 'node:fs/promises';

// ------------------------------------------- //

await rm('dist', { recursive: true, force: true });
await mkdir('dist', { recursive: true });

await build({
	entryPoints: ['src/index.js'],
	outfile: 'dist/index.js',
	bundle: true,
	format: 'esm',
	target: 'es2022',
	minify: true,
	sourcemap: true,
	legalComments: 'eof',
});

await copyFile('src/index.d.ts', 'dist/index.d.ts');
await copyFile('node_modules/three/LICENSE', 'THIRD_PARTY_LICENSES.txt');
