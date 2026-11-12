# ChessQ Lite V5

Use the INT8 ONNX release for CPU/WebAssembly inference. Download it from the
[ChessQ Lite model page](https://huggingface.co/joelseytre/chessq-lite) once the
release is published and verify its SHA-256 against the pinned release manifest.
That page owns artifact availability, download sizes, licenses and validation.
Weights remain separate from this source package. Code and weights retain
PolyForm Noncommercial 1.0.0.

For Python/CUDA, use the FP16 safetensors release through the same predictor:

```sh
poetry run chessqueries-predict photo.jpg --checkpoint chessq-lite-v5-fp16.safetensors --device cuda --json
```

CUDA preserves FP16 inference; `--device cpu` promotes the retained weights to FP32
arithmetic. Both return board placement and viewing direction in one encoder pass.
This does not require ONNX Runtime. The filename in these examples may be renamed
locally after downloading and verifying the artifact listed on the model page.

For INT8 ONNX:

```sh
poetry run pip install -r requirements/portable.txt
poetry run chessqueries-predict photo.jpg --checkpoint chessq-lite-v5-int8.onnx --device cpu --json
```

For a browser, install `onnxruntime-web@1.30.0`, matching Lichess's runtime, and
import the small adapter. Python uses `onnxruntime==1.30.0`.

```js
import * as ort from 'onnxruntime-web/wasm';
import { load } from './chessqueries/portable.mjs';

const model = await load(ort, verifiedModelBytes, { executionProviders: ['wasm'] });
try {
  const result = await model.predict(preprocessedImage);
  chessground.set({ fen: result.fen, orientation: result.orientation });
} finally {
  await model.dispose();
}
```

`preprocessedImage` is a `Float32Array` containing RGB channels in CHW order.
The reference preprocessing ignores EXIF rotation, discards alpha, expands grayscale
to RGB, divides by 255, resizes to 644 × 644 with antialiased bilinear interpolation,
then normalizes each channel using means `[0.485, 0.456, 0.406]` and standard
deviations `[0.229, 0.224, 0.225]`. Python uses that reference preprocessor.
Browser EXIF handling and canvas resizing differ; validate the actual photo-to-result
path as well as inference on prepared tensors. Direction refers to the image
actually passed to the model.

The INT8 graph keeps float32 input/output: `image[1,3,644,644]`,
`logits[1,64,13]` and `direction[1,2]`. Squares follow FEN order, a8 through h1;
classes are `.PNBRQKpnbrqk`. Both adapters return the raw
`viewing_direction: {cos, sin}` unit vector alongside FEN and display orientation.
Direction points from e1 toward e8; cosine is upward and sine is rightward.
Angles within −100° to +100°, including endpoints, place White at the bottom;
other angles place Black at the bottom. Preserve a manual flip control.
Orientation does not rotate the FEN or determine whose turn it is. The FEN's
turn, castling, en-passant and clock fields are placeholders.

ONNX support is optional; existing paper/V4 checkpoint commands retain their behavior.
The Python ONNX runner uses CPU execution. The browser needs neither Python nor a
recognition server. Runtime support and device performance apply only to the
artifact/backend combinations validated in the model's release information.
