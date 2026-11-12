import assert from 'node:assert/strict';
import test from 'node:test';
import { load, size } from '../chessqueries/portable.mjs';

test('position and turn remain unchanged when orientation flips', async () => {
  let calls = 0, disposed = 0, inputsDisposed = 0, released = false;
  const angles = [0, 180, -90, 90, -95, 95, -100, 100, -100.01, 100.01, -110, 110, -180];
  const ort = {
    Tensor: class {
      constructor(type, data, dims) { Object.assign(this, { type, data, dims }); }
      dispose() { inputsDisposed++; }
    },
    InferenceSession: { create: async () => ({
      async run({ image }) {
        assert.deepEqual(image.dims, [1, 3, size, size]);
        const data = new Float32Array(64 * 13);
        const angle = angles[calls++] * Math.PI / 180;
        for (let i = 0; i < 64; i++) data[i * 13 + (i === 4 ? 12 : i === 60 ? 6 : 0)] = 1;
        return {
          logits: { type: 'float32', dims: [1, 64, 13], data, dispose: () => disposed++ },
          direction: { type: 'float32', dims: [1, 2],
            data: new Float32Array([Math.cos(angle), Math.sin(angle)]), dispose: () => disposed++ },
        };
      },
      release: async () => { released = true; },
    }) },
  };
  const model = await load(ort, new Uint8Array());
  const input = new Float32Array(3 * size * size);
  const white = await model.predict(input), black = await model.predict(input);
  assert.equal(white.orientation, 'white');
  assert.equal(black.orientation, 'black');
  assert.equal(black.viewing_direction.cos, -1);
  assert.deepEqual(white.labels, black.labels);
  assert.equal(black.fen, '4k3/8/8/8/8/8/8/4K3 w - - 0 1');
  assert.equal(disposed, 4);
  for (const angle of angles.slice(2)) {
    const result = await model.predict(input);
    assert.equal(result.orientation, Math.abs(angle) <= 100 ? 'white' : 'black', `${angle}°`);
    assert.deepEqual(result.labels, white.labels);
    assert.equal(result.fen, white.fen);
  }
  assert.equal(disposed, 2 * angles.length);
  assert.equal(inputsDisposed, angles.length);
  await assert.rejects(model.predict(new Float32Array(3)), /Expected/);
  await model.dispose();
  assert.equal(released, true);
});

test('invalid outputs and failed inference release tensors', async () => {
  let disposed = 0, inputsDisposed = 0, fail = false;
  const output = {
    logits: { type: 'float32', dims: [1, 64, 13], data: new Float32Array(64 * 13),
      dispose: () => disposed++ },
    direction: { type: 'float32', dims: [1, 2], data: new Float32Array([0, 0]),
      dispose: () => disposed++ },
  };
  const ort = {
    Tensor: class { dispose() { inputsDisposed++; } },
    InferenceSession: { create: async () => ({
      run: async () => { if (fail) throw new Error('runtime failure'); return output; },
      release: async () => {},
    }) },
  };
  const model = await load(ort, new Uint8Array());
  const input = new Float32Array(3 * size * size);
  await assert.rejects(model.predict(input), /Invalid/);
  output.direction.data[0] = 1;
  output.logits.dims = [1, 13, 64];
  await assert.rejects(model.predict(input), /Invalid/);
  fail = true;
  await assert.rejects(model.predict(input), /runtime failure/);
  assert.equal(inputsDisposed, 3);
  assert.equal(disposed, 4);
  await model.dispose();
});
