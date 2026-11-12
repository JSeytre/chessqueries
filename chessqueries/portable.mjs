export const Color = Object.freeze({ WHITE: 'white', BLACK: 'black' });
export const pieces = '.PNBRQKpnbrqk';
export const size = 644;
const whiteViewCosine = Math.fround(Math.cos(100 * Math.PI / 180));

export async function load(ort, bytes, options = {}) {
  const session = await ort.InferenceSession.create(bytes, options);
  return {
    dispose: () => session.release(),
    async predict(image) {
      if (!(image instanceof Float32Array) || image.length !== 3 * size * size)
        throw new Error('Expected a float32 RGB CHW image at 644 × 644');
      const input = new ort.Tensor('float32', image, [1, 3, size, size]);
      let output;
      try {
        output = await session.run({ image: input });
        const logits = output.logits?.data, direction = output.direction?.data;
        if (output.logits?.type !== 'float32' || output.logits?.dims.join(',') !== '1,64,13' ||
            output.direction?.type !== 'float32' || output.direction?.dims.join(',') !== '1,2' ||
            logits?.length !== 64 * pieces.length || direction?.length !== 2 ||
            !logits.every(Number.isFinite) || !direction.every(Number.isFinite) ||
            direction.some(value => Math.abs(value) > 1) ||
            Math.abs(Math.hypot(...direction) - 1) > 1e-4)
          throw new Error('Invalid model output');
        const labels = Array.from({ length: 64 }, (_, square) => {
          const offset = square * pieces.length;
          let best = 0;
          for (let i = 1; i < pieces.length; i++)
            if (logits[offset + i] > logits[offset + best]) best = i;
          return best;
        });
        const placement = Array.from({ length: 8 }, (_, rank) =>
          labels.slice(rank * 8, rank * 8 + 8).map(i => pieces[i]).join('')
            .replace(/\.+/g, empty => String(empty.length))).join('/');
        return { labels, fen: `${placement} w - - 0 1`,
          viewing_direction: { cos: direction[0], sin: direction[1] },
          orientation: direction[0] < whiteViewCosine ? Color.BLACK : Color.WHITE };
      } finally {
        input.dispose();
        for (const tensor of Object.values(output ?? {})) tensor.dispose();
      }
    },
  };
}
