import { Tensor } from 'onnxruntime-web'
import model, { Model } from './model'
import gaussian from 'gaussian'

type def = (n: number, fn: () => number) => number[]
const gauss = gaussian(0, 1)

const SIZE = 1024

const getStyleInput = () => {
  // IMPORTANT: Replace with the correct shape from Netron (e.g., [1, 512])
  const styleShape = [1, 512]; 
  const styleSize = styleShape.reduce((a, b) => a * b, 1);
  const styleData = new Float32Array(styleSize); // Creates a tensor of zeros
  return new Tensor('float32', styleData, styleShape);
};

async function render(ctx: CanvasRenderingContext2D, res: any) {
  if (!ctx) {
    throw new Error('Could not get context')
  }
  const data = ctx.getImageData(0, 0, SIZE, SIZE)
  for (let x = 0; x < SIZE; x++) {
    for (let y = 0; y < SIZE; y++) {
      const r = ((res.get(0, 0, x, y) + 1.0) / 2.0) * 255
      const g = ((res.get(0, 1, x, y) + 1.0) / 2.0) * 255
      const b = ((res.get(0, 2, x, y) + 1.0) / 2.0) * 255
      const i = (y + x * SIZE) * 4
      data.data[i + 0] = r
      data.data[i + 1] = g
      data.data[i + 2] = b
      data.data[i + 3] = 255
    }
  }
  ctx.putImageData(data, 0, 0)
}

const getRandomZ = (dims = [1, 128]) => {
  const ndims = dims.reduce((a, b) => a * b, 1)
  const z = Float32Array.from((gauss.random as def)(ndims, Math.random))
  return new Tensor('float32', z, [1, model.latent])
}

async function generate(model: Model, ctx: CanvasRenderingContext2D) {
  document.body.classList.remove('loaded');
  const loader = document.querySelector('.loader');
  if (loader) {
    loader.innerHTML = 'Rendering...';
  }
  setTimeout(async () => {
    // Create both the latent and style tensors
    const latentTensor = getRandomZ([1, model.latent]);
    const styleTensor = getStyleInput();
    // Set the truncation value. 0.7 is a good default.
    const truncationValue = 0.7;
    // Use the shape and type you found in Netron (e.g., [1] and 'float32')
    const varTensor = new Tensor('float32', [truncationValue], [1]);
    
    // --- NEW CODE ENDS HERE ---


    // 2. Create the 'feeds' object with ALL THREE named inputs
    const feeds = {
      'z': latentTensor,
      'style': styleTensor,
      'var': varTensor // Add the new tensor here
    };
    // Pass the entire feeds object to the run method
    const res = await model.run(feeds); 
    
    await render(ctx, res);
    document.body.classList.add('loaded');
  }, 10);
}

async function main() {
  const canvas = document.createElement('canvas')
  canvas.id = 'render'
  canvas.width = SIZE
  canvas.height = SIZE
  document.body.appendChild(canvas)
  const ctx = canvas.getContext('2d')

  if (!ctx) {
    throw new Error('Could not get context')
  }
  try {
    await model.load()

    generate(model, ctx)
    canvas.addEventListener('click', () => {
      if (!document.body.classList.contains('loaded')) {
        return
      }
      generate(model, ctx)
    })
  } catch (e) {
    console.error(e)
    alert(e)
  }
}

main()
