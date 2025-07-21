// model.ts

import { InferenceSession, Tensor } from 'onnxruntime-web';

// This is the type definition or "contract" for your model
export interface Model {
  latent: number;
  run: (feeds: { [name: string]: Tensor }) => Promise<any>;
  load: () => Promise<void>;
  session?: InferenceSession;
}

// This is the class that implements the contract
class ModelImpl implements Model {
  latent = 512;
  session?: InferenceSession;

  async load() {
    this.session = await InferenceSession.create('./model.onnx');
  }

  // --- ENSURE THIS METHOD BODY IS CORRECT ---
  async run(feeds: { [name: string]: Tensor }) {
    if (!this.session) {
      throw new Error("Session not loaded. Call load() first.");
    }

    // The body should ONLY pass the 'feeds' object directly to the session.
    // Any other logic here that treats 'feeds' as a Tensor will cause an error.
    return this.session.run(feeds);
  }
}

const model: Model = new ModelImpl();

export default model;
