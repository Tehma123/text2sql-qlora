// Runs the text-to-SQL model off the main thread (Transformers.js on WebGPU).
//
// Messages in:  {type: "load", modelId}
//               {type: "generate", id, messages}
// Messages out: {type: "progress", file, loaded, total}
//               {type: "ready", loadMs}
//               {type: "token", id}             (first token only, for time-to-first-token)
//               {type: "result", id, text, ttftMs, totalMs, tokens}
//               {type: "error", id?, message}
import { AutoModelForCausalLM, AutoTokenizer, TextStreamer, env } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.2";

env.allowLocalModels = false;
env.useBrowserCache = true;

let tokenizer = null;
let model = null;

async function load(modelId) {
  const t0 = performance.now();
  const progress_callback = (p) => {
    if (p.status === "progress") postMessage({ type: "progress", file: p.file, loaded: p.loaded, total: p.total });
  };
  tokenizer = await AutoTokenizer.from_pretrained(modelId, { progress_callback });
  model = await AutoModelForCausalLM.from_pretrained(modelId, {
    dtype: "q4f16",
    device: "webgpu",
    use_external_data_format: true,
    progress_callback,
  });
  postMessage({ type: "ready", loadMs: performance.now() - t0 });
}

async function generate(id, messages) {
  const inputs = tokenizer.apply_chat_template(messages, { add_generation_prompt: true, return_dict: true });
  const promptTokens = inputs.input_ids.dims.at(-1);
  const start = performance.now();
  let ttftMs = null;
  const streamer = new TextStreamer(tokenizer, {
    skip_prompt: true,
    callback_function: () => {
      if (ttftMs === null) {
        ttftMs = performance.now() - start;
        postMessage({ type: "token", id });
      }
    },
  });
  const out = await model.generate({ ...inputs, max_new_tokens: 256, do_sample: false, streamer });
  const newIds = out.slice(null, [promptTokens, null]);
  const text = tokenizer.batch_decode(newIds, { skip_special_tokens: true })[0];
  postMessage({ type: "result", id, text, ttftMs, totalMs: performance.now() - start, tokens: newIds.dims.at(-1) });
}

self.onmessage = async ({ data }) => {
  try {
    if (data.type === "load") await load(data.modelId);
    else if (data.type === "generate") await generate(data.id, data.messages);
  } catch (e) {
    postMessage({ type: "error", id: data.id, message: String(e?.message ?? e) });
  }
};
