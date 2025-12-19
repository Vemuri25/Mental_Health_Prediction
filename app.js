let session = null;

const elStatus = document.getElementById("status");
const elOutput = document.getElementById("output");
const loadBtn = document.getElementById("loadBtn");
const runBtn = document.getElementById("runBtn");

function setStatus(msg) {
  elStatus.textContent = msg;
}

function parseRows(text) {
  // Each non-empty line is a row of 5 floats: "a,b,c,d,e"
  const lines = text
    .split(/\r?\n/)
    .map(l => l.trim())
    .filter(l => l.length > 0);

  const rows = lines.map((line, i) => {
    const parts = line.split(",").map(s => s.trim()).filter(Boolean);
    if (parts.length !== 5) {
      throw new Error(`Line ${i + 1} has ${parts.length} values; expected 5.`);
    }
    const nums = parts.map(Number);
    if (nums.some(n => Number.isNaN(n) || !Number.isFinite(n))) {
      throw new Error(`Line ${i + 1} contains a non-number.`);
    }
    return nums;
  });

  return rows;
}

loadBtn.addEventListener("click", async () => {
  try {
    setStatus("Loading model...");

    // Good default for local use
    ort.env.wasm.numThreads = navigator.hardwareConcurrency ?? 1;

    const modelPath = document.getElementById("modelPath").value.trim();

    session = await ort.InferenceSession.create(modelPath, {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
    });

    setStatus(
      "✅ Model loaded\n" +
      `Inputs:  ${session.inputNames.join(", ")}\n` +
      `Outputs: ${session.outputNames.join(", ")}`
    );

    runBtn.disabled = false;
  } catch (err) {
    console.error(err);
    setStatus("❌ ERROR loading model:\n" + (err?.stack || err));
  }
});

runBtn.addEventListener("click", async () => {
  if (!session) return;

  try {
    setStatus("Running inference...");

    // Your model details
    const INPUT_NAME = "input";
    const OUTPUT_NAME = "output";

    const rows = parseRows(document.getElementById("inputRows").value);
    const N = rows.length;

    // Flatten [N,5] into a single Float32Array
    const flat = new Float32Array(N * 5);
    for (let r = 0; r < N; r++) {
      for (let c = 0; c < 5; c++) {
        flat[r * 5 + c] = rows[r][c];
      }
    }

    const inputTensor = new ort.Tensor("float32", flat, [N, 5]);

    const results = await session.run({ [INPUT_NAME]: inputTensor });

    const out = results[OUTPUT_NAME] ?? results[session.outputNames[0]];

    const outData = Array.from(out.data);
    elOutput.textContent =
      `Output name: ${OUTPUT_NAME}\n` +
      `Type: ${out.type}\n` +
      `Dims: ${JSON.stringify(out.dims)}\n` +
      `Data (first 200): ${outData.slice(0, 200).join(", ")}`;

    setStatus("✅ Done.");
  } catch (err) {
    console.error(err);
    setStatus("❌ ERROR running inference:\n" + (err?.stack || err));
  }
});
