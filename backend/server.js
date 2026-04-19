const express = require("express");
const multer = require("multer");
const cors = require("cors");
const { execFile } = require("child_process");
const { promisify } = require("util");
const execFileAsync = promisify(execFile);
const path = require("path");
const fs = require("fs");

const app = express();
const PORT = 3001;

app.use(cors());

// Store uploads in backend/images/
const uploadDir = path.join(__dirname, "images");
if (!fs.existsSync(uploadDir)) fs.mkdirSync(uploadDir);

const upload = multer({ dest: uploadDir });

const GAUSSIAN_BINARY = path.join(__dirname, "bin", "gaussian_blur");
const GRAYSCALE_BINARY = path.join(__dirname, "bin", "grayscale");
const SOBEL_BINARY = path.join(__dirname, "bin", "sobel");
const MEDIAN_BINARY = path.join(__dirname, "bin", "median");
const KUWAHARA_BINARY = path.join(__dirname, "bin", "kuwahara");

// GET /api/system-info — returns CPU and GPU names
app.get("/api/system-info", async (req, res) => {
  let gpu = "Unknown GPU";
  let cpu = "Unknown CPU";

  try {
    const { stdout: gpuOut } = await execFileAsync("nvidia-smi", ["--query-gpu=name", "--format=csv,noheader,nounits"], { timeout: 5000 });
    gpu = gpuOut.trim().split("\n")[0];
  } catch {}

  try {
    const { stdout: cpuOut } = await execFileAsync("bash", ["-c", "grep -m1 'model name' /proc/cpuinfo | cut -d: -f2"], { timeout: 5000 });
    cpu = cpuOut.trim();
  } catch {}

  res.json({ cpu, gpu });
});

// Helper: run a CUDA binary, parse JSON output, return result with output image
function runBenchmark(binary, args, inputPath, outputPath, res) {
  execFile(
    binary,
    args,
    { timeout: 120000 },
    (err, stdout, stderr) => {
      fs.unlink(inputPath, () => {});

      if (err) {
        fs.unlink(outputPath, () => {});
        console.error("Benchmark error:", stderr || err.message);
        return res.status(500).json({ error: "Benchmark failed", details: stderr || err.message });
      }

      let result;
      try {
        result = JSON.parse(stdout.trim());
      } catch (parseErr) {
        fs.unlink(outputPath, () => {});
        return res.status(500).json({ error: "Failed to parse benchmark output", raw: stdout });
      }

      const gpuTime = result.sharedTime > 0 ? result.sharedTime : result.globalTime;

      let outputImage = null;
      if (fs.existsSync(outputPath)) {
        const imgBuffer = fs.readFileSync(outputPath);
        outputImage = "data:image/png;base64," + imgBuffer.toString("base64");
        fs.unlink(outputPath, () => {});
      }

      res.json({
        gpuTime: +gpuTime.toFixed(3),
        cpuTime: +result.cpuTime.toFixed(3),
        globalTime: +result.globalTime.toFixed(3),
        sharedTime: +result.sharedTime.toFixed(3),
        transfer: +result.transferTime.toFixed(3),
        throughput: result.throughput,
        width: result.width,
        height: result.height,
        kernelSize: result.kernelSize || null,
        outputImage,
      });
    }
  );
}

// POST /api/benchmark
// Body: multipart form with "image" file, "kernelSize" (3 or 31)
app.post("/api/benchmark", upload.single("image"), (req, res) => {
  const kernelSize = parseInt(req.body.kernelSize) || 31;
  const inputPath = req.file.path;
  const outputPath = inputPath + "_out.png";

  runBenchmark(GAUSSIAN_BINARY, [inputPath, outputPath, String(kernelSize), "ALL"], inputPath, outputPath, res);
});

// POST /api/grayscale
// Body: multipart form with "image" file
app.post("/api/grayscale", upload.single("image"), (req, res) => {
  const inputPath = req.file.path;
  const outputPath = inputPath + "_out.png";

  runBenchmark(GRAYSCALE_BINARY, [inputPath, outputPath, "ALL"], inputPath, outputPath, res);
});

// POST /api/sobel
// Body: multipart form with "image" file
app.post("/api/sobel", upload.single("image"), (req, res) => {
  const inputPath = req.file.path;
  const outputPath = inputPath + "_out.png";

  runBenchmark(SOBEL_BINARY, [inputPath, outputPath, "ALL"], inputPath, outputPath, res);
});

// POST /api/median
app.post("/api/median", upload.single("image"), (req, res) => {
  const inputPath = req.file.path;
  const outputPath = inputPath + "_out.png";

  runBenchmark(MEDIAN_BINARY, [inputPath, outputPath, "ALL"], inputPath, outputPath, res);
});

// POST /api/kuwahara
app.post("/api/kuwahara", upload.single("image"), (req, res) => {
  const inputPath = req.file.path;
  const outputPath = inputPath + "_out.png";

  runBenchmark(KUWAHARA_BINARY, [inputPath, outputPath, "ALL"], inputPath, outputPath, res);
});

// Map filter IDs to their binary and argument builder
const FILTER_MAP = {
  grayscale:   { binary: GRAYSCALE_BINARY, args: (input, output) => [input, output, "ALL"] },
  gaussian_3:  { binary: GAUSSIAN_BINARY,  args: (input, output) => [input, output, "3", "ALL"] },
  gaussian_31: { binary: GAUSSIAN_BINARY,  args: (input, output) => [input, output, "31", "ALL"] },
  sobel:       { binary: SOBEL_BINARY,     args: (input, output) => [input, output, "ALL"] },
  median:      { binary: MEDIAN_BINARY,   args: (input, output) => [input, output, "ALL"] },
  kuwahara:    { binary: KUWAHARA_BINARY, args: (input, output) => [input, output, "ALL"] },
};

// POST /api/pipeline
// Body: multipart form with "image" file, "steps" (JSON array of filter IDs)
app.post("/api/pipeline", upload.single("image"), async (req, res) => {
  let steps;
  try {
    steps = JSON.parse(req.body.steps);
  } catch {
    return res.status(400).json({ error: "Invalid steps JSON" });
  }

  if (!Array.isArray(steps) || steps.length === 0) {
    return res.status(400).json({ error: "steps must be a non-empty array" });
  }

  // Validate all steps have a backend implementation
  for (const stepId of steps) {
    if (!FILTER_MAP[stepId]) {
      return res.status(400).json({ error: `Unknown or unsupported filter: ${stepId}` });
    }
  }

  const inputPath = req.file.path;
  const tempFiles = [inputPath];
  const stepResults = [];
  let imgWidth = 0, imgHeight = 0;

  try {
    let currentInput = inputPath;

    for (let i = 0; i < steps.length; i++) {
      const stepId = steps[i];
      const filter = FILTER_MAP[stepId];
      const stepOutput = inputPath + `_step${i}_out.png`;
      tempFiles.push(stepOutput);

      const { stdout } = await execFileAsync(
        filter.binary,
        filter.args(currentInput, stepOutput),
        { timeout: 120000 }
      );

      let result;
      try {
        result = JSON.parse(stdout.trim());
      } catch {
        throw new Error(`Failed to parse output for step ${i} (${stepId}): ${stdout}`);
      }

      if (i === 0) {
        imgWidth = result.width;
        imgHeight = result.height;
      }

      const gpuTime = result.sharedTime > 0 ? result.sharedTime : result.globalTime;

      stepResults.push({
        step: i,
        filterId: stepId,
        gpuTime: +gpuTime.toFixed(3),
        cpuTime: +result.cpuTime.toFixed(3),
        globalTime: +result.globalTime.toFixed(3),
        sharedTime: +result.sharedTime.toFixed(3),
        transfer: +result.transferTime.toFixed(3),
        throughput: result.throughput,
      });

      currentInput = stepOutput;
    }

    // Read final output image
    let outputImage = null;
    const finalOutput = tempFiles[tempFiles.length - 1];
    if (fs.existsSync(finalOutput)) {
      const imgBuffer = fs.readFileSync(finalOutput);
      outputImage = "data:image/png;base64," + imgBuffer.toString("base64");
    }

    // Aggregate totals (transfer counted once for initial upload)
    const totalGpuTime = stepResults.reduce((s, r) => s + r.gpuTime, 0);
    const totalCpuTime = stepResults.reduce((s, r) => s + r.cpuTime, 0);
    const totalGlobalTime = stepResults.reduce((s, r) => s + r.globalTime, 0);
    const totalSharedTime = stepResults.reduce((s, r) => s + r.sharedTime, 0);
    const firstTransfer = stepResults[0]?.transfer || 0;
    const lastStep = stepResults[stepResults.length - 1];

    res.json({
      steps: stepResults,
      totals: {
        gpuTime: +totalGpuTime.toFixed(3),
        cpuTime: +totalCpuTime.toFixed(3),
        globalTime: +totalGlobalTime.toFixed(3),
        sharedTime: +totalSharedTime.toFixed(3),
        transfer: +firstTransfer.toFixed(3),
        throughput: lastStep?.throughput || 0,
      },
      width: imgWidth,
      height: imgHeight,
      outputImage,
    });
  } catch (err) {
    console.error("Pipeline error:", err.message);
    res.status(500).json({ error: "Pipeline failed", details: err.message });
  } finally {
    for (const f of tempFiles) {
      fs.unlink(f, () => {});
    }
  }
});

app.listen(PORT, () => {
  console.log(`CUDA benchmark server running on http://localhost:${PORT}`);
});
