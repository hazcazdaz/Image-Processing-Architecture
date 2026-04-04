const express = require("express");
const multer = require("multer");
const cors = require("cors");
const { execFile } = require("child_process");
const path = require("path");
const fs = require("fs");

const app = express();
const PORT = 3001;

app.use(cors());

// Store uploads in backend/images/
const uploadDir = path.join(__dirname, "images");
if (!fs.existsSync(uploadDir)) fs.mkdirSync(uploadDir);

const upload = multer({ dest: uploadDir });

const GAUSSIAN_BINARY = path.join(__dirname, "gaussian_blur");
const GRAYSCALE_BINARY = path.join(__dirname, "grayscale");
const SOBEL_BINARY = path.join(__dirname, "sobel");

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

app.listen(PORT, () => {
  console.log(`CUDA benchmark server running on http://localhost:${PORT}`);
});
