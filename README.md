Overview:

This project benchmarks the performance of CUDA GPU-accelerated image processing against traditional CPU-based execution across a range of filters and image resolutions. The goal is to demonstrate the performance gap between parallel GPU computing and sequential CPU execution.


## Running the Frontend Locally

**Requirements:** Node.js installed on your machine.

**First time setup:**
```bash
npm create vite@latest cuda-benchmark -- --template react
cd cuda-benchmark
npm install
copy ..\cuda_gui.jsx src\App.jsx
```

**Every time after that, just:**
```bash
cd cuda-benchmark
npm run dev
```

Then open `http://localhost:5173` in your browser. To stop the server press `Ctrl + C`.
