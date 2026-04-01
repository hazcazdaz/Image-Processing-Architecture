## Overview

This project benchmarks the performance of CUDA GPU-accelerated image processing against traditional CPU-based execution across a range of filters and image resolutions. The goal is to demonstrate the performance gap between parallel GPU computing and sequential CPU execution.


## Requirements

- Node.js
- NVIDIA CUDA Toolkit (`nvcc`)
- An NVIDIA GPU with CUDA support

## Running the Project

### Backend (CUDA benchmark server)

```bash
cd backend
make              # compile the CUDA binary (first time only)
node server.js    # starts on http://localhost:3001
```

### Frontend (React UI)

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
