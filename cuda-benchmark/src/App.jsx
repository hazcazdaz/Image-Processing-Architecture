import { useState, useRef, useCallback } from "react";

const FILTERS = [
  {
    id: "grayscale",
    name: "Grayscale",
    category: "Baseline",
    desc: "Weighted luminance conversion (R×0.299 + G×0.587 + B×0.114)",
    complexity: "Low",
  },
  {
    id: "gaussian_3",
    name: "Gaussian Blur 3×3",
    category: "Baseline",
    desc: "Small-kernel separable convolution — minimal GPU advantage",
    complexity: "Low",
  },
  {
    id: "gaussian_31",
    name: "Gaussian Blur 31×31",
    category: "Baseline",
    desc: "Large-kernel separable convolution — significant GPU speedup expected",
    complexity: "Medium",
  },
  {
    id: "sobel",
    name: "Sobel Edge Detection",
    category: "Baseline",
    desc: "Horizontal + vertical gradient magnitude across all pixels",
    complexity: "Low",
  },
  {
    id: "bilateral",
    name: "Bilateral Filter",
    category: "Complex",
    desc: "Edge-preserving smoothing — spatially & range-weighted per pixel",
    complexity: "High",
  },
  {
    id: "nlm",
    name: "Non-Local Means",
    category: "Complex",
    desc: "Patch-based denoising — O(n²) per pixel, GPU critical",
    complexity: "Very High",
  },
  {
    id: "unsharp",
    name: "Unsharp Masking",
    category: "Complex",
    desc: "Blur subtraction + sharpening in two chained passes",
    complexity: "Medium",
  },
];

const RESOLUTIONS = ["HD (1280×720)", "Full HD (1920×1080)", "4K UHD (3840×2160)", "8K UHD (7680×4320)"];

const complexityColor = {
  Low: "#00D4FF",
  Medium: "#FF8C42",
  High: "#FF4D6D",
  "Very High": "#BF00FF",
};

function MetricCard({ label, gpu, cpu, unit = "ms", highlight = false }) {
  const speedup = cpu && gpu ? (cpu / gpu).toFixed(1) : null;
  return (
    <div
      style={{
        background: highlight ? "rgba(0,212,255,0.08)" : "rgba(255,255,255,0.03)",
        border: `1px solid ${highlight ? "#00D4FF44" : "#ffffff11"}`,
        borderRadius: 10,
        padding: "14px 16px",
        display: "flex",
        flexDirection: "column",
        gap: 6,
      }}
    >
      <div style={{ fontSize: 11, color: "#8BA8C0", fontFamily: "monospace", letterSpacing: 1, textTransform: "uppercase" }}>{label}</div>
      <div style={{ display: "flex", gap: 16, alignItems: "flex-end" }}>
        <div>
          <div style={{ fontSize: 11, color: "#00E5A0", marginBottom: 2 }}>GPU</div>
          <div style={{ fontSize: 22, fontWeight: 700, color: "#fff", fontFamily: "monospace" }}>
            {gpu ?? "—"}<span style={{ fontSize: 12, color: "#8BA8C0", marginLeft: 3 }}>{unit}</span>
          </div>
        </div>
        <div>
          <div style={{ fontSize: 11, color: "#FF8C42", marginBottom: 2 }}>CPU</div>
          <div style={{ fontSize: 22, fontWeight: 700, color: "#fff", fontFamily: "monospace" }}>
            {cpu ?? "—"}<span style={{ fontSize: 12, color: "#8BA8C0", marginLeft: 3 }}>{unit}</span>
          </div>
        </div>
        {speedup && (
          <div style={{ marginLeft: "auto", textAlign: "right" }}>
            <div style={{ fontSize: 11, color: "#8BA8C0", marginBottom: 2 }}>Speedup</div>
            <div style={{ fontSize: 26, fontWeight: 800, color: "#00D4FF", fontFamily: "monospace" }}>{speedup}×</div>
          </div>
        )}
      </div>
    </div>
  );
}

function SpeedupBar({ label, value, max }) {
  const pct = Math.min((value / max) * 100, 100);
  const color = value >= 8 ? "#00E5A0" : value >= 4 ? "#00D4FF" : value >= 2 ? "#FF8C42" : "#FF4D6D";
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 8 }}>
      <div style={{ width: 130, fontSize: 12, color: "#8BA8C0", flexShrink: 0 }}>{label}</div>
      <div style={{ flex: 1, height: 8, background: "#ffffff0d", borderRadius: 4, overflow: "hidden" }}>
        <div style={{ width: `${pct}%`, height: "100%", background: color, borderRadius: 4, transition: "width 0.8s ease" }} />
      </div>
      <div style={{ width: 45, fontSize: 13, fontWeight: 700, color, fontFamily: "monospace", textAlign: "right" }}>{value}×</div>
    </div>
  );
}

// Simulated benchmark results (this is backend stuff lul! i js put temp values)
const MOCK_RESULTS = {
  grayscale:    { gpu: 2.1,  cpu: 18.4,  transfer: 0.8,  throughput: 890 },
  gaussian_3:   { gpu: 3.4,  cpu: 22.1,  transfer: 0.8,  throughput: 550 },
  gaussian_31:  { gpu: 8.2,  cpu: 210.3, transfer: 0.8,  throughput: 228 },
  sobel:        { gpu: 4.1,  cpu: 31.7,  transfer: 0.8,  throughput: 452 },
  bilateral:    { gpu: 12.4, cpu: 680.2, transfer: 0.8,  throughput: 150 },
  nlm:          { gpu: 48.2, cpu: 9820.4,transfer: 0.8,  throughput: 39  },
  unsharp:      { gpu: 9.8,  cpu: 145.6, transfer: 0.8,  throughput: 190 },
};

export default function App() {
  const [tab, setTab] = useState("single"); // single | pipeline | compare
  const [selectedFilter, setSelectedFilter] = useState(null);
  const [pipelineSteps, setPipelineSteps] = useState([]);
  const [selectedResolution, setSelectedResolution] = useState(RESOLUTIONS[1]);
  const [dragOver, setDragOver] = useState(false);
  const [uploadedFile, setUploadedFile] = useState(null);
  const [uploadedPreview, setUploadedPreview] = useState(null);
  const [running, setRunning] = useState(false);
  const [results, setResults] = useState(null);
  const [compareFilters, setCompareFilters] = useState([]);
  const fileInputRef = useRef();

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer?.files?.[0] || e.target?.files?.[0];
    if (!file) return;
    setUploadedFile(file);
    setResults(null);
    const reader = new FileReader();
    reader.onload = (ev) => setUploadedPreview(ev.target.result);
    reader.readAsDataURL(file);
  }, []);

  const handleRun = async () => {
    const target = tab === "single" ? selectedFilter : (tab === "pipeline" ? pipelineSteps.length > 0 : true);
    if (!target || !uploadedFile) return;
    setRunning(true);
    setResults(null);

    const LIVE_FILTERS = ["grayscale", "gaussian_3", "gaussian_31", "sobel"];
    const ENDPOINT_MAP = {
      grayscale: "http://localhost:3001/api/grayscale",
      gaussian_3: "http://localhost:3001/api/benchmark",
      gaussian_31: "http://localhost:3001/api/benchmark",
      sobel: "http://localhost:3001/api/sobel",
    };

    // --- Pipeline mode ---
    if (tab === "pipeline" && pipelineSteps.length > 0) {
      // Check all steps have live backends
      const allLive = pipelineSteps.every(id => LIVE_FILTERS.includes(id));
      if (!allLive) {
        const missing = pipelineSteps.filter(id => !LIVE_FILTERS.includes(id)).map(id => FILTERS.find(f => f.id === id)?.name);
        setResults({ error: `No CUDA backend for: ${missing.join(", ")}. Only Grayscale, Gaussian Blur, and Sobel are supported.` });
        setRunning(false);
        return;
      }

      const formData = new FormData();
      formData.append("image", uploadedFile);
      formData.append("steps", JSON.stringify(pipelineSteps));

      try {
        const resp = await fetch("http://localhost:3001/api/pipeline", {
          method: "POST",
          body: formData,
        });
        const data = await resp.json();
        if (!resp.ok) throw new Error(data.error || "Backend error");

        setResults({
          gpuTime: data.totals.gpuTime,
          cpuTime: data.totals.cpuTime,
          globalTime: data.totals.globalTime,
          sharedTime: data.totals.sharedTime,
          transfer: data.totals.transfer,
          throughput: data.totals.throughput,
          resolution: `${data.width}x${data.height}`,
          filter: "Custom Pipeline",
          outputImage: data.outputImage,
          real: true,
          pipeline: true,
          steps: data.steps.map(s => ({
            ...s,
            filterName: FILTERS.find(f => f.id === s.filterId)?.name || s.filterId,
          })),
        });
      } catch (err) {
        console.error("Pipeline error:", err);
        setResults({ error: err.message });
      } finally {
        setRunning(false);
      }
      return;
    }

    // --- Single filter mode ---
    const isLive = tab === "single" && LIVE_FILTERS.includes(selectedFilter);

    if (isLive) {
      const formData = new FormData();
      formData.append("image", uploadedFile);

      const endpoint = ENDPOINT_MAP[selectedFilter];
      if (selectedFilter === "gaussian_3") formData.append("kernelSize", 3);
      if (selectedFilter === "gaussian_31") formData.append("kernelSize", 31);

      try {
        const resp = await fetch(endpoint, {
          method: "POST",
          body: formData,
        });
        const data = await resp.json();
        if (!resp.ok) throw new Error(data.error || "Backend error");

        setResults({
          gpuTime: data.gpuTime,
          cpuTime: data.cpuTime,
          globalTime: data.globalTime,
          sharedTime: data.sharedTime,
          transfer: data.transfer,
          throughput: data.throughput,
          resolution: `${data.width}x${data.height}`,
          filter: FILTERS.find(f => f.id === selectedFilter)?.name,
          outputImage: data.outputImage,
          real: true,
        });
      } catch (err) {
        console.error("Backend error:", err);
        setResults({ error: err.message });
      } finally {
        setRunning(false);
      }
    } else {
      // Mock data for unsupported filters
      setTimeout(() => {
        const base = MOCK_RESULTS[selectedFilter] || MOCK_RESULTS["bilateral"];
        const resFactor = selectedResolution.includes("4K") ? 2.8 : selectedResolution.includes("8K") ? 6.2 : selectedResolution.includes("HD ") ? 0.5 : 1;
        setResults({
          gpuTime:    +(base.gpu * resFactor).toFixed(1),
          cpuTime:    +(base.cpu * resFactor).toFixed(1),
          transfer:   +(base.transfer).toFixed(1),
          throughput: Math.round(base.throughput / Math.sqrt(resFactor)),
          resolution: selectedResolution,
          filter: FILTERS.find(f => f.id === selectedFilter)?.name,
        });
        setRunning(false);
      }, 1800);
    }
  };

  const toggleCompare = (id) => {
    setCompareFilters(prev =>
      prev.includes(id) ? prev.filter(x => x !== id) : prev.length < 4 ? [...prev, id] : prev
    );
  };

  const S = {
    root: {
      minHeight: "100vh",
      background: "#0A1628",
      fontFamily: "'Calibri', 'Segoe UI', sans-serif",
      color: "#E8F4F8",
      display: "flex",
      flexDirection: "column",
    },
    header: {
      background: "#060e1c",
      borderBottom: "1px solid #00D4FF22",
      padding: "0 32px",
      display: "flex",
      alignItems: "center",
      gap: 20,
      height: 58,
      flexShrink: 0,
    },
    logo: {
      width: 32, height: 32,
      background: "#00D4FF22",
      border: "1.5px solid #00D4FF",
      borderRadius: 7,
      display: "flex", alignItems: "center", justifyContent: "center",
      fontSize: 16,
    },
    title: { fontSize: 16, fontWeight: 700, color: "#fff", letterSpacing: 0.5 },
    subtitle: { fontSize: 12, color: "#8BA8C0", marginLeft: 2 },
    badge: {
      marginLeft: "auto",
      fontSize: 11,
      background: "#00D4FF15",
      border: "1px solid #00D4FF44",
      color: "#00D4FF",
      padding: "3px 10px",
      borderRadius: 20,
      letterSpacing: 0.5,
    },
    main: { display: "flex", flex: 1, overflow: "hidden" },
    sidebar: {
      width: 300,
      background: "#0D1E35",
      borderRight: "1px solid #ffffff0a",
      display: "flex",
      flexDirection: "column",
      overflow: "hidden",
      flexShrink: 0,
    },
    content: { flex: 1, overflow: "auto", padding: 28, display: "flex", flexDirection: "column", gap: 20 },
    tabBar: {
      display: "flex",
      borderBottom: "1px solid #ffffff0a",
      padding: "0 16px",
    },
    tab: (active) => ({
      padding: "12px 14px",
      fontSize: 12,
      fontWeight: active ? 700 : 400,
      color: active ? "#00D4FF" : "#8BA8C0",
      borderBottom: active ? "2px solid #00D4FF" : "2px solid transparent",
      cursor: "pointer",
      letterSpacing: 0.3,
      transition: "all 0.15s",
    }),
    sectionLabel: {
      fontSize: 10,
      color: "#8BA8C0",
      letterSpacing: 1.5,
      textTransform: "uppercase",
      padding: "14px 16px 6px",
    },
    filterItem: (active) => ({
      padding: "9px 16px",
      cursor: "pointer",
      background: active ? "rgba(0,212,255,0.1)" : "transparent",
      borderLeft: `3px solid ${active ? "#00D4FF" : "transparent"}`,
      transition: "all 0.15s",
      display: "flex",
      justifyContent: "space-between",
      alignItems: "center",
    }),
    filterName: (active) => ({
      fontSize: 13,
      color: active ? "#fff" : "#C0D8E8",
      fontWeight: active ? 600 : 400,
    }),
    chip: (color) => ({
      fontSize: 9,
      padding: "2px 7px",
      borderRadius: 10,
      background: `${color}22`,
      color: color,
      border: `1px solid ${color}55`,
      fontWeight: 700,
      letterSpacing: 0.5,
    }),
    card: {
      background: "#112240",
      border: "1px solid #ffffff0a",
      borderRadius: 12,
      padding: 20,
    },
    btn: (disabled) => ({
      padding: "11px 26px",
      background: disabled ? "#1B3A6B" : "linear-gradient(135deg, #00D4FF, #0099BB)",
      color: disabled ? "#8BA8C0" : "#0A1628",
      border: "none",
      borderRadius: 8,
      fontSize: 13,
      fontWeight: 700,
      cursor: disabled ? "not-allowed" : "pointer",
      letterSpacing: 0.5,
      transition: "opacity 0.15s",
    }),
  };

  const baseline = FILTERS.filter(f => f.category === "Baseline");
  const complex = FILTERS.filter(f => f.category === "Complex");

  return (
    <div style={S.root}>
      {/* Header */}
      <div style={S.header}>
        <div style={S.logo}>⚡</div>
        <div>
          <div style={S.title}>CUDA Benchmark Tool</div>
          <div style={S.subtitle}>Digital Image Processing · ECEN 489</div>
        </div>
        <div style={S.badge}>CUDA BACKEND · GRAYSCALE, GAUSSIAN BLUR & SOBEL LIVE</div>
      </div>

      <div style={S.main}>
        {/* Sidebar */}
        <div style={S.sidebar}>
          <div style={S.tabBar}>
            {["single", "pipeline", "compare"].map(t => (
              <div key={t} style={S.tab(tab === t)} onClick={() => { setTab(t); setResults(null); setSelectedFilter(null); setPipelineSteps([]); }}>
                {t === "single" ? "Single Filter" : t === "pipeline" ? "Pipelines" : "Compare"}
              </div>
            ))}
          </div>

          <div style={{ flex: 1, overflowY: "auto" }}>
            {tab === "single" && (
              <>
                <div style={S.sectionLabel}>Baseline Filters</div>
                {baseline.map(f => (
                  <div key={f.id} style={S.filterItem(selectedFilter === f.id)} onClick={() => { setSelectedFilter(f.id); setResults(null); }}>
                    <span style={S.filterName(selectedFilter === f.id)}>{f.name}</span>
                    <span style={S.chip(complexityColor[f.complexity])}>{f.complexity}</span>
                  </div>
                ))}
                <div style={S.sectionLabel}>Complex Filters</div>
                {complex.map(f => (
                  <div key={f.id} style={S.filterItem(selectedFilter === f.id)} onClick={() => { setSelectedFilter(f.id); setResults(null); }}>
                    <span style={S.filterName(selectedFilter === f.id)}>{f.name}</span>
                    <span style={S.chip(complexityColor[f.complexity])}>{f.complexity}</span>
                  </div>
                ))}
              </>
            )}

            {tab === "pipeline" && (
              <>
                <div style={S.sectionLabel}>Add Filters to Pipeline</div>
                {FILTERS.map(f => (
                  <div key={f.id} style={S.filterItem(false)} onClick={() => { setPipelineSteps(prev => [...prev, f.id]); setResults(null); }}>
                    <span style={S.filterName(false)}>{f.name}</span>
                    <span style={{ fontSize: 16, color: "#00D4FF", cursor: "pointer", fontWeight: 700 }}>+</span>
                  </div>
                ))}
                {pipelineSteps.length > 0 && (
                  <>
                    <div style={S.sectionLabel}>Current Pipeline</div>
                    {pipelineSteps.map((stepId, i) => {
                      const f = FILTERS.find(x => x.id === stepId);
                      return (
                        <div key={i} style={{ ...S.filterItem(true), gap: 8 }}>
                          <span style={{ fontSize: 11, color: "#00D4FF", fontFamily: "monospace", fontWeight: 700, width: 18, flexShrink: 0 }}>{i + 1}.</span>
                          <span style={{ ...S.filterName(true), flex: 1 }}>{f?.name}</span>
                          <span
                            style={{ fontSize: 14, color: "#FF4D6D", cursor: "pointer", fontWeight: 700, padding: "0 4px" }}
                            onClick={(e) => { e.stopPropagation(); setPipelineSteps(prev => prev.filter((_, j) => j !== i)); setResults(null); }}
                          >×</span>
                        </div>
                      );
                    })}
                    <div
                      style={{ padding: "8px 16px", fontSize: 11, color: "#FF4D6D", cursor: "pointer", textAlign: "center" }}
                      onClick={() => { setPipelineSteps([]); setResults(null); }}
                    >
                      Clear All
                    </div>
                  </>
                )}
              </>
            )}

            {tab === "compare" && (
              <>
                <div style={S.sectionLabel}>Select up to 4 filters</div>
                {FILTERS.map(f => (
                  <div key={f.id} style={S.filterItem(compareFilters.includes(f.id))} onClick={() => toggleCompare(f.id)}>
                    <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                      <div style={{
                        width: 14, height: 14, border: `1.5px solid ${compareFilters.includes(f.id) ? "#00D4FF" : "#8BA8C0"}`,
                        borderRadius: 3, background: compareFilters.includes(f.id) ? "#00D4FF" : "transparent",
                        display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0,
                      }}>
                        {compareFilters.includes(f.id) && <span style={{ fontSize: 9, color: "#0A1628", fontWeight: 900 }}>✓</span>}
                      </div>
                      <span style={S.filterName(compareFilters.includes(f.id))}>{f.name}</span>
                    </div>
                    <span style={S.chip(complexityColor[f.complexity])}>{f.complexity}</span>
                  </div>
                ))}
              </>
            )}
          </div>

          {/* Resolution selector */}
          <div style={{ padding: "12px 16px", borderTop: "1px solid #ffffff0a" }}>
            <div style={{ fontSize: 10, color: "#8BA8C0", letterSpacing: 1.5, marginBottom: 8, textTransform: "uppercase" }}>Resolution</div>
            <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
              {RESOLUTIONS.map(r => (
                <div
                  key={r}
                  onClick={() => setSelectedResolution(r)}
                  style={{
                    padding: "6px 10px",
                    borderRadius: 6,
                    cursor: "pointer",
                    background: selectedResolution === r ? "rgba(0,212,255,0.12)" : "transparent",
                    color: selectedResolution === r ? "#00D4FF" : "#8BA8C0",
                    fontSize: 12,
                    fontWeight: selectedResolution === r ? 700 : 400,
                    border: `1px solid ${selectedResolution === r ? "#00D4FF44" : "transparent"}`,
                    transition: "all 0.15s",
                  }}
                >
                  {r}
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Main content */}
        <div style={S.content}>

          {/* Upload zone */}
          <div
            style={{
              ...S.card,
              border: dragOver ? "2px dashed #00D4FF" : uploadedFile ? "1px solid #00D4FF44" : "2px dashed #ffffff15",
              background: dragOver ? "rgba(0,212,255,0.05)" : uploadedFile ? "#112240" : "#0D1E35",
              cursor: "pointer",
              transition: "all 0.2s",
              display: "flex",
              alignItems: "center",
              gap: 20,
              minHeight: 110,
            }}
            onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
            onDragLeave={() => setDragOver(false)}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
          >
            <input ref={fileInputRef} type="file" accept="image/*" style={{ display: "none" }} onChange={handleDrop} />
            {uploadedPreview ? (
              <>
                <img src={uploadedPreview} alt="preview" style={{ width: 80, height: 80, objectFit: "cover", borderRadius: 8, border: "1px solid #00D4FF44" }} />
                <div>
                  <div style={{ fontSize: 14, color: "#fff", fontWeight: 600 }}>{uploadedFile.name}</div>
                  <div style={{ fontSize: 12, color: "#8BA8C0", marginTop: 4 }}>
                    {(uploadedFile.size / 1024).toFixed(0)} KB · Click to change
                  </div>
                </div>
                <div style={{ marginLeft: "auto", fontSize: 24 }}>✅</div>
              </>
            ) : (
              <div style={{ textAlign: "center", width: "100%", padding: "10px 0" }}>
                <div style={{ fontSize: 28, marginBottom: 8 }}>📁</div>
                <div style={{ fontSize: 13, color: "#8BA8C0" }}>
                  Drop an image here or <span style={{ color: "#00D4FF" }}>click to browse</span>
                </div>
                <div style={{ fontSize: 11, color: "#ffffff33", marginTop: 4 }}>PNG, JPG, BMP — any resolution</div>
              </div>
            )}
          </div>

          {/* Filter detail + run */}
          {tab === "single" && selectedFilter && (
            <div style={S.card}>
              {(() => {
                const f = FILTERS.find(x => x.id === selectedFilter);
                return (
                  <div style={{ display: "flex", alignItems: "flex-start", gap: 16 }}>
                    <div style={{ flex: 1 }}>
                      <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 6 }}>
                        <span style={{ fontSize: 16, fontWeight: 700, color: "#fff" }}>{f.name}</span>
                        <span style={S.chip(complexityColor[f.complexity])}>{f.complexity} complexity</span>
                        <span style={S.chip("#8BA8C0")}>{f.category}</span>
                      </div>
                      <div style={{ fontSize: 13, color: "#8BA8C0", lineHeight: 1.6 }}>{f.desc}</div>
                      <div style={{ fontSize: 11, color: "#ffffff33", marginTop: 8 }}>Target: {selectedResolution}</div>
                    </div>
                    <button
                      style={S.btn(!uploadedFile || running)}
                      disabled={!uploadedFile || running}
                      onClick={handleRun}
                    >
                      {running ? "Running…" : "▶  Run Benchmark"}
                    </button>
                  </div>
                );
              })()}
            </div>
          )}

          {tab === "pipeline" && pipelineSteps.length > 0 && (
            <div style={S.card}>
              <div style={{ display: "flex", alignItems: "flex-start", gap: 16 }}>
                <div style={{ flex: 1 }}>
                  <div style={{ fontSize: 16, fontWeight: 700, color: "#fff", marginBottom: 8 }}>Custom Pipeline ({pipelineSteps.length} {pipelineSteps.length === 1 ? "step" : "steps"})</div>
                  <div style={{ display: "flex", alignItems: "center", gap: 6, flexWrap: "wrap" }}>
                    {pipelineSteps.map((stepId, i) => {
                      const f = FILTERS.find(x => x.id === stepId);
                      return (
                        <span key={i} style={{ display: "flex", alignItems: "center", gap: 6 }}>
                          <span style={{ fontSize: 12, background: "#1B3A6B", padding: "3px 10px", borderRadius: 6, color: "#C0D8E8" }}>{f?.name}</span>
                          {i < pipelineSteps.length - 1 && <span style={{ color: "#00D4FF", fontSize: 12 }}>→</span>}
                        </span>
                      );
                    })}
                  </div>
                  <div style={{ fontSize: 11, color: "#ffffff33", marginTop: 8 }}>Target: {selectedResolution} · GPU transfer amortized across all stages</div>
                </div>
                <button style={S.btn(!uploadedFile || running)} disabled={!uploadedFile || running} onClick={handleRun}>
                  {running ? "Running…" : "▶  Run Pipeline"}
                </button>
              </div>
            </div>
          )}

          {tab === "compare" && compareFilters.length > 0 && (
            <div style={S.card}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <div>
                  <div style={{ fontSize: 14, fontWeight: 700, color: "#fff", marginBottom: 4 }}>Comparing {compareFilters.length} filters</div>
                  <div style={{ fontSize: 12, color: "#8BA8C0" }}>{compareFilters.map(id => FILTERS.find(f => f.id === id)?.name).join("  ·  ")}</div>
                </div>
                <button style={S.btn(!uploadedFile || running)} disabled={!uploadedFile || running} onClick={handleRun}>
                  {running ? "Running…" : "▶  Run All"}
                </button>
              </div>
            </div>
          )}

          {/* Results */}
          {running && (
            <div style={{ ...S.card, display: "flex", alignItems: "center", gap: 16 }}>
              <div style={{
                width: 36, height: 36, borderRadius: "50%",
                border: "3px solid #00D4FF33",
                borderTop: "3px solid #00D4FF",
                animation: "spin 0.8s linear infinite",
              }} />
              <div>
                <div style={{ fontSize: 14, color: "#fff", fontWeight: 600 }}>Running benchmark…</div>
                <div style={{ fontSize: 12, color: "#8BA8C0", marginTop: 2 }}>Dispatching CUDA kernels and CPU reference</div>
              </div>
              <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
            </div>
          )}

          {results && !running && results.error && (
            <div style={{ ...S.card, borderColor: "#FF4D6D44" }}>
              <div style={{ fontSize: 14, fontWeight: 700, color: "#FF4D6D", marginBottom: 6 }}>Benchmark Failed</div>
              <div style={{ fontSize: 13, color: "#8BA8C0" }}>{results.error}</div>
              <div style={{ fontSize: 11, color: "#ffffff33", marginTop: 8 }}>Make sure the backend server is running: cd backend && node server.js</div>
            </div>
          )}

          {results && !running && !results.error && (
            <>
              <div style={{ ...S.card, borderColor: "#00D4FF22" }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}>
                  <div>
                    <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                      <div style={{ fontSize: 16, fontWeight: 700, color: "#fff" }}>Benchmark Results</div>
                      {results.real && (
                        <span style={{ fontSize: 10, padding: "2px 8px", borderRadius: 10, background: "#00E5A022", color: "#00E5A0", border: "1px solid #00E5A055", fontWeight: 700 }}>LIVE CUDA</span>
                      )}
                      {!results.real && (
                        <span style={{ fontSize: 10, padding: "2px 8px", borderRadius: 10, background: "#FF8C4222", color: "#FF8C42", border: "1px solid #FF8C4255", fontWeight: 700 }}>MOCK DATA</span>
                      )}
                    </div>
                    <div style={{ fontSize: 12, color: "#8BA8C0", marginTop: 2 }}>{results.filter} · {results.resolution}</div>
                  </div>
                  <div style={{
                    fontSize: 36, fontWeight: 900, color: "#00D4FF", fontFamily: "monospace",
                    textShadow: "0 0 20px #00D4FF55",
                  }}>
                    {(results.cpuTime / results.gpuTime).toFixed(1)}×
                    <div style={{ fontSize: 11, fontWeight: 400, color: "#8BA8C0", textAlign: "center", textShadow: "none" }}>speedup</div>
                  </div>
                </div>

                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
                  <MetricCard label="Execution Time" gpu={results.gpuTime} cpu={results.cpuTime} unit="ms" highlight />
                  <MetricCard label="Effective (w/ Transfer)" gpu={+(results.gpuTime + results.transfer).toFixed(1)} cpu={results.cpuTime} unit="ms" />
                  <MetricCard label="Transfer Overhead" gpu={results.transfer} cpu="N/A" unit="ms" />
                  <MetricCard label="Throughput" gpu={results.throughput} cpu={Math.round(results.throughput / (results.cpuTime / results.gpuTime))} unit="MP/s" />
                </div>

                {/* Global vs Shared memory breakdown for real results */}
                {results.real && (results.globalTime > 0 || results.sharedTime > 0) && (
                  <div style={{ marginTop: 16, padding: "12px 14px", background: "rgba(0,212,255,0.04)", borderRadius: 8, border: "1px solid #00D4FF22" }}>
                    <div style={{ fontSize: 11, color: "#8BA8C0", fontFamily: "monospace", letterSpacing: 1, textTransform: "uppercase", marginBottom: 8 }}>GPU Memory Strategy Breakdown</div>
                    <div style={{ display: "flex", gap: 24 }}>
                      <div>
                        <div style={{ fontSize: 11, color: "#FF8C42", marginBottom: 2 }}>Global Memory</div>
                        <div style={{ fontSize: 18, fontWeight: 700, color: "#fff", fontFamily: "monospace" }}>
                          {results.globalTime > 0 ? results.globalTime.toFixed(3) : "N/A"}<span style={{ fontSize: 11, color: "#8BA8C0", marginLeft: 3 }}>ms</span>
                        </div>
                      </div>
                      <div>
                        <div style={{ fontSize: 11, color: "#00E5A0", marginBottom: 2 }}>Shared Memory</div>
                        <div style={{ fontSize: 18, fontWeight: 700, color: "#fff", fontFamily: "monospace" }}>
                          {results.sharedTime > 0 ? results.sharedTime.toFixed(3) : "N/A"}<span style={{ fontSize: 11, color: "#8BA8C0", marginLeft: 3 }}>ms</span>
                        </div>
                      </div>
                      {results.globalTime > 0 && results.sharedTime > 0 && (
                        <div>
                          <div style={{ fontSize: 11, color: "#BF00FF", marginBottom: 2 }}>Shared Speedup</div>
                          <div style={{ fontSize: 18, fontWeight: 700, color: "#BF00FF", fontFamily: "monospace" }}>
                            {(results.globalTime / results.sharedTime).toFixed(1)}×
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                )}
              </div>

              {/* Pipeline per-step breakdown */}
              {results.pipeline && results.steps && (
                <div style={S.card}>
                  <div style={{ fontSize: 14, fontWeight: 700, color: "#fff", marginBottom: 14 }}>Pipeline Step Breakdown</div>
                  <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                    {results.steps.map((step, i) => {
                      const stepSpeedup = step.cpuTime && step.gpuTime ? (step.cpuTime / step.gpuTime).toFixed(1) : "—";
                      return (
                        <div key={i} style={{
                          background: "rgba(255,255,255,0.03)",
                          border: "1px solid #ffffff11",
                          borderRadius: 8,
                          padding: "12px 14px",
                        }}>
                          <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 8 }}>
                            <span style={{ fontSize: 12, fontWeight: 700, color: "#00D4FF", fontFamily: "monospace" }}>Step {i + 1}</span>
                            <span style={{ fontSize: 13, fontWeight: 600, color: "#fff" }}>{step.filterName}</span>
                            <span style={{ marginLeft: "auto", fontSize: 16, fontWeight: 800, color: "#00D4FF", fontFamily: "monospace" }}>{stepSpeedup}×</span>
                          </div>
                          <div style={{ display: "flex", gap: 20, fontSize: 12 }}>
                            <div>
                              <span style={{ color: "#00E5A0" }}>GPU </span>
                              <span style={{ color: "#fff", fontFamily: "monospace", fontWeight: 600 }}>{step.gpuTime} ms</span>
                            </div>
                            <div>
                              <span style={{ color: "#FF8C42" }}>CPU </span>
                              <span style={{ color: "#fff", fontFamily: "monospace", fontWeight: 600 }}>{step.cpuTime} ms</span>
                            </div>
                            {step.globalTime > 0 && (
                              <div>
                                <span style={{ color: "#8BA8C0" }}>Global </span>
                                <span style={{ color: "#fff", fontFamily: "monospace", fontWeight: 600 }}>{step.globalTime} ms</span>
                              </div>
                            )}
                            {step.sharedTime > 0 && (
                              <div>
                                <span style={{ color: "#8BA8C0" }}>Shared </span>
                                <span style={{ color: "#fff", fontFamily: "monospace", fontWeight: 600 }}>{step.sharedTime} ms</span>
                              </div>
                            )}
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}

              {/* Output image preview for real results */}
              {results.real && results.outputImage && (
                <div style={S.card}>
                  <div style={{ fontSize: 14, fontWeight: 700, color: "#fff", marginBottom: 12 }}>Processed Output</div>
                  <div style={{ display: "flex", gap: 16, alignItems: "flex-start" }}>
                    <div>
                      <div style={{ fontSize: 11, color: "#8BA8C0", marginBottom: 6 }}>Original</div>
                      <img src={uploadedPreview} alt="original" style={{ maxWidth: 300, maxHeight: 220, borderRadius: 8, border: "1px solid #ffffff11" }} />
                    </div>
                    <div>
                      <div style={{ fontSize: 11, color: "#00D4FF", marginBottom: 6 }}>{results.filter} Output</div>
                      <img src={results.outputImage} alt="processed" style={{ maxWidth: 300, maxHeight: 220, borderRadius: 8, border: "1px solid #00D4FF44" }} />
                    </div>
                  </div>
                </div>
              )}

              {/* Speedup chart */}
              <div style={S.card}>
                <div style={{ fontSize: 14, fontWeight: 700, color: "#fff", marginBottom: 14 }}>
                  {results.real ? "Measured Speedup" : "Projected Speedup by Resolution"}
                </div>
                {results.real ? (
                  <>
                    <SpeedupBar label="GPU (Global)" value={results.globalTime > 0 ? +(results.cpuTime / results.globalTime).toFixed(1) : 0} max={Math.max(+(results.cpuTime / results.gpuTime).toFixed(0), 20)} />
                    {results.sharedTime > 0 && (
                      <SpeedupBar label="GPU (Shared)" value={+(results.cpuTime / results.sharedTime).toFixed(1)} max={Math.max(+(results.cpuTime / results.gpuTime).toFixed(0), 20)} />
                    )}
                    <SpeedupBar label="GPU (w/ Transfer)" value={+(results.cpuTime / (results.gpuTime + results.transfer)).toFixed(1)} max={Math.max(+(results.cpuTime / results.gpuTime).toFixed(0), 20)} />
                    <div style={{ fontSize: 11, color: "#ffffff33", marginTop: 10 }}>
                      Actual measurements from your GPU on the uploaded image ({results.resolution}).
                    </div>
                  </>
                ) : (
                  <>
                    <SpeedupBar label="HD (720p)" value={+(results.cpuTime / results.gpuTime * 0.6).toFixed(1)} max={20} />
                    <SpeedupBar label="Full HD (1080p)" value={+(results.cpuTime / results.gpuTime).toFixed(1)} max={20} />
                    <SpeedupBar label="4K UHD" value={+(results.cpuTime / results.gpuTime * 1.8).toFixed(1)} max={20} />
                    <SpeedupBar label="8K UHD" value={Math.min(+(results.cpuTime / results.gpuTime * 3.2).toFixed(1), 20)} max={20} />
                    <div style={{ fontSize: 11, color: "#ffffff33", marginTop: 10 }}>
                      Note: GPU speedup scales super-linearly with resolution as parallelism is better utilized.
                    </div>
                  </>
                )}
              </div>
            </>
          )}

          {/* Empty state */}
          {!selectedFilter && pipelineSteps.length === 0 && compareFilters.length === 0 && !results && (
            <div style={{ flex: 1, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", gap: 12, opacity: 0.5 }}>
              <div style={{ fontSize: 48 }}>⚡</div>
              <div style={{ fontSize: 16, color: "#8BA8C0" }}>Select a filter from the sidebar to get started</div>
              <div style={{ fontSize: 12, color: "#ffffff33" }}>Upload an image, choose a filter, and run the benchmark</div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
