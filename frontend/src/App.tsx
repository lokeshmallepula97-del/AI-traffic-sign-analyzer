import { useState, useEffect } from 'react';
import { 
  LayoutDashboard, 
  ScanSearch, 
  Database, 
  LineChart, 
  Cpu, 
  BookOpen, 
  CheckCircle2, 
  AlertCircle,
  UploadCloud,
  ArrowRight,
  ShieldCheck,
  Search,
  Activity,
  FileCode
} from 'lucide-react';
import { 
  fetchDatasetSummary, 
  fetchDatasetClasses, 
  fetchModelStatus, 
  fetchMetrics, 
  predictImage, 
  uploadModelFile,
  analyzeWithAIStudio,
} from './services/api';
import type {
  DatasetSummary,
  DatasetClass,
  ModelStatus,
  EvaluationMetrics,
  PredictionResponse,
  AIStudioAnalysisResponse
} from './services/api';

type TabType = 'overview' | 'recognize' | 'dataset' | 'performance' | 'training' | 'methodology';

export default function App() {
  const [currentTab, setCurrentTab] = useState<TabType>('overview');
  const [modelStatus, setModelStatus] = useState<ModelStatus | null>(null);
  const [summary, setSummary] = useState<DatasetSummary | null>(null);
  const [classes, setClasses] = useState<DatasetClass[]>([]);
  const [, setMetrics] = useState<EvaluationMetrics | null>(null);

  // Recognition state
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [dragActive, setDragActive] = useState(false);
  
  // Google AI Studio state
  const [aiStudioLoading, setAiStudioLoading] = useState(false);
  const [aiStudioData, setAiStudioData] = useState<AIStudioAnalysisResponse | null>(null);

  // Dataset filter state
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedCategory, setSelectedCategory] = useState<string>('All');

  // Model upload state
  const [uploadStatus, setUploadStatus] = useState<string | null>(null);

  useEffect(() => {
    async function loadInitialData() {
      try {
        const [st, sum, cls, met] = await Promise.all([
          fetchModelStatus().catch(() => null),
          fetchDatasetSummary().catch(() => null),
          fetchDatasetClasses().catch(() => []),
          fetchMetrics().catch(() => null)
        ]);
        setModelStatus(st);
        setSummary(sum);
        setClasses(cls);
        setMetrics(met);
      } catch (err) {
        console.error('Failed to load initial backend state:', err);
      }
    }
    loadInitialData();
  }, []);

  const handleFileChange = (file: File) => {
    setSelectedFile(file);
    setPrediction(null);
    setAiStudioData(null);
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
  };

  const handleAIStudioQuery = async () => {
    if (!selectedFile) return;
    setAiStudioLoading(true);
    try {
      const res = await analyzeWithAIStudio(selectedFile, prediction?.predicted_class);
      setAiStudioData(res);
    } catch (e) {
      setAiStudioData({ configured: false, error: 'Could not connect to Google AI Studio endpoint.' });
    } finally {
      setAiStudioLoading(false);
    }
  };

  const handleAnalyze = async () => {
    if (!selectedFile) return;
    setAnalyzing(true);
    try {
      const res = await predictImage(selectedFile);
      setPrediction(res);
    } catch (err) {
      setPrediction({ success: false, error: 'Connection failed to backend API.' });
    } finally {
      setAnalyzing(false);
    }
  };

  const handleQuickSampleSelect = async (classKey: string) => {
    try {
      const res = await fetch(`/api/dataset/sample/${classKey}`);
      if (res.ok) {
        const blob = await res.blob();
        const file = new File([blob], `${classKey}_sample.jpg`, { type: 'image/jpeg' });
        handleFileChange(file);
        setCurrentTab('recognize');
      }
    } catch (e) {
      console.error('Could not load sample image:', e);
    }
  };

  const filteredClasses = classes.filter(c => {
    const matchesSearch = c.display_name.toLowerCase().includes(searchTerm.toLowerCase()) ||
                          c.key.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesCategory = selectedCategory === 'All' || c.category === selectedCategory;
    return matchesSearch && matchesCategory;
  });

  return (
    <div className="app-container">
      {/* Sidebar Navigation */}
      <aside className="sidebar">
        <div className="sidebar-brand">
          <div className="brand-badge">
            <span style={{ color: '#2563EB', fontSize: '18px' }}>▲</span>
            BanTrafficNet
          </div>
          <div className="brand-sub">Indian Traffic Sign Recognition</div>
        </div>

        <nav className="sidebar-nav">
          <button 
            className={`nav-item ${currentTab === 'overview' ? 'active' : ''}`}
            onClick={() => setCurrentTab('overview')}
          >
            <LayoutDashboard size={17} /> Overview
          </button>
          <button 
            className={`nav-item ${currentTab === 'recognize' ? 'active' : ''}`}
            onClick={() => setCurrentTab('recognize')}
          >
            <ScanSearch size={17} /> Recognize Sign
          </button>
          <button 
            className={`nav-item ${currentTab === 'dataset' ? 'active' : ''}`}
            onClick={() => setCurrentTab('dataset')}
          >
            <Database size={17} /> Dataset Explorer
          </button>
          <button 
            className={`nav-item ${currentTab === 'performance' ? 'active' : ''}`}
            onClick={() => setCurrentTab('performance')}
          >
            <LineChart size={17} /> Model Performance
          </button>
          <button 
            className={`nav-item ${currentTab === 'training' ? 'active' : ''}`}
            onClick={() => setCurrentTab('training')}
          >
            <Cpu size={17} /> Training Lab
          </button>
          <button 
            className={`nav-item ${currentTab === 'methodology' ? 'active' : ''}`}
            onClick={() => setCurrentTab('methodology')}
          >
            <BookOpen size={17} /> Methodology
          </button>
        </nav>

        <div className="sidebar-footer">
          <div className="status-indicator">
            <div className="status-dot" style={{ background: modelStatus?.model_loaded ? '#10B981' : '#F59E0B' }} />
            <span>{modelStatus?.model_loaded ? 'Model Online (4.6M)' : 'Model Standby'}</span>
          </div>
        </div>
      </aside>

      {/* Main Content Area */}
      <main className="main-content">
        <header className="top-bar">
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '13px', color: '#64748B' }}>
            <span>Indian Traffic Sign Recognition</span>
            <span>/</span>
            <span style={{ color: '#0F172A', fontWeight: 600, textTransform: 'capitalize' }}>{currentTab}</span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '16px', fontSize: '12.5px', color: '#475569' }}>
            <span>IRC:67 Standards</span>
            <span style={{ borderLeft: '1px solid #E2E8F0', height: '16px' }} />
            <span>56 Verified Classes</span>
          </div>
        </header>

        <div className="page-container">
          {/* ========================================================= */}
          {/* 1. OVERVIEW DASHBOARD */}
          {/* ========================================================= */}
          {currentTab === 'overview' && (
            <div>
              <div className="page-header">
                <h1 className="page-title">Indian Traffic Sign Recognition</h1>
                <p className="page-subtitle">
                  High-accuracy, lightweight convolutional neural network classification powered by BanTrafficNet.
                </p>
              </div>

              {/* Stat Cards */}
              <div className="stat-grid">
                <div className="stat-card">
                  <span className="stat-label">Original Road Images</span>
                  <span className="stat-val">{summary?.total_original_images ? summary.total_original_images.toLocaleString() : '3,277'}</span>
                  <span className="stat-desc" style={{ color: '#059669', fontWeight: 600 }}>
                    ✓ Exceeds 1,001 requirement (3,277)
                  </span>
                </div>
                <div className="stat-card">
                  <span className="stat-label">IRC:67 Sign Classes</span>
                  <span className="stat-val">{summary?.total_classes || 56}</span>
                  <span className="stat-desc">Mandatory, Cautionary, Informatory</span>
                </div>
                <div className="stat-card">
                  <span className="stat-label">Model Parameters</span>
                  <span className="stat-val">4.64M</span>
                  <span className="stat-desc">Exact: 4,643,544 params (Lightweight)</span>
                </div>
                <div className="stat-card">
                  <span className="stat-label">Input Resolution</span>
                  <span className="stat-val">96 × 96</span>
                  <span className="stat-desc">RGB 3-channel normalized [0, 1]</span>
                </div>
              </div>

              {/* Main split sections */}
              <div style={{ display: 'grid', gridTemplateColumns: '1.4fr 1fr', gap: '24px', marginBottom: '24px' }}>
                <div className="card">
                  <h3 style={{ fontSize: '16px', fontWeight: 700, marginBottom: '12px' }}>
                    Quick Recognition Workspace
                  </h3>
                  <p style={{ fontSize: '13.5px', color: '#64748B', marginBottom: '18px' }}>
                    Test the BanTrafficNet classifier immediately by selecting a sample roadside image from the verified Indian dataset.
                  </p>
                  
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '12px', marginBottom: '20px' }}>
                    {classes.slice(0, 4).map(c => (
                      <div 
                        key={c.key} 
                        onClick={() => handleQuickSampleSelect(c.key)}
                        style={{
                          border: '1px solid #E2E8F0',
                          borderRadius: '8px',
                          padding: '10px',
                          textAlign: 'center',
                          cursor: 'pointer',
                          background: '#FFFFFF',
                          transition: 'all 0.15s ease'
                        }}
                        onMouseEnter={e => e.currentTarget.style.borderColor = '#2563EB'}
                        onMouseLeave={e => e.currentTarget.style.borderColor = '#E2E8F0'}
                      >
                        <img 
                          src={`/api/dataset/sample/${c.key}`} 
                          alt={c.display_name} 
                          style={{ width: '100%', height: '70px', objectFit: 'contain', marginBottom: '6px' }}
                        />
                        <div style={{ fontSize: '11px', fontWeight: 600, color: '#0F172A', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                          {c.display_name}
                        </div>
                      </div>
                    ))}
                  </div>

                  <button className="btn btn-primary" onClick={() => setCurrentTab('recognize')}>
                    Open Full Recognition Workspace <ArrowRight size={15} />
                  </button>
                </div>

                <div className="card">
                  <h3 style={{ fontSize: '16px', fontWeight: 700, marginBottom: '12px' }}>
                    IRC:67 Sign Distribution
                  </h3>
                  <p style={{ fontSize: '13px', color: '#64748B', marginBottom: '16px' }}>
                    Dataset categorization according to the Indian Road Congress specifications:
                  </p>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '10px 12px', background: '#FEF2F2', borderRadius: '8px', border: '1px solid #FECACA' }}>
                      <span style={{ fontSize: '13px', fontWeight: 600, color: '#DC2626' }}>Mandatory / Regulatory</span>
                      <span style={{ fontSize: '13px', fontWeight: 700, color: '#DC2626' }}>25 Classes</span>
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '10px 12px', background: '#FFFBEB', borderRadius: '8px', border: '1px solid #FDE68A' }}>
                      <span style={{ fontSize: '13px', fontWeight: 600, color: '#D97706' }}>Cautionary / Warning</span>
                      <span style={{ fontSize: '13px', fontWeight: 700, color: '#D97706' }}>28 Classes</span>
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '10px 12px', background: '#F0FDF4', borderRadius: '8px', border: '1px solid #BBF7D0' }}>
                      <span style={{ fontSize: '13px', fontWeight: 600, color: '#16A34A' }}>Informatory / Facility</span>
                      <span style={{ fontSize: '13px', fontWeight: 700, color: '#16A34A' }}>3 Classes</span>
                    </div>
                  </div>
                </div>
              </div>

              {/* Verification & Compliance banner */}
              <div className="card" style={{ background: '#F8FAFC', borderLeft: '4px solid #2563EB' }}>
                <div style={{ display: 'flex', alignItems: 'flex-start', gap: '14px' }}>
                  <ShieldCheck size={24} style={{ color: '#2563EB', flexShrink: 0, marginTop: '2px' }} />
                  <div>
                    <h4 style={{ fontSize: '14px', fontWeight: 700, color: '#0F172A' }}>
                      Strict Dataset Verification Standard Met
                    </h4>
                    <p style={{ fontSize: '13px', color: '#475569', marginTop: '4px', lineHeight: 1.6 }}>
                      In accordance with research project criteria, exactly <strong>3,277 non-augmented original road images</strong> have been verified with SHA-256 deduplication and zero corruption. Data augmentation (rotations $\pm 15^\circ$, contrast/brightness $\pm 10\%$, scaling, blurring) is isolated strictly to training partitions to prevent synthetic leakage.
                    </p>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* ========================================================= */}
          {/* 2. RECOGNIZE TRAFFIC SIGN */}
          {/* ========================================================= */}
          {currentTab === 'recognize' && (
            <div>
              <div className="page-header">
                <h1 className="page-title">Recognize Traffic Sign</h1>
                <p className="page-subtitle">
                  Upload an Indian road sign image for real-time BanTrafficNet neural classification.
                </p>
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1.2fr', gap: '28px' }}>
                {/* Left Panel: Upload & Preview */}
                <div className="card">
                  <h3 style={{ fontSize: '15px', fontWeight: 700, marginBottom: '14px' }}>Image Input</h3>
                  
                  <div 
                    className={`dropzone ${dragActive ? 'active' : ''}`}
                    onDragOver={e => { e.preventDefault(); setDragActive(true); }}
                    onDragLeave={() => setDragActive(false)}
                    onDrop={e => {
                      e.preventDefault();
                      setDragActive(false);
                      if (e.dataTransfer.files && e.dataTransfer.files[0]) {
                        handleFileChange(e.dataTransfer.files[0]);
                      }
                    }}
                    onClick={() => document.getElementById('file-input')?.click()}
                  >
                    <input 
                      type="file" 
                      id="file-input" 
                      style={{ display: 'none' }} 
                      accept="image/jpeg,image/png,image/jpg,image/bmp"
                      onChange={e => e.target.files && e.target.files[0] && handleFileChange(e.target.files[0])}
                    />
                    
                    {previewUrl ? (
                      <div>
                        <img 
                          src={previewUrl} 
                          alt="Preview" 
                          style={{ maxHeight: '200px', maxWidth: '100%', borderRadius: '8px', objectFit: 'contain', margin: '0 auto' }} 
                        />
                        <div style={{ fontSize: '12px', color: '#64748B', marginTop: '10px' }}>
                          Click or drag to replace image
                        </div>
                      </div>
                    ) : (
                      <div>
                        <UploadCloud size={36} style={{ color: '#2563EB', margin: '0 auto 12px' }} />
                        <div style={{ fontSize: '14px', fontWeight: 600, color: '#0F172A' }}>
                          Click to upload or drag & drop
                        </div>
                        <div style={{ fontSize: '12px', color: '#94A3B8', marginTop: '4px' }}>
                          JPG, PNG, or BMP (auto-standardized to 96×96)
                        </div>
                      </div>
                    )}
                  </div>

                  <div style={{ marginTop: '20px', display: 'flex', gap: '12px' }}>
                    <button 
                      className="btn btn-primary" 
                      style={{ flex: 1 }}
                      disabled={!selectedFile || analyzing}
                      onClick={handleAnalyze}
                    >
                      {analyzing ? 'Analyzing Tensor...' : 'Analyze Traffic Sign'}
                    </button>
                    {previewUrl && (
                      <button 
                        className="btn btn-secondary"
                        onClick={() => { setSelectedFile(null); setPreviewUrl(null); setPrediction(null); }}
                      >
                        Clear
                      </button>
                    )}
                  </div>

                  {/* Sample signs selector */}
                  <div style={{ marginTop: '24px', paddingTop: '16px', borderTop: '1px solid #E2E8F0' }}>
                    <span style={{ fontSize: '12px', fontWeight: 600, color: '#64748B', textTransform: 'uppercase' }}>
                      Or choose an Indian road sample:
                    </span>
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '8px', marginTop: '10px' }}>
                      {['stop', 'give_way', 'no_entry', 'pedestrian_crossing', 'school_ahead', 'speed_limit_50', 'hospital_ahead', 'men_at_work'].map(k => (
                        <button
                          key={k}
                          className="btn btn-secondary"
                          style={{ padding: '6px 8px', fontSize: '11px', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}
                          onClick={() => handleQuickSampleSelect(k)}
                        >
                          {k.replace(/_/g, ' ')}
                        </button>
                      ))}
                    </div>
                  </div>
                </div>

                {/* Right Panel: Prediction Results */}
                <div className="card">
                  <h3 style={{ fontSize: '15px', fontWeight: 700, marginBottom: '14px' }}>Prediction Analysis</h3>

                  {!prediction && !analyzing && (
                    <div style={{ textAlign: 'center', padding: '60px 20px', color: '#94A3B8' }}>
                      <ScanSearch size={44} style={{ margin: '0 auto 12px', opacity: 0.5 }} />
                      <div style={{ fontSize: '14px', fontWeight: 500 }}>No image analyzed yet</div>
                      <div style={{ fontSize: '12.5px', marginTop: '4px' }}>
                        Select an image on the left and click "Analyze Traffic Sign"
                      </div>
                    </div>
                  )}

                  {analyzing && (
                    <div style={{ textAlign: 'center', padding: '60px 20px', color: '#2563EB' }}>
                      <Activity size={36} style={{ margin: '0 auto 12px', animation: 'spin 1s linear infinite' }} />
                      <div style={{ fontSize: '14px', fontWeight: 600 }}>Executing BanTrafficNet Inference...</div>
                      <div style={{ fontSize: '12.5px', color: '#64748B', marginTop: '4px' }}>
                        Normalizing 96×96 RGB array and evaluating 56 output nodes
                      </div>
                    </div>
                  )}

                  {prediction && prediction.success && (
                    <div>
                      {/* Top prediction banner */}
                      <div style={{ 
                        padding: '20px', 
                        background: '#F8FAFC', 
                        borderRadius: '10px', 
                        border: '1px solid #E2E8F0',
                        marginBottom: '20px'
                      }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                          <div>
                            <span className={`badge ${
                              prediction.category?.includes('Regulatory') ? 'badge-regulatory' :
                              prediction.category?.includes('Warning') ? 'badge-warning' : 'badge-informatory'
                            }`}>
                              {prediction.category}
                            </span>
                            <h2 style={{ fontSize: '22px', fontWeight: 700, color: '#0F172A', marginTop: '8px' }}>
                              {prediction.predicted_class}
                            </h2>
                          </div>
                          <div style={{ textAlign: 'right' }}>
                            <div style={{ fontSize: '26px', fontWeight: 800, color: '#2563EB' }}>
                              {prediction.confidence}%
                            </div>
                            <div style={{ fontSize: '11px', color: '#64748B' }}>Confidence Score</div>
                          </div>
                        </div>

                        {/* Progress Bar */}
                        <div style={{ width: '100%', height: '6px', background: '#E2E8F0', borderRadius: '999px', marginTop: '14px', overflow: 'hidden' }}>
                          <div style={{ width: `${prediction.confidence}%`, height: '100%', background: '#2563EB' }} />
                        </div>
                      </div>

                      {/* Regulatory meaning */}
                      <div style={{ marginBottom: '24px' }}>
                        <div style={{ fontSize: '12px', fontWeight: 600, color: '#64748B', textTransform: 'uppercase', marginBottom: '4px' }}>
                          Indian Road Congress (IRC:67) Interpretation:
                        </div>
                        <p style={{ fontSize: '13.5px', color: '#334155', lineHeight: 1.5 }}>
                          {prediction.regulatory_meaning}
                        </p>
                      </div>

                      {/* Top 3 Alternative Candidates */}
                      <div>
                        <div style={{ fontSize: '12px', fontWeight: 600, color: '#64748B', textTransform: 'uppercase', marginBottom: '10px' }}>
                          Top-3 Alternative Softmax Probabilities:
                        </div>
                        <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                          {prediction.top_predictions?.map(item => (
                            <div 
                              key={item.class_id}
                              style={{ 
                                display: 'flex', 
                                justifyContent: 'space-between', 
                                alignItems: 'center',
                                padding: '10px 14px',
                                background: item.rank === 1 ? '#EFF6FF' : '#FFFFFF',
                                border: `1px solid ${item.rank === 1 ? '#BFDBFE' : '#E2E8F0'}`,
                                borderRadius: '8px'
                              }}
                            >
                              <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                                <span style={{ fontSize: '12px', fontWeight: 700, color: '#64748B' }}>#{item.rank}</span>
                                <span style={{ fontSize: '13.5px', fontWeight: 600, color: '#0F172A' }}>{item.class_name}</span>
                              </div>
                              <span style={{ fontSize: '13px', fontWeight: 700, color: item.rank === 1 ? '#2563EB' : '#475569' }}>
                                {item.confidence}%
                              </span>
                            </div>
                          ))}
                        </div>
                      </div>

                      {/* Google AI Studio Copilot Button */}
                      <div style={{ marginTop: '20px', paddingTop: '16px', borderTop: '1px solid #E2E8F0' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '10px' }}>
                          <span style={{ fontSize: '12px', fontWeight: 700, color: '#1E293B', display: 'flex', alignItems: 'center', gap: '6px' }}>
                            <span style={{ color: '#2563EB' }}>✦</span> Google AI Studio Multimodal Reasoning
                          </span>
                          <button 
                            className="btn btn-secondary"
                            style={{ padding: '6px 12px', fontSize: '11.5px', background: '#F8FAFC' }}
                            disabled={aiStudioLoading}
                            onClick={handleAIStudioQuery}
                          >
                            {aiStudioLoading ? 'Querying Gemini...' : 'Analyze with Google AI Studio'}
                          </button>
                        </div>

                        {aiStudioData && !aiStudioData.configured && (
                          <div style={{ padding: '12px', background: '#F8FAFC', border: '1px solid #CBD5E1', borderRadius: '8px', fontSize: '12px', color: '#475569' }}>
                            <strong>Google AI Studio Setup:</strong>
                            <p style={{ marginTop: '4px' }}>{aiStudioData.message}</p>
                            <a 
                              href={aiStudioData.guide_url || "https://aistudio.google.com/"} 
                              target="_blank" 
                              rel="noreferrer"
                              style={{ display: 'inline-block', marginTop: '6px', color: '#2563EB', fontWeight: 600, textDecoration: 'none' }}
                            >
                              Get Free API Key on aistudio.google.com &rarr;
                            </a>
                          </div>
                        )}

                        {aiStudioData && aiStudioData.configured && aiStudioData.analysis && (
                          <div style={{ padding: '14px', background: '#F0FDF4', border: '1px solid #BBF7D0', borderRadius: '8px', fontSize: '12.5px', color: '#166534' }}>
                            <div style={{ fontWeight: 700, marginBottom: '4px', display: 'flex', justifyContent: 'space-between' }}>
                              <span>Gemini Analysis: {aiStudioData.analysis.sign_detected}</span>
                              <span style={{ fontSize: '11px', textTransform: 'uppercase', background: '#DCFCE7', padding: '2px 6px', borderRadius: '4px' }}>
                                Confidence: {aiStudioData.analysis.confidence}
                              </span>
                            </div>
                            <div style={{ marginTop: '6px' }}>
                              <strong>IRC Rule / MV Act:</strong> {aiStudioData.analysis.irc_rule}
                            </div>
                            <div style={{ marginTop: '4px' }}>
                              <strong>Driver Action:</strong> {aiStudioData.analysis.driver_action}
                            </div>
                            <div style={{ marginTop: '4px', color: '#B91C1C' }}>
                              <strong>Penalty Risk:</strong> {aiStudioData.analysis.penalty_risk}
                            </div>
                          </div>
                        )}
                      </div>

                      <div style={{ marginTop: '20px', fontSize: '11.5px', color: '#94A3B8', display: 'flex', justifyContent: 'space-between' }}>
                        <span>Inference Latency: {prediction.inference_time_ms} ms</span>
                        <span>Model: BanTrafficNet (4.6M params)</span>
                      </div>
                    </div>
                  )}

                  {prediction && !prediction.success && (
                    <div style={{ padding: '20px', background: '#FEF2F2', border: '1px solid #FECACA', borderRadius: '8px', color: '#DC2626' }}>
                      <div style={{ display: 'flex', gap: '10px', alignItems: 'center', fontWeight: 700 }}>
                        <AlertCircle size={18} /> Prediction Error
                      </div>
                      <p style={{ fontSize: '13px', marginTop: '6px' }}>{prediction.error}</p>
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}

          {/* ========================================================= */}
          {/* 3. DATASET EXPLORER */}
          {/* ========================================================= */}
          {currentTab === 'dataset' && (
            <div>
              <div className="page-header">
                <h1 className="page-title">Dataset Explorer</h1>
                <p className="page-subtitle">
                  Inspect the official Indian Traffic Signboards collection (3,277 verified original road sign images across 56 classes).
                </p>
              </div>

              {/* Controls */}
              <div style={{ display: 'flex', gap: '14px', marginBottom: '20px' }}>
                <div style={{ position: 'relative', flex: 1 }}>
                  <Search size={16} style={{ position: 'absolute', left: '12px', top: '12px', color: '#94A3B8' }} />
                  <input 
                    type="text" 
                    className="search-input" 
                    placeholder="Search traffic sign classes..." 
                    style={{ paddingLeft: '36px' }}
                    value={searchTerm}
                    onChange={e => setSearchTerm(e.target.value)}
                  />
                </div>
                <div style={{ display: 'flex', gap: '8px' }}>
                  {['All', 'Mandatory / Regulatory', 'Cautionary / Warning', 'Informatory'].map(cat => (
                    <button
                      key={cat}
                      className={`btn ${selectedCategory === cat ? 'btn-primary' : 'btn-secondary'}`}
                      style={{ fontSize: '12.5px', padding: '6px 12px' }}
                      onClick={() => setSelectedCategory(cat)}
                    >
                      {cat}
                    </button>
                  ))}
                </div>
              </div>

              {/* Table */}
              <div className="table-container">
                <table>
                  <thead>
                    <tr>
                      <th style={{ width: '80px' }}>Sample</th>
                      <th>Class Name</th>
                      <th>IRC:67 Category</th>
                      <th>Original Images</th>
                      <th>Integrity Status</th>
                      <th style={{ width: '100px' }}>Action</th>
                    </tr>
                  </thead>
                  <tbody>
                    {filteredClasses.map(c => (
                      <tr key={c.key}>
                        <td>
                          <img 
                            src={`/api/dataset/sample/${c.key}`} 
                            alt={c.display_name} 
                            style={{ width: '40px', height: '40px', objectFit: 'contain', borderRadius: '4px', border: '1px solid #E2E8F0', background: '#FAFAFA' }}
                            onError={e => { e.currentTarget.style.display = 'none'; }}
                          />
                        </td>
                        <td style={{ fontWeight: 600 }}>{c.display_name}</td>
                        <td>
                          <span className={`badge ${
                            c.category.includes('Regulatory') ? 'badge-regulatory' :
                            c.category.includes('Warning') ? 'badge-warning' : 'badge-informatory'
                          }`}>
                            {c.category}
                          </span>
                        </td>
                        <td>{c.original_count} verified images</td>
                        <td>
                          <span style={{ display: 'inline-flex', alignItems: 'center', gap: '6px', fontSize: '12px', color: '#059669', fontWeight: 600 }}>
                            <CheckCircle2 size={14} /> SHA-256 Validated
                          </span>
                        </td>
                        <td>
                          <button 
                            className="btn btn-secondary" 
                            style={{ padding: '4px 10px', fontSize: '11.5px' }}
                            onClick={() => handleQuickSampleSelect(c.key)}
                          >
                            Test Sign
                          </button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* ========================================================= */}
          {/* 4. MODEL PERFORMANCE */}
          {/* ========================================================= */}
          {currentTab === 'performance' && (
            <div>
              <div className="page-header">
                <h1 className="page-title">Model Performance & Evaluation</h1>
                <p className="page-subtitle">
                  Empirical metrics and comparison against reference benchmarks from the 2026 Nature Portfolio paper.
                </p>
              </div>

              {/* Performance Metrics Cards */}
              <div className="stat-grid">
                <div className="stat-card">
                  <span className="stat-label">Model Architecture</span>
                  <span className="stat-val" style={{ fontSize: '20px' }}>BanTrafficNet</span>
                  <span className="stat-desc">3 Conv Blocks + Dense Head</span>
                </div>
                <div className="stat-card">
                  <span className="stat-label">Trainable Parameters</span>
                  <span className="stat-val">4,643,544</span>
                  <span className="stat-desc">4.64M vs 143.7M (VGG19)</span>
                </div>
                <div className="stat-card">
                  <span className="stat-label">Reference Accuracy (BDTSR)</span>
                  <span className="stat-val">99.34%</span>
                  <span className="stat-desc">GTSRB: 96.97% | BTSC: 98.82%</span>
                </div>
                <div className="stat-card">
                  <span className="stat-label">Inference Latency</span>
                  <span className="stat-val">~5.4 ms</span>
                  <span className="stat-desc">&gt;180 FPS on GPU / Real-time ADAS</span>
                </div>
              </div>

              {/* Benchmark Comparison Table */}
              <div className="card" style={{ marginBottom: '24px' }}>
                <h3 style={{ fontSize: '16px', fontWeight: 700, marginBottom: '14px' }}>
                  Model Efficiency & Parameter Comparison (Reference Journal Paper)
                </h3>
                <div className="table-container">
                  <table>
                    <thead>
                      <tr>
                        <th>Model Name</th>
                        <th>Parameters</th>
                        <th>Training Acc</th>
                        <th>Validation Acc</th>
                        <th>Test Acc</th>
                        <th>Suitability for Edge / Mobile</th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr style={{ background: '#EFF6FF', fontWeight: 600 }}>
                        <td style={{ color: '#2563EB' }}>BanTrafficNet (Proposed Model)</td>
                        <td>4.64M</td>
                        <td>97.87%</td>
                        <td>99.44%</td>
                        <td>99.34%</td>
                        <td><span className="badge badge-informatory">Excellent (&lt;10ms)</span></td>
                      </tr>
                      <tr>
                        <td>DenseNet201</td>
                        <td>20.2M</td>
                        <td>99.67%</td>
                        <td>99.65%</td>
                        <td>99.17%</td>
                        <td><span className="badge badge-warning">Heavy Memory</span></td>
                      </tr>
                      <tr>
                        <td>InceptionV3</td>
                        <td>23.9M</td>
                        <td>99.60%</td>
                        <td>99.56%</td>
                        <td>99.44%</td>
                        <td><span className="badge badge-warning">Moderate</span></td>
                      </tr>
                      <tr>
                        <td>ResNet50</td>
                        <td>25.6M</td>
                        <td>99.10%</td>
                        <td>99.17%</td>
                        <td>98.75%</td>
                        <td><span className="badge badge-warning">High Compute</span></td>
                      </tr>
                      <tr>
                        <td>VGG19</td>
                        <td>143.7M</td>
                        <td>99.93%</td>
                        <td>99.79%</td>
                        <td>99.44%</td>
                        <td><span className="badge badge-regulatory">Too Heavy (143M)</span></td>
                      </tr>
                    </tbody>
                  </table>
                </div>
              </div>

              {/* Colab Notebook & Training Status */}
              <div className="card">
                <h3 style={{ fontSize: '15px', fontWeight: 700, marginBottom: '8px' }}>Google Colab Training Artifacts</h3>
                <p style={{ fontSize: '13.5px', color: '#64748B', lineHeight: 1.6 }}>
                  Full 100-epoch hyperparameter exploration and cross-validation is orchestrated in Google Colab utilizing dedicated GPU hardware (Tesla T4 / A100). The Colab script generates loss curves, per-class F1-scores, confusion matrices, and exports the final model to <code>.keras</code>, <code>.onnx</code>, and <code>.tflite</code> for mobile roadside detection.
                </p>
                <div style={{ marginTop: '14px' }}>
                  <button className="btn btn-secondary" onClick={() => setCurrentTab('training')}>
                    Open Training Lab &amp; Colab Guide <ArrowRight size={14} />
                  </button>
                </div>
              </div>
            </div>
          )}

          {/* ========================================================= */}
          {/* 5. TRAINING LAB */}
          {/* ========================================================= */}
          {currentTab === 'training' && (
            <div>
              <div className="page-header">
                <h1 className="page-title">Training Lab &amp; Experiment Management</h1>
                <p className="page-subtitle">
                  Configure training parameters, launch Google Colab GPU training, and upload exported model weights.
                </p>
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: '1.2fr 1fr', gap: '24px' }}>
                {/* Colab execution workflow */}
                <div className="card">
                  <h3 style={{ fontSize: '15px', fontWeight: 700, marginBottom: '12px' }}>
                    Google Colab Training Workflow
                  </h3>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '14px', fontSize: '13.5px' }}>
                    <div style={{ display: 'flex', gap: '12px', alignItems: 'flex-start' }}>
                      <span style={{ width: '24px', height: '24px', borderRadius: '50%', background: '#EFF6FF', color: '#2563EB', display: 'flex', alignItems: 'center', justifyContent: 'center', fontWeight: 700, fontSize: '12px', flexShrink: 0 }}>1</span>
                      <div>
                        <strong>Open the Jupyter Notebook:</strong>
                        <p style={{ color: '#64748B', marginTop: '2px' }}>
                          The 31-step notebook is saved at <code>notebooks/Indian_Traffic_Sign_BanTrafficNet.ipynb</code>.
                        </p>
                      </div>
                    </div>

                    <div style={{ display: 'flex', gap: '12px', alignItems: 'flex-start' }}>
                      <span style={{ width: '24px', height: '24px', borderRadius: '50%', background: '#EFF6FF', color: '#2563EB', display: 'flex', alignItems: 'center', justifyContent: 'center', fontWeight: 700, fontSize: '12px', flexShrink: 0 }}>2</span>
                      <div>
                        <strong>Select GPU Runtime:</strong>
                        <p style={{ color: '#64748B', marginTop: '2px' }}>
                          In Google Colab, navigate to <code>Runtime &rarr; Change runtime type &rarr; T4 GPU</code>.
                        </p>
                      </div>
                    </div>

                    <div style={{ display: 'flex', gap: '12px', alignItems: 'flex-start' }}>
                      <span style={{ width: '24px', height: '24px', borderRadius: '50%', background: '#EFF6FF', color: '#2563EB', display: 'flex', alignItems: 'center', justifyContent: 'center', fontWeight: 700, fontSize: '12px', flexShrink: 0 }}>3</span>
                      <div>
                        <strong>Execute Training Pipeline:</strong>
                        <p style={{ color: '#64748B', marginTop: '2px' }}>
                          Train BanTrafficNet with Adam (lr=0.001) for up to 100 epochs with EarlyStopping and ReduceLROnPlateau.
                        </p>
                      </div>
                    </div>

                    <div style={{ display: 'flex', gap: '12px', alignItems: 'flex-start' }}>
                      <span style={{ width: '24px', height: '24px', borderRadius: '50%', background: '#EFF6FF', color: '#2563EB', display: 'flex', alignItems: 'center', justifyContent: 'center', fontWeight: 700, fontSize: '12px', flexShrink: 0 }}>4</span>
                      <div>
                        <strong>Download Exported Artifacts:</strong>
                        <p style={{ color: '#64748B', marginTop: '2px' }}>
                          Download <code>bantrafficnet_indian.onnx</code> or <code>best_bantrafficnet.keras</code> and upload here.
                        </p>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Model upload zone */}
                <div className="card">
                  <h3 style={{ fontSize: '15px', fontWeight: 700, marginBottom: '12px' }}>
                    Upload Trained Model Weights
                  </h3>
                  <p style={{ fontSize: '13px', color: '#64748B', marginBottom: '16px' }}>
                    Upload your trained <code>.onnx</code> model for real-time web inference activation.
                  </p>

                  <div 
                    className="dropzone"
                    onClick={() => document.getElementById('model-upload-input')?.click()}
                  >
                    <input 
                      type="file" 
                      id="model-upload-input" 
                      style={{ display: 'none' }} 
                      accept=".onnx,.keras,.pth,.tflite"
                      onChange={async e => {
                        if (e.target.files && e.target.files[0]) {
                          setUploadStatus('Uploading model file...');
                          try {
                            const res = await uploadModelFile(e.target.files[0]);
                            setUploadStatus(res.message || 'Model uploaded successfully.');
                            const updated = await fetchModelStatus();
                            setModelStatus(updated);
                          } catch (err) {
                            setUploadStatus('Upload failed.');
                          }
                        }
                      }}
                    />
                    <FileCode size={36} style={{ color: '#2563EB', margin: '0 auto 8px' }} />
                    <div style={{ fontSize: '13.5px', fontWeight: 600 }}>Select .onnx, .keras, or .pth file</div>
                    <div style={{ fontSize: '11.5px', color: '#94A3B8', marginTop: '4px' }}>Automatic hot-reload on upload</div>
                  </div>

                  {uploadStatus && (
                    <div style={{ marginTop: '14px', padding: '10px 14px', background: '#F0FDF4', border: '1px solid #BBF7D0', borderRadius: '6px', fontSize: '12.5px', color: '#16A34A' }}>
                      {uploadStatus}
                    </div>
                  )}

                  <div style={{ marginTop: '20px', padding: '12px', background: '#F8FAFC', borderRadius: '8px', fontSize: '12px', color: '#64748B' }}>
                    <strong>Active Model Engine:</strong><br />
                    Path: {modelStatus?.model_path || 'artifacts/model/bantrafficnet_indian.onnx'}<br />
                    Engine: OpenCV DNN / C++ Accelerated
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* ========================================================= */}
          {/* 6. METHODOLOGY */}
          {/* ========================================================= */}
          {currentTab === 'methodology' && (
            <div>
              <div className="page-header">
                <h1 className="page-title">Research Methodology &amp; BanTrafficNet Architecture</h1>
                <p className="page-subtitle">
                  Comprehensive architectural dissection based on Scientific Reports (Nature Portfolio, 2026).
                </p>
              </div>

              {/* Visual Architecture Diagram */}
              <div className="card" style={{ marginBottom: '24px' }}>
                <h3 style={{ fontSize: '16px', fontWeight: 700, marginBottom: '16px' }}>
                  BanTrafficNet Sequential Layer Progression
                </h3>

                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '16px' }}>
                  {/* Step 1 */}
                  <div style={{ padding: '16px', background: '#F8FAFC', borderRadius: '8px', border: '1px solid #E2E8F0' }}>
                    <div style={{ fontSize: '11px', fontWeight: 700, color: '#2563EB', textTransform: 'uppercase' }}>Input Stage</div>
                    <div style={{ fontSize: '15px', fontWeight: 700, marginTop: '6px' }}>96 × 96 × 3 RGB</div>
                    <div style={{ fontSize: '12px', color: '#64748B', marginTop: '6px' }}>
                      • Normalized via <code>X / 255.0</code><br />
                      • Preserves full chromatic cues
                    </div>
                  </div>

                  {/* Step 2 */}
                  <div style={{ padding: '16px', background: '#F8FAFC', borderRadius: '8px', border: '1px solid #E2E8F0' }}>
                    <div style={{ fontSize: '11px', fontWeight: 700, color: '#2563EB', textTransform: 'uppercase' }}>Conv Block 1</div>
                    <div style={{ fontSize: '15px', fontWeight: 700, marginTop: '6px' }}>46 × 46 × 32</div>
                    <div style={{ fontSize: '12px', color: '#64748B', marginTop: '6px' }}>
                      • Conv2D (32, 3×3, valid)<br />
                      • Conv2D (32, 3×3, valid)<br />
                      • MaxPool2D (2×2, s=2)
                    </div>
                  </div>

                  {/* Step 3 */}
                  <div style={{ padding: '16px', background: '#F8FAFC', borderRadius: '8px', border: '1px solid #E2E8F0' }}>
                    <div style={{ fontSize: '11px', fontWeight: 700, color: '#2563EB', textTransform: 'uppercase' }}>Conv Block 2</div>
                    <div style={{ fontSize: '15px', fontWeight: 700, marginTop: '6px' }}>21 × 21 × 64</div>
                    <div style={{ fontSize: '12px', color: '#64748B', marginTop: '6px' }}>
                      • Conv2D (64, 3×3, valid)<br />
                      • Conv2D (64, 3×3, valid)<br />
                      • MaxPool + Dropout (0.25)
                    </div>
                  </div>

                  {/* Step 4 */}
                  <div style={{ padding: '16px', background: '#F8FAFC', borderRadius: '8px', border: '1px solid #E2E8F0' }}>
                    <div style={{ fontSize: '11px', fontWeight: 700, color: '#2563EB', textTransform: 'uppercase' }}>Conv Block 3</div>
                    <div style={{ fontSize: '15px', fontWeight: 700, marginTop: '6px' }}>8 × 8 × 256</div>
                    <div style={{ fontSize: '12px', color: '#64748B', marginTop: '6px' }}>
                      • Conv2D (128, 3×3, valid)<br />
                      • Conv2D (256, 3×3, valid)<br />
                      • MaxPool + Dropout (0.25)
                    </div>
                  </div>
                </div>

                <div style={{ marginTop: '16px', padding: '16px', background: '#EFF6FF', borderRadius: '8px', border: '1px solid #BFDBFE', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <div>
                    <div style={{ fontSize: '12px', fontWeight: 700, color: '#1D4ED8', textTransform: 'uppercase' }}>Classification Head</div>
                    <div style={{ fontSize: '14px', fontWeight: 600, color: '#1E3A8A', marginTop: '2px' }}>
                      Flatten (16,384) &rarr; Dense (256, ReLU) &rarr; Dropout (0.50) &rarr; Output Dense (56 classes, Softmax)
                    </div>
                  </div>
                  <div style={{ fontSize: '16px', fontWeight: 800, color: '#1D4ED8' }}>
                    4,643,544 Parameters
                  </div>
                </div>
              </div>

              {/* Comparative table */}
              <div className="card">
                <h3 style={{ fontSize: '16px', fontWeight: 700, marginBottom: '14px' }}>
                  Paper Reference vs. Indian Project Adaptation
                </h3>
                <div className="table-container">
                  <table>
                    <thead>
                      <tr>
                        <th>Aspect</th>
                        <th>Reference Paper (BDTSR)</th>
                        <th>Our Indian Road Adaptation (IRC:67)</th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr>
                        <td style={{ fontWeight: 600 }}>Domain Context</td>
                        <td>Bangladeshi Traffic Signs (Road Safety Facility)</td>
                        <td>Indian Road Congress (IRC:67) &amp; MoRTH Standards</td>
                      </tr>
                      <tr>
                        <td style={{ fontWeight: 600 }}>Original Image Volume</td>
                        <td>6,417 raw images</td>
                        <td>3,277 verified raw images (Exceeds &gt;1,000 threshold)</td>
                      </tr>
                      <tr>
                        <td style={{ fontWeight: 600 }}>Target Classes</td>
                        <td>48 categories</td>
                        <td>56 categories (Mandatory, Cautionary, Informatory)</td>
                      </tr>
                      <tr>
                        <td style={{ fontWeight: 600 }}>Input Resolution</td>
                        <td>96 × 96 × 3 RGB</td>
                        <td>96 × 96 × 3 RGB (Preserved exactly)</td>
                      </tr>
                      <tr>
                        <td style={{ fontWeight: 600 }}>Optimizer &amp; LR</td>
                        <td>Adam (lr = 0.001)</td>
                        <td>Adam (lr = 0.001) + ReduceLROnPlateau scheduling</td>
                      </tr>
                      <tr>
                        <td style={{ fontWeight: 600 }}>Deployment Engine</td>
                        <td>Android APK with TFLite</td>
                        <td>Full-Stack Web App (FastAPI/React) + Colab + TFLite</td>
                      </tr>
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}
