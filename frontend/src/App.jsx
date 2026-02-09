import { useState, useRef, useEffect } from 'react'
import './App.css'

const WORKFLOW_STEPS = [
  'Ingestion',
  'Context',
  'Memory',
  'Retriever',
  'Recommender',
  'Reviewer'
]

function App() {
  const [patientHash, setPatientHash] = useState('')
  const [deviceId, setDeviceId] = useState('web_device_01')
  const [textNotes, setTextNotes] = useState('')
  const [audioFile, setAudioFile] = useState(null)
  const [imageFiles, setImageFiles] = useState([])
  const [docFiles, setDocFiles] = useState([])

  const [isProcessing, setIsProcessing] = useState(false)
  const [logs, setLogs] = useState([])
  const [currentStep, setCurrentStep] = useState(-1)
  const [completedSteps, setCompletedSteps] = useState([])
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)

  const logPanelRef = useRef(null)
  const wsRef = useRef(null)

  // Auto-scroll logs
  useEffect(() => {
    if (logPanelRef.current) {
      logPanelRef.current.scrollTop = logPanelRef.current.scrollHeight
    }
  }, [logs])

  const addLog = (message, type = 'info') => {
    const time = new Date().toLocaleTimeString('en-US', {
      hour12: false,
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit'
    })
    setLogs(prev => [...prev, { time, message, type }])
  }

  const connectWebSocket = () => {
    const ws = new WebSocket(`ws://${window.location.hostname}:8000/ws/logs`)

    ws.onopen = () => {
      addLog('Connected to server', 'success')
    }

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data)

      if (data.type === 'log') {
        addLog(data.message, data.level || 'info')
      } else if (data.type === 'step') {
        const stepIndex = WORKFLOW_STEPS.findIndex(s =>
          s.toLowerCase() === data.step.toLowerCase()
        )
        if (stepIndex >= 0) {
          setCurrentStep(stepIndex)
          if (data.status === 'complete') {
            setCompletedSteps(prev => [...prev, stepIndex])
          }
        }
      } else if (data.type === 'thinking') {
        addLog(data.message, 'thinking')
      } else if (data.type === 'result') {
        setResult(data.data)
      } else if (data.type === 'error') {
        addLog(data.message, 'error')
        setError(data.message)
      } else if (data.type === 'complete') {
        addLog('Workflow completed successfully!', 'success')
        setIsProcessing(false)
      }
    }

    ws.onerror = () => {
      addLog('WebSocket error', 'error')
    }

    ws.onclose = () => {
      addLog('Disconnected from server', 'info')
    }

    wsRef.current = ws
    return ws
  }

  const handleSubmit = async (e) => {
    e.preventDefault()

    if (!patientHash.trim()) {
      setError('Patient ID is required')
      return
    }

    setIsProcessing(true)
    setLogs([])
    setCurrentStep(-1)
    setCompletedSteps([])
    setResult(null)
    setError(null)

    addLog('Starting patient data processing...', 'info')

    try {
      // Connect WebSocket for live logs
      const ws = connectWebSocket()

      // Wait for connection
      await new Promise(resolve => setTimeout(resolve, 500))

      // Build form data
      const formData = new FormData()
      formData.append('patient_hash', patientHash)
      formData.append('device_id', deviceId)
      formData.append('raw_text', textNotes)
      formData.append('offline', 'false')

      if (audioFile) formData.append('audio', audioFile)
      imageFiles.forEach(f => formData.append('images', f))
      docFiles.forEach(f => formData.append('documents', f))

      addLog('Sending data to backend...', 'info')

      const response = await fetch('/api/process/files', {
        method: 'POST',
        body: formData
      })

      const data = await response.json()

      if (data.success) {
        setResult(data)
        addLog('Processing complete!', 'success')
      } else {
        throw new Error(data.error || 'Processing failed')
      }

    } catch (err) {
      addLog(`Error: ${err.message}`, 'error')
      setError(err.message)
    } finally {
      setIsProcessing(false)
      if (wsRef.current) {
        wsRef.current.close()
      }
    }
  }

  return (
    <div className="app">
      {/* Header */}
      <header className="header">
        <h1>Convolve</h1>
        <p>Patient Analysis & Recommendation System</p>
      </header>

      {/* Input Form */}
      <section className="section">
        <h2>Patient Data Entry</h2>
        <form onSubmit={handleSubmit}>
          <div className="form-grid">
            <div className="form-group">
              <label>Patient ID</label>
              <input
                type="text"
                className="form-control"
                placeholder="e.g. PAT-2026-001"
                value={patientHash}
                onChange={e => setPatientHash(e.target.value)}
                required
              />
            </div>
            <div className="form-group">
              <label>Device ID</label>
              <input
                type="text"
                className="form-control"
                value={deviceId}
                onChange={e => setDeviceId(e.target.value)}
              />
            </div>
            <div className="form-group full-width">
              <label>Clinical Notes</label>
              <textarea
                className="form-control"
                placeholder="Describe symptoms, history, observations..."
                value={textNotes}
                onChange={e => setTextNotes(e.target.value)}
              />
            </div>
          </div>

          {/* File Uploads */}
          <div className="upload-grid">
            <label className="upload-box">
              <div className="upload-box-title">Voice Recording</div>
              <div className="upload-box-desc">Upload audio (MP3/WAV)</div>
              {audioFile && <div className="file-list">{audioFile.name}</div>}
              <input
                type="file"
                accept="audio/*"
                onChange={e => setAudioFile(e.target.files[0])}
              />
            </label>

            <label className="upload-box">
              <div className="upload-box-title">Patient Photos</div>
              <div className="upload-box-desc">Upload images</div>
              {imageFiles.length > 0 && (
                <div className="file-list">{imageFiles.length} file(s)</div>
              )}
              <input
                type="file"
                accept="image/*"
                multiple
                onChange={e => setImageFiles(Array.from(e.target.files))}
              />
            </label>

            <label className="upload-box">
              <div className="upload-box-title">Medical Documents</div>
              <div className="upload-box-desc">Upload PDFs/reports</div>
              {docFiles.length > 0 && (
                <div className="file-list">{docFiles.length} file(s)</div>
              )}
              <input
                type="file"
                accept=".pdf,.txt,.doc,.docx"
                multiple
                onChange={e => setDocFiles(Array.from(e.target.files))}
              />
            </label>
          </div>

          <button
            type="submit"
            className="btn-primary"
            disabled={isProcessing}
          >
            {isProcessing ? 'Processing...' : 'Process Patient Data'}
          </button>
        </form>

        {error && (
          <div style={{
            marginTop: 20,
            padding: 15,
            background: 'rgba(239, 68, 68, 0.1)',
            border: '1px solid var(--error)',
            borderRadius: 8,
            color: 'var(--error)'
          }}>
            {error}
          </div>
        )}
      </section>

      {/* Workflow Status */}
      {(isProcessing || completedSteps.length > 0) && (
        <section className="section">
          <h2>Workflow Progress</h2>
          <div className="workflow-status">
            {WORKFLOW_STEPS.map((step, idx) => (
              <div
                key={step}
                className={`workflow-step ${currentStep === idx ? 'active' : ''
                  } ${completedSteps.includes(idx) ? 'completed' : ''}`}
              >
                {step}
              </div>
            ))}
          </div>
        </section>
      )}

      {/* Logs Panel */}
      {logs.length > 0 && (
        <section className="section">
          <h2>Processing Logs</h2>
          <div className="log-panel" ref={logPanelRef}>
            {logs.map((log, idx) => (
              <div key={idx} className={`log-entry ${log.type}`}>
                <span className="log-time">{log.time}</span>
                <span className="log-message">{log.message}</span>
              </div>
            ))}
            {isProcessing && (
              <div className="log-entry info loading">
                <span className="log-time">...</span>
                <span className="log-message">Processing...</span>
              </div>
            )}
          </div>
        </section>
      )}

      {/* Results */}
      {result && (
        <section className="section">
          <h2>Analysis Results</h2>

          {result.analysis && (
            <div className="result-block">
              <h3>Case Analysis</h3>
              <p style={{ color: 'var(--text-light)', whiteSpace: 'pre-wrap' }}>
                {result.analysis.summary}
              </p>
              <p style={{ marginTop: 10, color: 'var(--text-muted)' }}>
                {result.analysis.explanation}
              </p>
            </div>
          )}

          {result.verdict && (
            <div className={result.approved ? 'verdict-approved' : 'verdict-attention'}>
              <p style={{
                fontWeight: 600,
                fontSize: '1.25rem',
                color: result.approved ? '#4ade80' : '#f87171',
                marginBottom: 8
              }}>
                {result.approved ? '✓ Approved' : '⚠ Needs Attention'}
              </p>
              {result.verdict.reasoning && (
                <p style={{ color: '#94a3b8', fontSize: '0.95rem', lineHeight: 1.6 }}>
                  {result.verdict.reasoning}
                </p>
              )}
            </div>
          )}

          {/* Raw JSON */}
          <details style={{ marginTop: 20 }}>
            <summary style={{ cursor: 'pointer', color: 'var(--text-muted)' }}>
              View Raw JSON
            </summary>
            <pre style={{
              marginTop: 10,
              padding: 15,
              background: 'var(--bg-dark)',
              borderRadius: 8,
              overflow: 'auto',
              fontSize: '0.8rem',
              color: 'var(--text-muted)'
            }}>
              {JSON.stringify(result, null, 2)}
            </pre>
          </details>
        </section>
      )}

      {/* Chat Interface - appears after results */}
      {result && (
        <ChatSection result={result} />
      )}
    </div>
  )
}

// Chat Component
function ChatSection({ result }) {
  const [messages, setMessages] = useState([
    { role: 'bot', text: 'Analysis complete! Ask me anything about the results, recommendations, or patient data.' }
  ])
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const chatRef = useRef(null)

  useEffect(() => {
    if (chatRef.current) {
      chatRef.current.scrollTop = chatRef.current.scrollHeight
    }
  }, [messages])

  const sendMessage = async () => {
    if (!input.trim() || isLoading) return

    const userMessage = input.trim()
    setInput('')
    setMessages(prev => [...prev, { role: 'user', text: userMessage }])
    setIsLoading(true)

    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: userMessage,
          context: {
            patient_hash: result.patient_hash,
            analysis: result.analysis,
            verdict: result.verdict
          }
        })
      })

      const data = await response.json()
      setMessages(prev => [...prev, { role: 'bot', text: data.reply }])
    } catch (err) {
      setMessages(prev => [...prev, { role: 'bot', text: 'Sorry, I encountered an error. Please try again.' }])
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <section className="section">
      <h2>AI Assistant</h2>
      <div style={{
        display: 'flex',
        flexDirection: 'column',
        height: '400px'
      }}>
        <div
          ref={chatRef}
          style={{
            flex: 1,
            overflowY: 'auto',
            background: 'var(--bg-dark)',
            borderRadius: 12,
            padding: 20,
            marginBottom: 15
          }}
        >
          {messages.map((msg, idx) => (
            <div
              key={idx}
              style={{
                maxWidth: '80%',
                marginBottom: 12,
                padding: '12px 16px',
                borderRadius: 12,
                ...(msg.role === 'user' ? {
                  marginLeft: 'auto',
                  background: 'var(--accent)',
                  color: 'white',
                  borderBottomRightRadius: 4
                } : {
                  background: 'var(--input-dark)',
                  color: 'var(--text-light)',
                  borderBottomLeftRadius: 4
                })
              }}
            >
              {msg.text}
            </div>
          ))}
          {isLoading && (
            <div style={{
              padding: '12px 16px',
              background: 'var(--input-dark)',
              borderRadius: 12,
              color: 'var(--text-muted)',
              maxWidth: '80%'
            }}>
              Thinking...
            </div>
          )}
        </div>
        <div style={{ display: 'flex', gap: 10 }}>
          <input
            type="text"
            className="form-control"
            placeholder="Ask about the analysis..."
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyPress={e => e.key === 'Enter' && sendMessage()}
            style={{ flex: 1 }}
          />
          <button
            onClick={sendMessage}
            disabled={isLoading}
            style={{
              padding: '0 25px',
              background: 'var(--accent)',
              border: 'none',
              borderRadius: 10,
              color: 'white',
              fontWeight: 600,
              cursor: isLoading ? 'not-allowed' : 'pointer'
            }}
          >
            Send
          </button>
        </div>
      </div>
    </section>
  )
}

export default App
