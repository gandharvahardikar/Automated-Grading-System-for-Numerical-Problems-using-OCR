import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

const API_BASE_URL = 'http://localhost:5000';

interface GradingResults {
  total_score: number;
  max_total_score: number;
  percentage: number;
  question_grades: Record<string, any>;
  summary: any;
}

function App() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [processing, setProcessing] = useState(false);
  const [results, setResults] = useState<any>(null);
  const [error, setError] = useState('');

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setSelectedFile(file);
      setResults(null);
      setError('');
    }
  };

  const processFile = async () => {
    if (!selectedFile) {
      setError('Please select a file first');
      return;
    }

    setUploading(true);
    setError('');

    try {
      // Upload file
      const formData = new FormData();
      formData.append('file', selectedFile);
      formData.append('paper_type', 'answer');

      const uploadResponse = await axios.post(`${API_BASE_URL}/api/upload`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });

      const { filename, file_id } = uploadResponse.data;
      setUploading(false);
      setProcessing(true);

      // Process file
      const processResponse = await axios.post(`${API_BASE_URL}/api/process-paper`, {
        filename,
        file_id,
        paper_type: 'answer'
      });

      setResults(processResponse.data.processing_results);
    } catch (err: any) {
      setError(err.response?.data?.error || 'An error occurred');
    } finally {
      setUploading(false);
      setProcessing(false);
    }
  };

  const gradingResults = results?.grading_results;

  return (
    <div className="App">
      <div style={{ padding: '40px', maxWidth: '1200px', margin: '0 auto' }}>
        {/* Header */}
        <div style={{ 
          textAlign: 'center', 
          marginBottom: '40px',
          background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
          color: 'white',
          padding: '30px',
          borderRadius: '15px'
        }}>
          <h1 style={{ fontSize: '2.5em', margin: '0 0 10px 0' }}>
            🎓 OCR-Based Automated Exam Grading System
          </h1>
          <p style={{ fontSize: '1.2em', margin: '0' }}>
            Upload handwritten exam papers for automatic OCR processing and intelligent grading
          </p>
        </div>

        {/* File Upload Section */}
        <div style={{ 
          background: '#f8f9fa',
          padding: '30px',
          borderRadius: '10px',
          marginBottom: '30px',
          border: '2px dashed #ccc'
        }}>
          <h3 style={{ marginTop: '0' }}>📄 Upload Answer Paper</h3>
          
          <input
            type="file"
            accept="image/*,.pdf"
            onChange={handleFileSelect}
            style={{ 
              width: '100%',
              padding: '10px',
              margin: '10px 0',
              border: '1px solid #ddd',
              borderRadius: '5px'
            }}
          />

          {selectedFile && (
            <div style={{ 
              background: '#d4edda',
              color: '#155724',
              padding: '10px',
              borderRadius: '5px',
              margin: '10px 0',
              border: '1px solid #c3e6cb'
            }}>
              📁 Selected: <strong>{selectedFile.name}</strong> ({(selectedFile.size / 1024 / 1024).toFixed(2)} MB)
            </div>
          )}

          <button
            onClick={processFile}
            disabled={!selectedFile || uploading || processing}
            style={{
              background: uploading || processing ? '#ccc' : 'linear-gradient(45deg, #2196F3 30%, #21CBF3 90%)',
              color: 'white',
              padding: '15px 30px',
              border: 'none',
              borderRadius: '25px',
              fontSize: '1.1em',
              width: '100%',
              cursor: uploading || processing ? 'not-allowed' : 'pointer',
              transition: 'transform 0.2s'
            }}
          >
            {uploading ? '📤 Uploading...' : processing ? '🔍 Processing OCR & Grading...' : '🚀 Process & Grade Paper'}
          </button>

          {(uploading || processing) && (
            <div style={{ 
              margin: '20px 0',
              textAlign: 'center',
              color: '#666'
            }}>
              <div style={{ 
                width: '100%',
                height: '4px',
                background: '#e0e0e0',
                borderRadius: '2px',
                overflow: 'hidden',
                marginBottom: '10px'
              }}>
                <div style={{
                  width: '100%',
                  height: '100%',
                  background: 'linear-gradient(45deg, #2196F3, #21CBF3)',
                  animation: 'loading 1.5s infinite'
                }}></div>
              </div>
              {uploading ? 'Uploading file to server...' : 'Running OCR and grading analysis...'}
            </div>
          )}
        </div>

        {/* Error Display */}
        {error && (
          <div style={{
            background: '#f8d7da',
            color: '#721c24',
            padding: '15px',
            borderRadius: '5px',
            margin: '20px 0',
            border: '1px solid #f5c6cb'
          }}>
            ❌ {error}
          </div>
        )}

        {/* Results Display */}
        {results && (
          <div>
            {/* OCR Results */}
            <div style={{ 
              background: 'white',
              padding: '20px',
              borderRadius: '10px',
              marginBottom: '20px',
              boxShadow: '0 2px 10px rgba(0,0,0,0.1)'
            }}>
              <h3>🔍 OCR Extraction Results</h3>
              <details>
                <summary style={{ cursor: 'pointer', padding: '10px 0' }}>
                  📝 Click to view extracted text from paper
                </summary>
                <div style={{ 
                  background: '#f1f1f1',
                  padding: '15px',
                  borderRadius: '5px',
                  fontFamily: 'monospace',
                  whiteSpace: 'pre-wrap',
                  fontSize: '0.9em'
                }}>
                  {results.ocr_results?.full_text || 'No text extracted'}
                </div>
              </details>
            </div>

            {/* Grading Results */}
            {gradingResults && (
              <div>
                <h2 style={{ color: '#2196F3' }}>🎯 Automatic Grading Results</h2>

                {/* Overall Score Card */}
                <div style={{
                  background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                  color: 'white',
                  padding: '30px',
                  borderRadius: '15px',
                  marginBottom: '20px',
                  textAlign: 'center'
                }}>
                  <div style={{ display: 'flex', justifyContent: 'space-around', flexWrap: 'wrap', gap: '20px' }}>
                    <div>
                      <h3 style={{ margin: '0 0 10px 0' }}>Total Score</h3>
                      <h1 style={{ margin: '0', fontSize: '2.5em' }}>
                        {gradingResults.total_score} / {gradingResults.max_total_score}
                      </h1>
                    </div>
                    <div>
                      <h3 style={{ margin: '0 0 10px 0' }}>Percentage</h3>
                      <h1 style={{ margin: '0', fontSize: '2.5em' }}>
                        {gradingResults.percentage.toFixed(1)}%
                      </h1>
                    </div>
                    <div>
                      <h3 style={{ margin: '0 0 10px 0' }}>Grade</h3>
                      <h1 style={{ margin: '0', fontSize: '2.5em' }}>
                        {gradingResults.percentage >= 90 ? 'A+' : 
                         gradingResults.percentage >= 80 ? 'A' :
                         gradingResults.percentage >= 70 ? 'B' :
                         gradingResults.percentage >= 60 ? 'C' :
                         gradingResults.percentage >= 50 ? 'D' : 'F'}
                      </h1>
                    </div>
                    <div>
                      <h3 style={{ margin: '0 0 10px 0' }}>Status</h3>
                      <h1 style={{ margin: '0', fontSize: '2.5em' }}>
                        {gradingResults.percentage >= 50 ? '✅ PASS' : '❌ FAIL'}
                      </h1>
                    </div>
                  </div>
                </div>

                {/* Question-wise Analysis */}
                <div style={{ 
                  background: 'white',
                  padding: '20px',
                  borderRadius: '10px',
                  marginBottom: '20px',
                  boxShadow: '0 2px 10px rgba(0,0,0,0.1)'
                }}>
                  <h3>📊 Detailed Question Analysis</h3>
                  <div style={{ overflowX: 'auto' }}>
                    <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                      <thead>
                        <tr style={{ background: '#f8f9fa' }}>
                          <th style={{ padding: '12px', border: '1px solid #ddd', textAlign: 'left' }}>Question</th>
                          <th style={{ padding: '12px', border: '1px solid #ddd', textAlign: 'left' }}>Score</th>
                          <th style={{ padding: '12px', border: '1px solid #ddd', textAlign: 'left' }}>Percentage</th>
                          <th style={{ padding: '12px', border: '1px solid #ddd', textAlign: 'left' }}>Status</th>
                          <th style={{ padding: '12px', border: '1px solid #ddd', textAlign: 'left' }}>Feedback</th>
                        </tr>
                      </thead>
                      <tbody>
                        {Object.entries(gradingResults.question_grades).map(([qId, grade]: [string, any]) => (
                          <tr key={qId}>
                            <td style={{ padding: '12px', border: '1px solid #ddd', fontWeight: 'bold' }}>{qId}</td>
                            <td style={{ padding: '12px', border: '1px solid #ddd' }}>
                              {grade.score.toFixed(1)} / {grade.max_score}
                            </td>
                            <td style={{ padding: '12px', border: '1px solid #ddd' }}>
                              <span style={{
                                background: grade.percentage >= 80 ? '#d4edda' : 
                                           grade.percentage >= 60 ? '#fff3cd' : '#f8d7da',
                                color: grade.percentage >= 80 ? '#155724' : 
                                       grade.percentage >= 60 ? '#856404' : '#721c24',
                                padding: '4px 8px',
                                borderRadius: '12px',
                                fontSize: '0.9em'
                              }}>
                                {grade.percentage.toFixed(1)}%
                              </span>
                            </td>
                            <td style={{ padding: '12px', border: '1px solid #ddd', fontSize: '1.5em' }}>
                              {grade.percentage >= 70 ? '✅' : '❌'}
                            </td>
                            <td style={{ padding: '12px', border: '1px solid #ddd' }}>
                              {grade.feedback?.map((fb: string, idx: number) => (
                                <div key={idx} style={{ 
                                  fontSize: '0.9em',
                                  color: fb.includes('✅') ? '#28a745' : 
                                         fb.includes('⚠️') ? '#ffc107' : '#6c757d',
                                  marginBottom: '2px'
                                }}>
                                  {fb}
                                </div>
                              ))}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>

                {/* Performance Summary */}
                <div style={{ 
                  background: 'white',
                  padding: '20px',
                  borderRadius: '10px',
                  boxShadow: '0 2px 10px rgba(0,0,0,0.1)'
                }}>
                  <h3>📈 Performance Analysis & Recommendations</h3>
                  
                  <div style={{ display: 'flex', gap: '20px', marginBottom: '20px', flexWrap: 'wrap' }}>
                    <div style={{ flex: 1, textAlign: 'center', padding: '15px', border: '1px solid #ddd', borderRadius: '8px' }}>
                      <h2 style={{ color: '#2196F3', margin: '0' }}>{gradingResults.summary.correct_questions}</h2>
                      <p style={{ margin: '5px 0 0 0', color: '#666' }}>Questions Correct</p>
                    </div>
                    <div style={{ flex: 1, textAlign: 'center', padding: '15px', border: '1px solid #ddd', borderRadius: '8px' }}>
                      <h2 style={{ color: '#2196F3', margin: '0' }}>{gradingResults.summary.total_questions}</h2>
                      <p style={{ margin: '5px 0 0 0', color: '#666' }}>Total Questions</p>
                    </div>
                    <div style={{ flex: 1, textAlign: 'center', padding: '15px', border: '1px solid #ddd', borderRadius: '8px' }}>
                      <h2 style={{ color: '#2196F3', margin: '0' }}>{gradingResults.summary.accuracy}%</h2>
                      <p style={{ margin: '5px 0 0 0', color: '#666' }}>Overall Accuracy</p>
                    </div>
                    <div style={{ flex: 1, textAlign: 'center', padding: '15px', border: '1px solid #ddd', borderRadius: '8px' }}>
                      <h2 style={{ margin: '0' }}>
                        {gradingResults.summary.accuracy >= 85 ? '🌟' :
                         gradingResults.summary.accuracy >= 70 ? '👍' :
                         gradingResults.summary.accuracy >= 50 ? '⚡' : '📚'}
                      </h2>
                      <p style={{ margin: '5px 0 0 0', color: '#666' }}>Performance</p>
                    </div>
                  </div>

                  <hr style={{ margin: '20px 0', border: 'none', borderTop: '1px solid #eee' }} />

                  <div style={{ display: 'flex', gap: '20px', flexWrap: 'wrap' }}>
                    {gradingResults.summary.strengths?.length > 0 && (
                      <div style={{ flex: 1, minWidth: '300px' }}>
                        <h4 style={{ color: '#28a745', margin: '0 0 10px 0' }}>
                          ✅ Strengths & Good Performance
                        </h4>
                        <div style={{ 
                          background: '#d4edda',
                          padding: '15px',
                          borderRadius: '8px',
                          border: '1px solid #c3e6cb'
                        }}>
                          {gradingResults.summary.strengths.map((strength: string, idx: number) => (
                            <p key={idx} style={{ margin: '5px 0', color: '#155724' }}>
                              • {strength}
                            </p>
                          ))}
                        </div>
                      </div>
                    )}

                    {gradingResults.summary.improvements?.length > 0 && (
                      <div style={{ flex: 1, minWidth: '300px' }}>
                        <h4 style={{ color: '#ffc107', margin: '0 0 10px 0' }}>
                          🎯 Areas for Improvement
                        </h4>
                        <div style={{ 
                          background: '#fff3cd',
                          padding: '15px',
                          borderRadius: '8px',
                          border: '1px solid #ffeaa7'
                        }}>
                          {gradingResults.summary.improvements.map((improvement: string, idx: number) => (
                            <p key={idx} style={{ margin: '5px 0', color: '#856404' }}>
                              • {improvement}
                            </p>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {/* Footer */}
        <div style={{ 
          marginTop: '40px',
          paddingTop: '20px',
          borderTop: '1px solid #eee',
          textAlign: 'center',
          color: '#666'
        }}>
          🤖 Powered by Advanced OCR Technology & AI-Based Grading System
        </div>
      </div>

      <style>{`
        @keyframes loading {
          0% { transform: translateX(-100%); }
          100% { transform: translateX(100%); }
        }
      `}</style>
    </div>
  );
}

export default App;
