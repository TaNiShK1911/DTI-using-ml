import React, { useState } from 'react'
import axios from 'axios'

interface PredictionResult {
  affinity: number
  uncertainty: number
  confidence_level: string
  attention_weights: number[][]
  detailed_explanation: {
    affinity_calculation: {
      raw_prediction: number
      final_affinity: number
      interpretation: string
      binding_strength: string
    }
    uncertainty_breakdown: {
      total_uncertainty: number
      epistemic_uncertainty: number
      aleatoric_uncertainty: number
      confidence_level: string
      confidence_description: string
      explanation: {
        epistemic: string
        aleatoric: string
        total: string
      }
    }
    evidential_parameters: {
      gamma: number
      nu: number
      alpha: number
      beta: number
      parameter_meanings: {
        gamma: string
        nu: string
        alpha: string
        beta: string
      }
    }
    attention_analysis: {
      attention_shape: string
      max_attention: number
      min_attention: number
      interpretation: string
    }
    input_analysis: {
      smiles_length: number
      sequence_length: number
      smiles_complexity: any
      sequence_composition: any
    }
  }
}

const EXAMPLES = {
  aspirin: {
    smiles: 'CC(=O)Oc1ccccc1C(=O)O',
    sequence: 'MKKFFDSRREQGGSGLGSGSSGGGGSTSGLGSGYIGRVFGIGRQQVTVDEVLAEGGFAIVFLVRTSNGMKCALKRMFVNNEHDLQVCKREIQIMRDLSGHKNIVGYIDSSINNVSSGDVWEVLILMDFCRGGQVVNLMNQRLQTGFTENEVLQIFCDTCEAVARLHQCKTPIIHRDLKVENILLHDRGHYVLCDFGSATNKFQNPQTEGVNAVEDEIKKYTTLSYRAPEMVNLYSGKIITTKADIWALGCLLYKLCYFTLPFGESQVAICDGNFTIPDNSRYSQDMHCLIRYMLEPDPDKRPDIYQVSYFSFKLLKKECPIPNVQNSPIPAKLPEPVKASEAAAKKTQPKARLTDPIPTTETSIAPRQRPKAGQTQPNQAQGSGQPTTPTGQEKPSPHSTLPQPKTQGLAKDAWEIPRESLRLEVKLGQGCFGEVWMGTWNGTTRVAIKTLKPGTMSPEAFLQEAQVMKKLRHEKLVQLYAVVSEEPIYIVTEYMSKGSLLDFLKGETGKYLRLPQLVDMAAQIASGMAYVERMNYVHRDLRAANILVGENLVCKVADFGLARLIEDNEYTARQGAKFPIKWTAPEAALYGRFTIKSDVWSFGILLTELTTKGRVPYPGMVNREVLDQVERGYRMPCPPECPESLHDLMCQCWRKEPEERPTFEYLQAFLEDYFTSTEPQYQPGENL'
  },
  ibuprofen: {
    smiles: 'CC(C)Cc1ccc(cc1)C(C)C(=O)O',
    sequence: 'MLARALLLCAVLALSHTANPCCSHPCQNRGVCMSVGFDQYKCDCTRTGFYGENCSTPEFLTRIKLFLKPTPNTVHYILTHFKGFWNVVNNIPFLRNAIMSYVLTSRSHLIDSPPTYNADYGYKSWEAFSNLSYYTRALPPVPDDCPTPLGVKGKKQLPDSNEIVEKLLLRRKFIPD'
  }
}

function App() {
  const [smiles, setSmiles] = useState('')
  const [sequence, setSequence] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<PredictionResult | null>(null)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const response = await axios.post<PredictionResult>('/api/predict', {
        smiles,
        sequence
      })
      setResult(response.data)
    } catch (err: any) {
      setError(err.response?.data?.detail || 'An error occurred during prediction')
    } finally {
      setLoading(false)
    }
  }

  const loadExample = (example: keyof typeof EXAMPLES) => {
    setSmiles(EXAMPLES[example].smiles)
    setSequence(EXAMPLES[example].sequence)
    setError(null)
    setResult(null)
  }

  return (
    <div className="container">
      <div className="header">
        <h1>🧬 DTI Prediction Platform</h1>
        <p>Drug-Target Interaction Prediction with Uncertainty Quantification</p>
      </div>

      <div className="card">
        <form onSubmit={handleSubmit}>
          <div className="form-group">
            <label htmlFor="smiles">Drug SMILES String</label>
            <textarea
              id="smiles"
              rows={3}
              value={smiles}
              onChange={(e) => setSmiles(e.target.value)}
              placeholder="Enter SMILES string (e.g., CC(=O)Oc1ccccc1C(=O)O)"
              required
            />
          </div>

          <div className="form-group">
            <label htmlFor="sequence">Protein Sequence</label>
            <textarea
              id="sequence"
              rows={6}
              value={sequence}
              onChange={(e) => setSequence(e.target.value)}
              placeholder="Enter amino acid sequence (e.g., MKKFFDSR...)"
              required
            />
          </div>

          <button type="submit" className="button" disabled={loading}>
            {loading ? 'Predicting...' : 'Predict Binding Affinity'}
          </button>
        </form>

        <div className="example-section">
          <h3>Try Examples:</h3>
          <button
            type="button"
            className="example-button"
            onClick={() => loadExample('aspirin')}
          >
            Aspirin + COX-2
          </button>
          <button
            type="button"
            className="example-button"
            onClick={() => loadExample('ibuprofen')}
          >
            Ibuprofen + COX-1
          </button>
        </div>
      </div>

      {error && (
        <div className="card">
          <div className="error">{error}</div>
        </div>
      )}

      {loading && (
        <div className="card">
          <div className="loading">Analyzing drug-target interaction...</div>
        </div>
      )}

      {result && (
        <div className="card">
          <h2>Prediction Results</h2>
          <div className="results">
            <div className="metric">
              <span className="metric-label">Binding Affinity (pKi/pKd/pIC50)</span>
              <span className="metric-value">{result.affinity.toFixed(2)}</span>
            </div>

            <div className="metric">
              <span className="metric-label">Uncertainty Score</span>
              <span className="metric-value">{result.uncertainty.toFixed(3)}</span>
            </div>

            <div className="metric">
              <span className="metric-label">Confidence Level</span>
              <span
                className={`confidence-badge confidence-${result.confidence_level.toLowerCase()}`}
              >
                {result.confidence_level}
              </span>
            </div>
          </div>

          <div className="detailed-explanation">
            <h3>Detailed Analysis</h3>
            
            <div className="explanation-section">
              <h4>🎯 Affinity Calculation</h4>
              <div className="explanation-content">
                <p><strong>Interpretation:</strong> {result.detailed_explanation.affinity_calculation.interpretation}</p>
                <p><strong>Binding Strength:</strong> {result.detailed_explanation.affinity_calculation.binding_strength}</p>
                <p><strong>Raw Model Output:</strong> {result.detailed_explanation.affinity_calculation.raw_prediction.toFixed(4)}</p>
              </div>
            </div>

            <div className="explanation-section">
              <h4>📊 Uncertainty Breakdown</h4>
              <div className="explanation-content">
                <p><strong>Confidence:</strong> {result.detailed_explanation.uncertainty_breakdown.confidence_description}</p>
                <div className="uncertainty-details">
                  <div className="uncertainty-item">
                    <span>Epistemic (Model) Uncertainty:</span>
                    <span>{result.detailed_explanation.uncertainty_breakdown.epistemic_uncertainty.toFixed(3)}</span>
                  </div>
                  <div className="uncertainty-item">
                    <span>Aleatoric (Data) Uncertainty:</span>
                    <span>{result.detailed_explanation.uncertainty_breakdown.aleatoric_uncertainty.toFixed(3)}</span>
                  </div>
                  <div className="uncertainty-item total">
                    <span>Total Uncertainty:</span>
                    <span>{result.detailed_explanation.uncertainty_breakdown.total_uncertainty.toFixed(3)}</span>
                  </div>
                </div>
                <div className="uncertainty-explanations">
                  <p><em>Epistemic:</em> {result.detailed_explanation.uncertainty_breakdown.explanation.epistemic}</p>
                  <p><em>Aleatoric:</em> {result.detailed_explanation.uncertainty_breakdown.explanation.aleatoric}</p>
                </div>
              </div>
            </div>

            <div className="explanation-section">
              <h4>🔬 Evidential Parameters</h4>
              <div className="explanation-content">
                <div className="parameter-grid">
                  <div className="parameter-item">
                    <span>γ (Gamma):</span>
                    <span>{result.detailed_explanation.evidential_parameters.gamma.toFixed(4)}</span>
                    <small>{result.detailed_explanation.evidential_parameters.parameter_meanings.gamma}</small>
                  </div>
                  <div className="parameter-item">
                    <span>ν (Nu):</span>
                    <span>{result.detailed_explanation.evidential_parameters.nu.toFixed(4)}</span>
                    <small>{result.detailed_explanation.evidential_parameters.parameter_meanings.nu}</small>
                  </div>
                  <div className="parameter-item">
                    <span>α (Alpha):</span>
                    <span>{result.detailed_explanation.evidential_parameters.alpha.toFixed(4)}</span>
                    <small>{result.detailed_explanation.evidential_parameters.parameter_meanings.alpha}</small>
                  </div>
                  <div className="parameter-item">
                    <span>β (Beta):</span>
                    <span>{result.detailed_explanation.evidential_parameters.beta.toFixed(4)}</span>
                    <small>{result.detailed_explanation.evidential_parameters.parameter_meanings.beta}</small>
                  </div>
                </div>
              </div>
            </div>

            <div className="explanation-section">
              <h4>📈 Input Analysis</h4>
              <div className="explanation-content">
                <div className="input-analysis">
                  <div className="analysis-item">
                    <h5>Drug (SMILES)</h5>
                    <p>Length: {result.detailed_explanation.input_analysis.smiles_length} characters</p>
                    {result.detailed_explanation.input_analysis.smiles_complexity.num_atoms !== "N/A" && (
                      <div className="molecule-info">
                        <p>Atoms: {result.detailed_explanation.input_analysis.smiles_complexity.num_atoms}</p>
                        <p>Bonds: {result.detailed_explanation.input_analysis.smiles_complexity.num_bonds}</p>
                        <p>Rings: {result.detailed_explanation.input_analysis.smiles_complexity.num_rings}</p>
                        <p>Molecular Weight: {result.detailed_explanation.input_analysis.smiles_complexity.molecular_weight.toFixed(2)} Da</p>
                      </div>
                    )}
                  </div>
                  <div className="analysis-item">
                    <h5>Protein Sequence</h5>
                    <p>Length: {result.detailed_explanation.input_analysis.sequence_length} amino acids</p>
                    <p>Unique AAs: {result.detailed_explanation.input_analysis.sequence_composition.unique_amino_acids}</p>
                    <p>Most Common: {result.detailed_explanation.input_analysis.sequence_composition.most_common[0]} ({result.detailed_explanation.input_analysis.sequence_composition.most_common[1]} times)</p>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <div className="attention-visualization">
            <h3>🔍 Attention Visualization</h3>
            <div className="attention-info">
              <p><strong>Shape:</strong> {result.detailed_explanation.attention_analysis.attention_shape}</p>
              <p><strong>Range:</strong> {result.detailed_explanation.attention_analysis.min_attention.toFixed(4)} - {result.detailed_explanation.attention_analysis.max_attention.toFixed(4)}</p>
              <p><em>{result.detailed_explanation.attention_analysis.interpretation}</em></p>
            </div>
            
            <div className="attention-heatmap">
              {result.attention_weights && result.attention_weights.length > 0 ? (
                <div className="heatmap-container">
                  <div className="heatmap-grid">
                    {result.attention_weights.map((row, i) => (
                      <div key={i} className="heatmap-row">
                        {row.map((weight, j) => (
                          <div
                            key={j}
                            className="heatmap-cell"
                            style={{
                              backgroundColor: `rgba(102, 126, 234, ${weight})`,
                              opacity: 0.3 + weight * 0.7
                            }}
                            title={`Attention: ${weight.toFixed(4)}`}
                          >
                            {weight.toFixed(3)}
                          </div>
                        ))}
                      </div>
                    ))}
                  </div>
                  <div className="heatmap-legend">
                    <span>Low</span>
                    <div className="legend-gradient"></div>
                    <span>High</span>
                  </div>
                </div>
              ) : (
                <p className="no-attention">No attention weights available</p>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default App
