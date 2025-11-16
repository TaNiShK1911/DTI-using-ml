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
  strong: {
    name: 'Strong Binding',
    description: 'Imatinib (Gleevec) + BCR-ABL Kinase',
    smiles: 'CN1CCN(CC1)Cc2ccc(cc2)C(=O)Nc3ccc(c(c3)Nc4nccc(n4)c5cccnc5)C',
    sequence: 'MGQQPGKVLGDQRRPSLPALHFIKGAGKKESSRHGGPHCNVFVEHEALQRPVASDFEPQGLSEAARWNSKENLLAGPSENDPNLFVALYDFVASGDNTLSITKGEKLRVLGYNHNGEWCEAQTKNGQGWVPSNYITPVNSLEKHSWYHGPVSRNAAEYLLSSGINGSFLVRESESSPGQRSISLRYEGRVYHYRINTASDGKLYVSSESRFNTLAELVHHHSTVADGLITTLHYPAPKRNKPTVYGVSPNYDKWEMERTDITMKHKLGGGQYGEVYEGVWKKYSLTVAVKTLKEDTMEVEEFLKEAAVMKEIKHPNLVQLLGVCTREPPFYIITEFMTYGNLLDYLRECNRQEVNAVVLLYMATQISSAMEYLEKKNFIHRDLAARNCLVGENHLVKVADFGLSRLMTGDTYTAHAGAKFPIKWTAPESLAYNKFSIKSDVWAFGVLLWEIATYGMSPYPGIDLSQVYELLEKDYRMERPEGCPEKVYELMRACWQWNPSDRPSFAEIHQAFETMFQESSISDEVEKELGKQGVRGAVSTLLQAPELPTKTRTSRRAAEHRDTTDVPEMPHSKGQGESDPLDHEPAVSPLLPRKERGPPEGGLNEDERLLPKDKKTNLFSALIKKKKKTAPTPPKRSSSFREMDGQPERRGAGEEEGRDISNGALAFTPLDTADPAKSPKPSNGAGVPNGALRESGGSGFRSPHLWKKSSTLTSSRLATGEEEGGGSSSKRFLRSCSASCVPHGAKDTEWRSVTLPRDLQSTGRQFDSSTFGGHKSEKPALPRKRAGENRSDQVTRGTVTPPPRLVKKNEEAADEVFKDIMESSPGSSPPNLTPKPLRRQVTVAPASGLPHKEEAGKGSALGTPAAAEPVTPTSKAGSGAPGGTSKGPAEESRVRRHKHSSESPGRDKGKLSRLKPAPPPPPAASAGKAGGKPSQSPSQEAAGEAVLGAKTKATSLVDAVNSDAAKPSQPGEGLKKPVLPATPKPQSAKPSGTPISPAPVPSTLPSASSALAGDQPSSTAFIPLISTRVSLRKTRQPPERIASGAITKGVVLDSTEALCLAISRNSEQMASHSAVLEAGKNLYTFCVSYVDSIQQMRNKFAFREAINKLENNLRELQICPATAGSGPAATQDFSKLLSSVKEISDIVQR'
  },
  weak: {
    name: 'Weak Binding',
    description: 'Caffeine + Random Protein',
    smiles: 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',
    sequence: 'MTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGETCLLDILDTAGQEEYSAMRDQYMRTGEGFLCVFAINNTKSFEDIHHYREQIKRVKDSEDVPMVLVGNKCDLPARTVETRQAQDLARSYGIPFIETSAKTRQGVDDAFYTLVREIRKHKEKMSKDGKKKKKKSKTKCVIM'
  },
  moderate: {
    name: 'Moderate Binding',
    description: 'Aspirin + COX-2',
    smiles: 'CC(=O)Oc1ccccc1C(=O)O',
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
      if (err.code === 'ERR_NETWORK' || err.message.includes('Network Error')) {
        setError('Cannot connect to backend server. Please ensure the backend is running on http://localhost:8000')
      } else {
        setError(err.response?.data?.detail || 'An error occurred during prediction')
      }
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
            onClick={() => loadExample('strong')}
            title={EXAMPLES.strong.description}
          >
            {EXAMPLES.strong.name}
          </button>
          <button
            type="button"
            className="example-button"
            onClick={() => loadExample('moderate')}
            title={EXAMPLES.moderate.description}
          >
            {EXAMPLES.moderate.name}
          </button>
          <button
            type="button"
            className="example-button"
            onClick={() => loadExample('weak')}
            title={EXAMPLES.weak.description}
          >
            {EXAMPLES.weak.name}
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


        </div>
      )}
    </div>
  )
}

export default App
