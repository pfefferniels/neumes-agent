import { useState } from "react";
import "./App.css";
import { ImageUploader } from "./components/ImageUploader";
import { RegionSelector } from "./components/RegionSelector";
import { NeumeResults } from "./components/NeumeResults";
import type { NeumeMatch } from "./components/NeumeResults";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:3001";

interface AnalysisResponse {
  success: boolean;
  description: string;
  matches: NeumeMatch[];
  needsDisambiguation: boolean;
}

function App() {
  const [imageSrc, setImageSrc] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [analysisResult, setAnalysisResult] =
    useState<AnalysisResponse | null>(null);
  const [selectedNeume, setSelectedNeume] = useState<NeumeMatch | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleImageLoaded = (src: string) => {
    setImageSrc(src);
    setAnalysisResult(null);
    setSelectedNeume(null);
    setError(null);
  };

  const handleRegionSelected = async (regionImageData: string) => {
    setIsLoading(true);
    setError(null);
    setSelectedNeume(null);

    try {
      const response = await fetch(`${API_URL}/api/analyze`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ image: regionImageData }),
      });

      if (!response.ok) {
        throw new Error(`Analysis failed: ${response.statusText}`);
      }

      const result: AnalysisResponse = await response.json();
      setAnalysisResult(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : "An error occurred");
    } finally {
      setIsLoading(false);
    }
  };

  const handleSelectNeume = (neume: NeumeMatch) => {
    setSelectedNeume(neume);
  };

  const handleReset = () => {
    setImageSrc(null);
    setAnalysisResult(null);
    setSelectedNeume(null);
    setError(null);
  };

  return (
    <div className="app">
      <header className="app-header">
        <h1>Neumes Agent</h1>
        <p>AI-powered medieval music notation identifier</p>
      </header>

      <main className="app-main">
        {!imageSrc ? (
          <ImageUploader onImageLoaded={handleImageLoaded} />
        ) : (
          <div className="workspace">
            <div className="workspace-header">
              <button onClick={handleReset} className="reset-button">
                ← Upload New Image
              </button>
            </div>

            <div className="workspace-content">
              <div className="image-panel">
                <RegionSelector
                  imageSrc={imageSrc}
                  onRegionSelected={handleRegionSelected}
                />
              </div>

              <div className="results-panel">
                {error && (
                  <div className="error-message">
                    <p>{error}</p>
                  </div>
                )}

                {selectedNeume ? (
                  <div className="selection-confirmed">
                    <h3>Selected Neume</h3>
                    <div className="confirmed-card">
                      <span className="confirmed-name">{selectedNeume.name}</span>
                      <p>{selectedNeume.description}</p>
                    </div>
                  </div>
                ) : (
                  (isLoading || analysisResult) && (
                    <NeumeResults
                      description={analysisResult?.description || ""}
                      matches={analysisResult?.matches || []}
                      needsDisambiguation={
                        analysisResult?.needsDisambiguation || false
                      }
                      onSelectNeume={handleSelectNeume}
                      isLoading={isLoading}
                    />
                  )
                )}

                {!isLoading && !analysisResult && !error && (
                  <div className="instructions-panel">
                    <h3>How to use</h3>
                    <ol>
                      <li>Click and drag on the image to select a region containing a neume</li>
                      <li>Click "Analyze Selected Region" to identify the neume</li>
                      <li>If multiple matches are found, select the best one</li>
                    </ol>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
