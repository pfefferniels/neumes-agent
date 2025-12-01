import "./NeumeResults.css";

export interface NeumeMatch {
  id: string;
  name: string;
  description: string;
  similarity: number;
}

interface NeumeResultsProps {
  description: string;
  matches: NeumeMatch[];
  needsDisambiguation: boolean;
  onSelectNeume: (neume: NeumeMatch) => void;
  isLoading: boolean;
}

export function NeumeResults({
  description,
  matches,
  needsDisambiguation,
  onSelectNeume,
  isLoading,
}: NeumeResultsProps) {
  if (isLoading) {
    return (
      <div className="neume-results loading">
        <div className="spinner"></div>
        <p>Analyzing neume shape...</p>
      </div>
    );
  }

  if (matches.length === 0) {
    return (
      <div className="neume-results">
        <p className="no-matches">No matching neumes found</p>
      </div>
    );
  }

  return (
    <div className="neume-results">
      <div className="analysis-description">
        <h3>AI Analysis</h3>
        <p>{description}</p>
      </div>

      {needsDisambiguation ? (
        <div className="disambiguation">
          <h3>Multiple matches found - please select the best match:</h3>
          <div className="matches-list">
            {matches.map((match) => (
              <button
                key={match.id}
                className="match-option"
                onClick={() => onSelectNeume(match)}
              >
                <div className="match-header">
                  <span className="match-name">{match.name}</span>
                  <span className="match-similarity">
                    {(match.similarity * 100).toFixed(1)}% match
                  </span>
                </div>
                <p className="match-description">{match.description}</p>
              </button>
            ))}
          </div>
        </div>
      ) : (
        <div className="single-result">
          <h3>Identified Neume</h3>
          <div className="result-card">
            <div className="result-header">
              <span className="result-name">{matches[0]?.name}</span>
              <span className="result-similarity">
                {((matches[0]?.similarity ?? 0) * 100).toFixed(1)}% match
              </span>
            </div>
            <p className="result-description">{matches[0]?.description}</p>
          </div>
        </div>
      )}
    </div>
  );
}
