import "./NeumeResults.css";

export interface NeumeMatch {
  type: string;
  modifiers?: string[];
  uncertainty?: string;
  alternatives?: string[];
}

interface NeumeResultsProps {
  matches: NeumeMatch[];
  onSelectNeume: (neume: NeumeMatch) => void;
  isLoading: boolean;
}

export function NeumeResults({
  matches,
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
      <div className="disambiguation">
        <div className="matches-list">
          {matches.map((match) => (
            <button
              key={match.type}
              className="match-option"
              onClick={() => onSelectNeume(match)}
            >
              <div className="match-header">
                <span className="match-name">{match.type} {match.modifiers && match.modifiers.join(',')}</span>
                <span className="match-similarity">
                  {match.uncertainty}
                </span>

                <div>
                  {match.alternatives && match.alternatives.map((d) => (
                    <div key={d} className="disambiguation-item">
                      <button className="match-option">
                        {d}
                      </button>
                    </div>
                  ))}
                </div>
              </div>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
