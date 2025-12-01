/**
 * Neumes Lexicon - A collection of neume shapes with their descriptions
 * Each entry contains the neume name, description, and characteristics
 */

export interface NeumeEntry {
  type: string;
  shape: string;
  strokeCount: number;
}

export const neumesLexicon: NeumeEntry[] = [
  {
    type: "punctum",
    shape: "dot",
    strokeCount: 1,
  },
  {
    type: "celeriter",
    shape: "c-shaped hook",
    strokeCount: 1,
  },
  {
    type: "pes",
    shape: "curved stroke, starting with small hook from bottom left to top right; check-mark / swoosh",
    strokeCount: 1,
  },
  {
    type: "virga",
    shape: "diagonal stroke from bottom left to top right",
    strokeCount: 1,
  },
  {
    type: "pes quadratus",
    shape: "diagonal stroke from bottom left to top right; square hooks at both ends",
    strokeCount: 1,
  },
  {
    type: "tenete",
    shape: "t-shaped",
    strokeCount: 1,
  },
  {
    type: "pes subbipunctis",
    shape: "curved stroke from bottom left to top right; hooks at bottom left; dot below; another dot below",
    strokeCount: 3,
  },
  {
    type: "pes quadratus subbipunctis",
    shape: "diagonal stroke from bottom left to top right; square hooks at both ends; two dots below",
    strokeCount: 3,
  },
  {
    type: "quilisma",
    shape: "three small zig-zag strokes/loops",
    strokeCount: 1,
  },
  {
    type: "stropha",
    shape: "comma-shaped vertical stroke",
    strokeCount: 1,
  },
  {
    type: "tractulus",
    shape: "horizontal line, like a dash",
    strokeCount: 1,
  },
  {
    type: "oriscus",
    shape: "s-shaped curve",
    strokeCount: 1,
  },
  {
    type: "clivis",
    shape: "curved stroke, inverted-U / arch-like. Rises from bottom left, peaks, then descends toward bottom right",
    strokeCount: 1,
  },
  {
    type: "bistropha",
    shape: "comma-shaped curve downwards; comma-shaped curve downwards, right of first",
    strokeCount: 2,
  },
];

export function getAllNeumeDescriptions(): string[] {
  return neumesLexicon.map((neume) => `${neume.type}: ${neume.shape}`);
}

/**
 * Get neumes filtered by stroke count
 */
export function getNeumesByStrokeCount(strokeCount: number): NeumeEntry[] {
  return neumesLexicon.filter((neume) => neume.strokeCount === strokeCount);
}

/**
 * Get neumes within a stroke count range (for flexibility)
 */
export function getNeumesByStrokeCountRange(
  minStrokes: number,
  maxStrokes: number
): NeumeEntry[] {
  return neumesLexicon.filter(
    (neume) =>
      neume.strokeCount >= minStrokes && neume.strokeCount <= maxStrokes
  );
}
