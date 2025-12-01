/**
 * Neumes Lexicon - A collection of neume shapes with their descriptions
 * Each entry contains the neume name, description, and characteristics
 */

export interface NeumeEntry {
  type: string;
  shape: string
}

export const neumesLexicon: NeumeEntry[] = [
  {
    type: "punctum",
    shape: "dot",
  },
  {
    type: "celeriter",
    shape: "c-shaped hook"
  },
  {
    type: "pes",
    shape: "curved stroke, starting with small hook from bottom left to top right; check-mark / swoosh"
  },
  {
    type: "virga",
    shape: "diagonal stroke from bottom left to top right"
  },
  {
    type: "pes quadratus",
    shape: "diagonal stroke from bottom left to top right; square hooks at both ends"
  },
  {
    type: "tenete",
    shape: "t-shaped"
  },
  {
    type: "pes subbipunctis",
    shape: "curved stroke from bottom left to top right; hooks at bottom left; dot below; another dot below"
  },
  {
    type: "pes quadratus subbipunctis",
    shape: "diagonal stroke from bottom left to top right; square hooks at both ends; two dots below"
  },
  {
    type: "quilisma",
    shape: "three small zig-zag strokes/loops"
  },
  {
    type: "stropha",
    shape: "comma-shaped vertical stroke"
  },
  {
    type: "tractulus",
    shape: "horizontal line, like a dash"
  },
  {
    type: "oriscus",
    shape: "s-shaped curve"
  },
  {
    type: "clivis",
    shape: "curved stroke, inverted-U / arch-like. Rises from bottom left, peaks, then descends toward bottom right",
  },
  {
    type: "bistropha",
    shape: "comma-shaped curve downwards; comma-shaped curve downwards, right of first",
  },

];

export function getAllNeumeDescriptions(): string[] {
  return neumesLexicon.map(neume =>
    `${neume.type}: ${neume.shape}`
  );
}
