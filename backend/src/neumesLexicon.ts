/**
 * Neumes Lexicon - A collection of neume shapes with their descriptions
 * Each entry contains the neume name, description, and characteristics
 */

export interface NeumeEntry {
  type: string;
  shape: string;
  traditions?: { [tradition: string]: string };
  modifier?: boolean
}

export const neumesLexicon: NeumeEntry[] = [
  {
    type: "celeriter",
    shape: "c-shaped hook",
    modifier: true,
  },
  {
    type: "tenete",
    shape: "t-shaped",
    modifier: true,
  },
  {
    type: "episema",
    shape: "thicker emphasized head of a virga or pes, sometimes as pronounced hook – or a short horizontal stroke on the peak of a clivis",
    modifier: true
  },
  {
    type: "punctum",
    shape: "dot",
  },
  {
    type: "pes",
    shape: "diagonal stroke bottom left to top right with curved hook at bottom left",
    traditions: {
      'csg-382': 'very pronounced curved hook on bottom left, overall shape almost like a lowercase d'
    }
  },
  {
    type: "virga",
    shape: "diagonal stroke from bottom left to top right",
  },
  {
    type: "virga",
    shape: "Like virga but two times"
  },
  {
    type: "pes quadratus",
    shape: "diagonal stroke from bottom left to top right; square-ish hook at the bottom left",

  },
  {
    type: "pes subbipunctis",
    shape: "stroke from bottom left to top right, curved hook only at bottom left; two dots below",
  },
  {
    type: "pes quadratus subbipunctis",
    shape: "diagonal stroke from bottom left to top right with somewhat square hook at bottom left; two dots below",
    traditions: {
      'csg-382': 'excluded'
    }
  },
  {
    type: "quilisma",
    shape: "three short hook-like waves followed by one longer stroke going upwards",
  },
  {
    type: "stropha",
    shape: "comma-shaped vertical stroke",
  },
  {
    type: "bistropha",
    shape: "two comma-shaped strokes downwards",
  },
  {
    type: "tractulus",
    shape: "short horizontal line",
  },
  {
    type: "oriscus",
    shape: "upright S-shape, can have somewhat squarish elements",
  },
  {
    type: "clivis",
    shape: "unseparatedly going diagonally up and left, then diagonally left down, n-shaped, often slightly slanted to the right, so the line going down seems shorter. No dots",
  },
  {
    type: "cephalicus",
    shape: "p-like shape; vertical line with loop at top right",
  },
  {
    type: "climacus",
    shape: "Diagonal stroke bottom left to top right; two dots below",
  },
  {
    type: "torculus",
    shape: "Smotthly curved S-shape slightly angled to the right",
  },
  {
    type: "pressus",
    shape: "diagonal stroke going bottom left to top right, hook on the top right going downwards, followed by a distinct dot",
  }
];

export function getAllNeumeDescriptions(): string[] {
  return neumesLexicon.map((neume) => `${neume.type}: ${neume.shape}`);
}

