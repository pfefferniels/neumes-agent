/**
 * Neumes Lexicon - A collection of neume shapes with their descriptions
 * Each entry contains the neume name, description, and characteristics
 */

export interface NeumeEntry {
  id: string;
  name: string;
  description: string;
  characteristics: string[];
}

export const neumesLexicon: NeumeEntry[] = [
  {
    id: "punctum",
    name: "Punctum",
    description: "A single dot or small square representing one note. The most basic neume shape, appearing as a small filled square or rhombus.",
    characteristics: ["single note", "square shape", "basic", "simple", "dot-like"]
  },
  {
    id: "virga",
    name: "Virga",
    description: "A single note with a vertical stem extending upward. Represents a higher pitch note, appearing as a square with an ascending line.",
    characteristics: ["single note", "vertical stem", "ascending line", "higher pitch"]
  },
  {
    id: "pes",
    name: "Pes (Podatus)",
    description: "Two notes ascending, starting low and moving to a higher pitch. The shape shows a curved or angled connection moving upward from left to right.",
    characteristics: ["two notes", "ascending", "curved upward", "low to high"]
  },
  {
    id: "clivis",
    name: "Clivis",
    description: "Two notes descending, starting high and moving to a lower pitch. The shape shows a curved or angled connection moving downward from left to right.",
    characteristics: ["two notes", "descending", "curved downward", "high to low"]
  },
  {
    id: "scandicus",
    name: "Scandicus",
    description: "Three or more notes in ascending order. A sequence of notes moving progressively higher, often appearing as connected dots or squares climbing upward.",
    characteristics: ["three or more notes", "ascending sequence", "climbing", "multiple ascending"]
  },
  {
    id: "climacus",
    name: "Climacus",
    description: "Three or more notes in descending order. Begins with a higher note followed by a descending sequence, often appearing as dots or squares stepping down.",
    characteristics: ["three or more notes", "descending sequence", "stepping down", "multiple descending"]
  },
  {
    id: "torculus",
    name: "Torculus",
    description: "Three notes in a low-high-low pattern. The middle note is the highest, creating a hill or arch shape. Resembles an inverted valley.",
    characteristics: ["three notes", "low-high-low", "arch shape", "peak in middle"]
  },
  {
    id: "porrectus",
    name: "Porrectus",
    description: "Three notes in a high-low-high pattern. The middle note is the lowest, creating a valley or U shape. Resembles an inverted arch.",
    characteristics: ["three notes", "high-low-high", "valley shape", "dip in middle"]
  },
  {
    id: "quilisma",
    name: "Quilisma",
    description: "A wavy or zigzag shaped neume, often appearing between two ascending notes. Has a distinctive trembling or jagged appearance.",
    characteristics: ["wavy", "zigzag", "trembling", "jagged", "ornamental"]
  },
  {
    id: "oriscus",
    name: "Oriscus",
    description: "A small curved or hook-like neume that modifies the note it's attached to. Often appears at the end of a neume group with a distinctive curve.",
    characteristics: ["curved", "hook-like", "ornamental", "modifier", "small curve"]
  },
  {
    id: "pressus",
    name: "Pressus",
    description: "A compound neume indicating a note to be emphasized or lengthened. Often appears as a thickened or doubled shape.",
    characteristics: ["emphasis", "lengthened", "thickened", "doubled", "stressed"]
  },
  {
    id: "strophicus",
    name: "Strophicus (Apostropha)",
    description: "A repeated note neume, often appearing as multiple dots or marks in succession on the same pitch level.",
    characteristics: ["repeated notes", "same pitch", "multiple dots", "horizontal repetition"]
  },
  {
    id: "salicus",
    name: "Salicus",
    description: "An ascending neume with three or more notes, featuring a special oriscus in the middle that indicates an expressive leap.",
    characteristics: ["ascending", "three notes", "expressive leap", "oriscus included"]
  },
  {
    id: "trigon",
    name: "Trigon",
    description: "Three notes arranged in a triangular pattern, often descending then ascending or vice versa.",
    characteristics: ["three notes", "triangular", "mixed direction"]
  },
  {
    id: "bistropha",
    name: "Bistropha",
    description: "Two repeated notes at the same pitch, appearing as two dots or marks side by side.",
    characteristics: ["two notes", "same pitch", "repeated", "side by side"]
  },
  {
    id: "tristropha",
    name: "Tristropha",
    description: "Three repeated notes at the same pitch, appearing as three dots or marks in succession.",
    characteristics: ["three notes", "same pitch", "repeated", "triple repetition"]
  },
  {
    id: "epiphonus",
    name: "Epiphonus",
    description: "A liquescent neume descending from a higher note, with a small tail or diminished second note.",
    characteristics: ["two notes", "descending", "liquescent", "diminished ending", "tail"]
  },
  {
    id: "cephalicus",
    name: "Cephalicus",
    description: "A liquescent neume ascending to a higher note, with a rounded or curved head at the top.",
    characteristics: ["two notes", "ascending", "liquescent", "rounded head", "curved top"]
  }
];

export function getAllNeumeDescriptions(): string[] {
  return neumesLexicon.map(neume => 
    `${neume.name}: ${neume.description}`
  );
}
