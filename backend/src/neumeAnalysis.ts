import OpenAI from "openai";
import {
  neumesLexicon,
} from "./neumesLexicon.js";
import type { NeumeEntry } from "./neumesLexicon.js";

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

export interface NeumeMatchResult {
  type: string;
  alternatives?: string[];
}

export interface AnalysisResult {
  description: string;
  matches: NeumeMatchResult[];
  needsDisambiguation: boolean;
}

/**
 * Count the number of pen strokes in an image using AI vision
 */
export async function analyze(imageBase64: string): Promise<string> {
  const response = await openai.responses.create({
    model: "gpt-5.1",
    input: [
      {
        role: "user",
        content: [
          {
            type: "input_text",
            text: `
Precisely describe the image in terms of distinct pen strokes/dots.
Gestures usually go from left to right or from top to bottom.
Each distinct stroke/dot becomes an <nc> element. Possible attributes are:
- @tilt: direction of stroke, use combination of "n", "e", "s", "w" (e.g. "ne" for northeast)
- @rellen: length of the pen stroke relative to the previous stroke within the gesture, can be "l" or "s" (longer/shorter). Do not use for first <nc>
- @s-shape: direction of an s-shaped pen stroke; "w" for the standard letter S, "e" for its mirror image, "s" for the letter S turned 90-degrees anti-clockwise, and "n" for its mirror image. 
- @curve: Indicates direction of curvature. Use only for *significantly* curved strokes. Can be "a" (anticlockwise) or "c" (clockwise)
- @con: Use only when two strokes occur, but they belong to the same pen "gesture"; "g" (gapped; not connected), "l" (looped) or "e" (extended)

- for wavy horizontal stroke, use <quilisma> as child of <nc>. Document the number of @waves.
- for gesture going arch-like up and down, use <oriscus> as child of <nc>.

If you find significative letters, such as "c" or "t", use the element <signifLet> and
indicate its relative position. Example: <signifLet place="above-right">c</signifLet>
`.trim()
          },
          {
            type: "input_image",
            image_url: `data:image/png;base64,${imageBase64}`,
            detail: "high",
          },
        ],
      },
    ],
  });

  const text = response.output_text.trim();
  return text;
}

/**
 * Count the number of pen strokes in an image using AI vision
 */
export async function countPenStrokes(imageBase64: string): Promise<[number, number]> {
  console.log('imageBase64:', imageBase64);
  const response = await openai.responses.create({
    model: "gpt-5.1",
    input: [
      {
        role: "user",
        content: [
          {
            type: "input_text",
            text: `
Count the number of distinct pen strokes in the image.
A pen stroke is a mark made without lifting the pen.
Strokes might be visually connected or overlapping, but
still count as separate if the pen would likely have been
lifted. Dots count as separate strokes.

Respond with a pair of possible strokes in strict JSON format
e.g. [1,2] for one or possibly two strokes or [2,2] for
exactly two strokes.`.trim()
          },
          {
            type: "input_image",
            image_url: `data:image/png;base64,${imageBase64}`,
            detail: "high",
          },
        ],
      },
    ],
  });

  const text = response.output_text.trim();
  return JSON.parse(text);
}

/**
 * Match the neume image against filtered candidates using AI reasoning
 */
export async function matchNeumeWithCandidates(
  imageBase64: string,
  candidates: NeumeEntry[]
): Promise<NeumeMatchResult[]> {
  // Build candidate list for the prompt
  const candidateList = candidates
    .filter(c => !c.modifier)
    .map((c) => `- "${c.type}": ${c.shape}`)
    .join("\n");
  
  const modifiers = candidates
    .filter(c => c.modifier === true)
    .map((c) => `- "${c.type}": ${c.shape}`)
    .join("\n");

  const response = await openai.responses.create({
    model: "gpt-5.1",
    reasoning: { effort: "medium" },
    input: [
      {
        role: "user",
        content: [
          {
            type: "input_text",
            text: `
Neume types:
${candidateList}

Modifiers: 
${modifiers}

Instructions:
1. Precisely analyse the image in terms of feather ink strokes or dots, their shape, direction and relative position.
2. Select the neume type from the list above most closely resembling the image. 
   In case of doubt, give the user alternative options to choose from.

Respond with a JSON array, e.g.:
  [
    { "type": "virga", modifiers: ["episema"], "alternatives": ["Thickened end is episema", "Thickened end is no episema", "There is a hook on the bottom left, so it is actually a pes", "..."] }
  ]
`,
          },
          {
            type: "input_image",
            image_url: `data:image/png;base64,${imageBase64}`,
            detail: "high",
          },
        ],
      },
    ],
  });

  const text = response.output_text;

  // Parse the response
  let matches: NeumeMatchResult[] = [];
  try {
    matches = JSON.parse(text);
  } catch (e) {
    console.error("Failed to parse AI response:", e);
  }

  return matches;
}


/**
 * Complete analysis: analyze image and find matching neumes
 * Uses pen stroke counting and AI reasoning for higher quality matching
 */
export async function analyzeAndMatchNeume(
  imageBase64: string
): Promise<NeumeMatchResult[]> {
  // Step 3: Use AI reasoning to match against filtered candidates
  return await matchNeumeWithCandidates(
    imageBase64,
    neumesLexicon
  );
}
