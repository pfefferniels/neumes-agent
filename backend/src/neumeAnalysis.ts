import OpenAI from "openai";
import {
  neumesLexicon,
  getNeumesByStrokeCountRange,
} from "./neumesLexicon.js";
import type { NeumeEntry } from "./neumesLexicon.js";

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

export interface NeumeMatchResult {
  neume: NeumeEntry;
  similarity: number;
}

export interface AnalysisResult {
  description: string;
  matches: NeumeMatchResult[];
  needsDisambiguation: boolean;
}

/**
 * Count the number of pen strokes in an image using AI vision
 */
export async function countPenStrokes(imageBase64: string): Promise<number> {
  const response = await openai.responses.create({
    model: "gpt-5.1",
    input: [
      {
        role: "user",
        content: [
          {
            type: "input_text",
            text: `Count the number of distinct pen strokes in this medieval neume notation image. 
A pen stroke is a continuous mark made without lifting the pen.
Dots count as separate strokes.
Connected curves or lines that form a single continuous shape count as one stroke.
Respond with ONLY a single number (e.g., "1", "2", "3").`,
          },
          {
            type: "input_image",
            image_url: `data:image/png;base64,${imageBase64}`,
            detail: "low",
          },
        ],
      },
    ],
  });

  const text = response.output_text.trim();
  const parsed = parseInt(text, 10);
  // Default to 1 if parsing fails
  return isNaN(parsed) ? 1 : parsed;
}

/**
 * Analyze an image region to describe the neume shape using OpenAI's vision API
 */
export async function analyzeNeumeImage(imageBase64: string): Promise<string> {
  const response = await openai.responses.create({
    model: "gpt-5.1",
    input: [
      {
        role: "user",
        content: [
          {
            type: "input_text",
            text: `
For each pen stroke from top to bottom and left to right, describe in very few keywords:
- its shape (e.g. dot, diagonal stroke, vertical line, horizontal line, s-shaped, c-shaped, t-shaped)
- if helpful for identifying, the direction (e.g. from bottom left to top right; hook at bottom left)
- relative position to other components (only if more than one stroke).
Separate multiple stroke descriptions with semicolons.`.trim(),
          },
          {
            type: "input_image",
            image_url: `data:image/png;base64,${imageBase64}`,
            detail: "low",
          },
        ],
      },
    ],
  });

  return response.output_text;
}

/**
 * Match the neume image against filtered candidates using AI reasoning
 */
export async function matchNeumeWithCandidates(
  imageBase64: string,
  candidates: NeumeEntry[]
): Promise<{ matchedType: string; reasoning: string }> {
  // Build candidate list for the prompt
  const candidateList = candidates
    .map((c, i) => `${i + 1}. ${c.type}: ${c.shape}`)
    .join("\n");

  const response = await openai.responses.create({
    model: "gpt-5.1",
    input: [
      {
        role: "user",
        content: [
          {
            type: "input_text",
            text: `You are an expert in medieval neume notation. Analyze this neume image and determine which of the following neume types best matches it.

Possible neume types (filtered by stroke count):
${candidateList}

Instructions:
1. Carefully observe the shape, direction, and any distinctive features of the neume in the image.
2. Compare it against each candidate's description.
3. Select the best matching neume type.

Respond in this exact format:
MATCH: [neume type name]
REASONING: [brief explanation of why this matches and others don't]`,
          },
          {
            type: "input_image",
            image_url: `data:image/png;base64,${imageBase64}`,
            detail: "low",
          },
        ],
      },
    ],
  });

  const text = response.output_text;

  // Parse the response
  const matchLine = text.match(/MATCH:\s*(.+)/i);
  const reasoningLine = text.match(/REASONING:\s*(.+)/is);

  const matchedType = matchLine?.[1]?.trim() ?? candidates[0]?.type ?? "";
  const reasoning = reasoningLine?.[1]?.trim() ?? text;

  return { matchedType, reasoning };
}

/**
 * Generate embeddings for a text using OpenAI's embedding API
 */
async function getEmbedding(text: string): Promise<number[]> {
  const response = await openai.embeddings.create({
    model: "text-embedding-3-large",
    input: text,
  });

  const embedding = response.data[0]?.embedding;
  if (!embedding) {
    throw new Error("Failed to generate embedding");
  }
  return embedding;
}

/**
 * Calculate cosine similarity between two vectors
 */
function cosineSimilarity(vecA: number[], vecB: number[]): number {
  let dotProduct = 0;
  let normA = 0;
  let normB = 0;

  for (let i = 0; i < vecA.length; i++) {
    const a = vecA[i] ?? 0;
    const b = vecB[i] ?? 0;
    dotProduct += a * b;
    normA += a * a;
    normB += b * b;
  }

  return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
}

/**
 * Find matching neumes using embeddings
 */
export async function findMatchingNeumes(
  description: string,
  threshold: number = 0.6,
  maxMatches: number = 5
): Promise<NeumeMatchResult[]> {
  // Get embedding for the description
  const descriptionEmbedding = await getEmbedding(description);

  // Calculate similarity with each neume in the lexicon
  const matches: NeumeMatchResult[] = [];

  for (const neume of neumesLexicon) {
    // Create a comprehensive text for the neume
    const neumeText = neume.shape;
    const neumeEmbedding = await getEmbedding(neumeText);

    const similarity = cosineSimilarity(descriptionEmbedding, neumeEmbedding);

    if (similarity >= threshold) {
      matches.push({ neume, similarity });
    }
  }

  // Sort by similarity (highest first)
  matches.sort((a, b) => b.similarity - a.similarity);

  // Return top matches
  return matches.slice(0, maxMatches);
}

/**
 * Complete analysis: analyze image and find matching neumes
 * Uses pen stroke counting and AI reasoning for higher quality matching
 */
export async function analyzeAndMatchNeume(
  imageBase64: string
): Promise<AnalysisResult> {
  // Step 1: Count pen strokes in the image
  const strokeCount = await countPenStrokes(imageBase64);

  // Step 2: Filter candidates by stroke count (with tolerance of ±1)
  let candidates = getNeumesByStrokeCountRange(
    Math.max(1, strokeCount - 1),
    strokeCount + 1
  );

  // If no candidates found, fall back to all neumes
  if (candidates.length === 0) {
    candidates = neumesLexicon;
  }

  // Step 3: Use AI reasoning to match against filtered candidates
  const { matchedType, reasoning } = await matchNeumeWithCandidates(
    imageBase64,
    candidates
  );

  // Step 4: Build the matches array
  // Put the matched neume first with high similarity
  const matches: NeumeMatchResult[] = [];
  const matchedNeume = candidates.find(
    (c) => c.type.toLowerCase() === matchedType.toLowerCase()
  );

  if (matchedNeume) {
    matches.push({ neume: matchedNeume, similarity: 1.0 });
  }

  // Add other candidates as alternatives with lower similarity
  for (const candidate of candidates) {
    if (candidate.type.toLowerCase() !== matchedType.toLowerCase()) {
      matches.push({ neume: candidate, similarity: 0.5 });
    }
  }

  // Determine if disambiguation is needed
  // If there are multiple candidates with same stroke count, user may want to verify
  const needsDisambiguation = candidates.length > 1;

  // Build description that includes the reasoning
  const description = `Stroke count: ${strokeCount}. ${reasoning}`;

  return {
    description,
    matches,
    needsDisambiguation,
  };
}
