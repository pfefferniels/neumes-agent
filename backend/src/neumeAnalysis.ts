import OpenAI from "openai";
import { neumesLexicon } from "./neumesLexicon.js";
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
 */
export async function analyzeAndMatchNeume(
  imageBase64: string
): Promise<AnalysisResult> {
  // Step 1: Analyze the image to get a description
  const description = await analyzeNeumeImage(imageBase64);

  // Step 2: Find matching neumes using embeddings
  const matches = await findMatchingNeumes(description);

  // Determine if disambiguation is needed
  // If there are multiple matches with similar scores (within 0.05), ask user
  let needsDisambiguation = false;
  if (matches.length > 1 && matches[0]) {
    const topScore = matches[0].similarity;
    const closeMatches = matches.filter(
      (m) => topScore - m.similarity < 0.05
    );
    needsDisambiguation = closeMatches.length > 1;
  }

  return {
    description,
    matches,
    needsDisambiguation,
  };
}
