/**
 * Embedding providers for rag.js
 */

import type { EmbeddingProvider } from './types.js';

/** Default embedding model */
export const DEFAULT_MODEL = 'Xenova/all-MiniLM-L6-v2';

/** Common embedding dimensions by model */
const MODEL_DIMENSIONS: Record<string, number> = {
  'Xenova/all-MiniLM-L6-v2': 384,
  'Xenova/all-MiniLM-L12-v2': 384,
  'Xenova/bge-small-en-v1.5': 384,
  'Xenova/bge-base-en-v1.5': 768,
  'Xenova/e5-small-v2': 384,
  'Xenova/gte-small': 384,
};

/**
 * Embedding provider using transformers.js
 */
export class TransformersEmbedding implements EmbeddingProvider {
  readonly model: string;
  readonly dimensions: number;
  
  private pipeline: unknown = null;
  private loadPromise: Promise<void> | null = null;

  constructor(model: string = DEFAULT_MODEL) {
    this.model = model;
    this.dimensions = MODEL_DIMENSIONS[model] ?? 384;
  }

  private async ensureLoaded(): Promise<void> {
    if (this.pipeline) return;
    
    if (!this.loadPromise) {
      this.loadPromise = this.loadPipeline();
    }
    
    await this.loadPromise;
  }

  private async loadPipeline(): Promise<void> {
    try {
      // Dynamic import for transformers.js (optional peer dependency)
      const { pipeline } = await import('@xenova/transformers');
      this.pipeline = await pipeline('feature-extraction', this.model, {
        quantized: true,
      });
    } catch (error) {
      throw new Error(
        `Failed to load transformers.js model "${this.model}". ` +
        `Make sure @xenova/transformers is installed: ${error}`
      );
    }
  }

  async embed(texts: string[]): Promise<number[][]> {
    await this.ensureLoaded();
    
    const results: number[][] = [];
    
    for (const text of texts) {
      // @ts-expect-error - pipeline type is dynamic
      const output = await this.pipeline(text, {
        pooling: 'mean',
        normalize: true,
      });
      
      results.push(Array.from(output.data as Float32Array));
    }
    
    return results;
  }
}

/**
 * Custom embedding provider for user-provided embeddings
 */
export class CustomEmbedding implements EmbeddingProvider {
  readonly model: string;
  readonly dimensions: number;
  
  private embedFn: (texts: string[]) => Promise<number[][]>;

  constructor(
    embedFn: (texts: string[]) => Promise<number[][]>,
    options: { model?: string; dimensions: number }
  ) {
    this.embedFn = embedFn;
    this.model = options.model ?? 'custom';
    this.dimensions = options.dimensions;
  }

  async embed(texts: string[]): Promise<number[][]> {
    return this.embedFn(texts);
  }
}

/**
 * Mock embedding provider for testing (generates random vectors)
 */
export class MockEmbedding implements EmbeddingProvider {
  readonly model = 'mock';
  readonly dimensions: number;
  
  private vectors: Map<string, number[]> = new Map();

  constructor(dimensions: number = 384) {
    this.dimensions = dimensions;
  }

  async embed(texts: string[]): Promise<number[][]> {
    return texts.map(text => {
      // Return consistent vectors for the same text
      if (this.vectors.has(text)) {
        return this.vectors.get(text)!;
      }
      
      // Generate deterministic pseudo-random vector based on text
      const vector = this.generateVector(text);
      this.vectors.set(text, vector);
      return vector;
    });
  }

  private generateVector(text: string): number[] {
    const vector: number[] = [];
    let seed = this.hashCode(text);
    
    for (let i = 0; i < this.dimensions; i++) {
      seed = (seed * 1103515245 + 12345) & 0x7fffffff;
      vector.push((seed / 0x7fffffff) * 2 - 1);
    }
    
    // Normalize
    const magnitude = Math.sqrt(vector.reduce((sum, v) => sum + v * v, 0));
    return vector.map(v => v / magnitude);
  }

  private hashCode(str: string): number {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
      hash = ((hash << 5) - hash + str.charCodeAt(i)) | 0;
    }
    return Math.abs(hash);
  }
}

/**
 * Create an embedding provider
 */
export function createEmbeddingProvider(
  modelOrProvider?: string | EmbeddingProvider
): EmbeddingProvider {
  if (!modelOrProvider) {
    return new TransformersEmbedding();
  }
  
  if (typeof modelOrProvider === 'string') {
    return new TransformersEmbedding(modelOrProvider);
  }
  
  return modelOrProvider;
}
