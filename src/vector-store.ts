/**
 * Vector storage and similarity search for rag.js
 */

import type {
  EmbeddedChunk,
  SearchOptions,
  SearchResult,
  SerializedIndex,
} from './types.js';

const INDEX_VERSION = 1;

/**
 * Compute cosine similarity between two vectors
 */
export function cosineSimilarity(a: number[], b: number[]): number {
  if (a.length !== b.length) {
    throw new Error(`Vector dimension mismatch: ${a.length} vs ${b.length}`);
  }

  let dotProduct = 0;
  let normA = 0;
  let normB = 0;

  for (let i = 0; i < a.length; i++) {
    dotProduct += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }

  const magnitude = Math.sqrt(normA) * Math.sqrt(normB);
  if (magnitude === 0) return 0;

  return dotProduct / magnitude;
}

/**
 * Vector store with flat index (brute force search)
 * Suitable for small to medium datasets (< 10k vectors)
 */
export class VectorStore {
  private chunks: EmbeddedChunk[] = [];
  private model: string;
  private dimensions: number;

  constructor(model: string = 'unknown', dimensions: number = 384) {
    this.model = model;
    this.dimensions = dimensions;
  }

  /** Number of chunks in the store */
  get size(): number {
    return this.chunks.length;
  }

  /** Model used for embeddings */
  get embeddingModel(): string {
    return this.model;
  }

  /** Embedding dimensions */
  get embeddingDimensions(): number {
    return this.dimensions;
  }

  /**
   * Add a single embedded chunk
   */
  add(chunk: EmbeddedChunk): void {
    this.validateDimensions(chunk.embedding);
    this.chunks.push(chunk);
  }

  /**
   * Add multiple embedded chunks
   */
  addAll(chunks: EmbeddedChunk[]): void {
    for (const chunk of chunks) {
      this.add(chunk);
    }
  }

  /**
   * Remove a chunk by ID
   */
  remove(id: string): boolean {
    const index = this.chunks.findIndex(c => c.id === id);
    if (index === -1) return false;
    this.chunks.splice(index, 1);
    return true;
  }

  /**
   * Get a chunk by ID
   */
  get(id: string): EmbeddedChunk | undefined {
    return this.chunks.find(c => c.id === id);
  }

  /**
   * Check if a chunk exists
   */
  has(id: string): boolean {
    return this.chunks.some(c => c.id === id);
  }

  /**
   * Clear all chunks
   */
  clear(): void {
    this.chunks = [];
  }

  /**
   * Search for similar chunks
   */
  search(
    queryEmbedding: number[],
    options: SearchOptions = {}
  ): SearchResult[] {
    const { topK = 5, threshold = 0, filter } = options;

    this.validateDimensions(queryEmbedding);

    // Calculate similarities
    const results: SearchResult[] = [];

    for (const chunk of this.chunks) {
      // Apply metadata filter
      if (filter && !filter(chunk.metadata)) {
        continue;
      }

      const score = cosineSimilarity(queryEmbedding, chunk.embedding);

      if (score >= threshold) {
        results.push({
          chunk: {
            id: chunk.id,
            content: chunk.content,
            metadata: chunk.metadata,
          },
          score,
        });
      }
    }

    // Sort by score descending and return top K
    results.sort((a, b) => b.score - a.score);
    return results.slice(0, topK);
  }

  /**
   * Serialize the index for storage
   */
  serialize(): SerializedIndex {
    return {
      version: INDEX_VERSION,
      model: this.model,
      dimensions: this.dimensions,
      chunks: this.chunks,
    };
  }

  /**
   * Export to JSON string
   */
  toJSON(): string {
    return JSON.stringify(this.serialize());
  }

  /**
   * Export to compact binary format
   */
  toBinary(): ArrayBuffer {
    const serialized = this.serialize();
    const json = JSON.stringify({
      version: serialized.version,
      model: serialized.model,
      dimensions: serialized.dimensions,
      chunkMeta: serialized.chunks.map(c => ({
        id: c.id,
        content: c.content,
        metadata: c.metadata,
      })),
    });

    const encoder = new TextEncoder();
    const jsonBytes = encoder.encode(json);

    // Header: magic (4) + version (4) + json length (4) + chunk count (4)
    const headerSize = 16;
    
    // Pad JSON to ensure vectors start at 4-byte aligned offset
    const unalignedOffset = headerSize + jsonBytes.length;
    const padding = (4 - (unalignedOffset % 4)) % 4;
    const vectorOffset = unalignedOffset + padding;
    
    const vectorsSize = this.chunks.length * this.dimensions * 4;
    const totalSize = vectorOffset + vectorsSize;

    const buffer = new ArrayBuffer(totalSize);
    const view = new DataView(buffer);
    const uint8 = new Uint8Array(buffer);

    // Write header
    view.setUint32(0, 0x52414753, false); // "RAGS" magic
    view.setUint32(4, INDEX_VERSION, false);
    view.setUint32(8, jsonBytes.length, false);
    view.setUint32(12, this.chunks.length, false);

    // Write JSON
    uint8.set(jsonBytes, headerSize);

    // Write vectors (at aligned offset)
    const vectors = new Float32Array(buffer, vectorOffset, this.chunks.length * this.dimensions);
    let vi = 0;
    for (const chunk of this.chunks) {
      for (const val of chunk.embedding) {
        vectors[vi++] = val;
      }
    }

    return buffer;
  }

  /**
   * Load from serialized index
   */
  static fromSerialized(data: SerializedIndex): VectorStore {
    if (data.version !== INDEX_VERSION) {
      throw new Error(`Unsupported index version: ${data.version}`);
    }

    const store = new VectorStore(data.model, data.dimensions);
    store.addAll(data.chunks);
    return store;
  }

  /**
   * Load from JSON string
   */
  static fromJSON(json: string): VectorStore {
    const data = JSON.parse(json) as SerializedIndex;
    return VectorStore.fromSerialized(data);
  }

  /**
   * Load from binary format
   */
  static fromBinary(buffer: ArrayBuffer): VectorStore {
    const view = new DataView(buffer);
    const uint8 = new Uint8Array(buffer);

    // Read header
    const magic = view.getUint32(0, false);
    if (magic !== 0x52414753) {
      throw new Error('Invalid binary index format');
    }

    const version = view.getUint32(4, false);
    if (version !== INDEX_VERSION) {
      throw new Error(`Unsupported index version: ${version}`);
    }

    const jsonLength = view.getUint32(8, false);
    const chunkCount = view.getUint32(12, false);

    // Read JSON
    const headerSize = 16;
    const decoder = new TextDecoder();
    const json = decoder.decode(uint8.slice(headerSize, headerSize + jsonLength));
    const meta = JSON.parse(json);

    // Calculate vector offset with padding for alignment
    const unalignedOffset = headerSize + jsonLength;
    const padding = (4 - (unalignedOffset % 4)) % 4;
    const vectorOffset = unalignedOffset + padding;
    
    const vectors = new Float32Array(buffer, vectorOffset, chunkCount * meta.dimensions);

    // Reconstruct chunks
    const chunks: EmbeddedChunk[] = [];
    for (let i = 0; i < chunkCount; i++) {
      const chunkMeta = meta.chunkMeta[i];
      const embedding = Array.from(vectors.slice(i * meta.dimensions, (i + 1) * meta.dimensions));
      chunks.push({
        id: chunkMeta.id,
        content: chunkMeta.content,
        metadata: chunkMeta.metadata,
        embedding,
      });
    }

    const store = new VectorStore(meta.model, meta.dimensions);
    store.addAll(chunks);
    return store;
  }

  private validateDimensions(embedding: number[]): void {
    if (embedding.length !== this.dimensions) {
      throw new Error(
        `Embedding dimension mismatch: expected ${this.dimensions}, got ${embedding.length}`
      );
    }
  }

  /**
   * Iterate over all chunks
   */
  [Symbol.iterator](): Iterator<EmbeddedChunk> {
    return this.chunks[Symbol.iterator]();
  }
}
