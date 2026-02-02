/**
 * Core types for rag.js
 */

/** A document chunk with content and metadata */
export interface Chunk {
  /** Unique identifier for this chunk */
  id: string;
  /** The text content */
  content: string;
  /** Optional metadata (source file, page number, etc.) */
  metadata?: Record<string, unknown>;
}

/** A chunk with its computed embedding vector */
export interface EmbeddedChunk extends Chunk {
  /** The embedding vector */
  embedding: number[];
}

/** Search result with similarity score */
export interface SearchResult {
  /** The matching chunk */
  chunk: Chunk;
  /** Similarity score (0-1, higher is more similar) */
  score: number;
}

/** Options for chunking text */
export interface ChunkOptions {
  /** Maximum characters per chunk (default: 512) */
  chunkSize?: number;
  /** Overlap between chunks in characters (default: 50) */
  chunkOverlap?: number;
  /** Separator to split on (default: paragraph/sentence boundaries) */
  separator?: string | RegExp;
}

/** Options for vector search */
export interface SearchOptions {
  /** Maximum number of results (default: 5) */
  topK?: number;
  /** Minimum similarity threshold (default: 0) */
  threshold?: number;
  /** Filter by metadata */
  filter?: (metadata: Record<string, unknown> | undefined) => boolean;
}

/** Serialized index format for precompiled indices */
export interface SerializedIndex {
  /** Version for compatibility checking */
  version: number;
  /** Embedding model used */
  model: string;
  /** Embedding dimensions */
  dimensions: number;
  /** All embedded chunks */
  chunks: EmbeddedChunk[];
}

/** Options for the RAG instance */
export interface RagOptions {
  /** Embedding model name (default: 'Xenova/all-MiniLM-L6-v2') */
  model?: string;
  /** Chunking options */
  chunkOptions?: ChunkOptions;
  /** Search options defaults */
  searchOptions?: SearchOptions;
}

/** Embedding provider interface */
export interface EmbeddingProvider {
  /** Model name */
  readonly model: string;
  /** Embedding dimensions */
  readonly dimensions: number;
  /** Generate embeddings for texts */
  embed(texts: string[]): Promise<number[][]>;
}
