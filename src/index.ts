/**
 * rag.js - A RAG library for transformers.js
 * 
 * @example
 * ```typescript
 * import { RAG } from 'rag.js';
 * 
 * // Dynamic mode
 * const rag = new RAG();
 * await rag.addDocument('doc1', 'Your document content...');
 * const results = await rag.search('query');
 * 
 * // Precompiled mode
 * const index = await fetch('/index.json').then(r => r.json());
 * const rag = RAG.fromIndex(index);
 * const results = await rag.search('query');
 * ```
 * 
 * @packageDocumentation
 */

// Main RAG class
export { RAG, createRAG } from './rag.js';

// Chunking utilities
export {
  chunkText,
  createChunks,
  createChunksFromDocuments,
} from './chunker.js';

// Embedding providers
export {
  TransformersEmbedding,
  CustomEmbedding,
  MockEmbedding,
  createEmbeddingProvider,
  DEFAULT_MODEL,
} from './embeddings.js';

// Vector store
export {
  VectorStore,
  cosineSimilarity,
} from './vector-store.js';

// Types
export type {
  Chunk,
  ChunkOptions,
  EmbeddedChunk,
  EmbeddingProvider,
  RagOptions,
  SearchOptions,
  SearchResult,
  SerializedIndex,
} from './types.js';
