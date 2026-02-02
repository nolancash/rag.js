/**
 * Main RAG (Retrieval Augmented Generation) class
 */

import type {
  Chunk,
  ChunkOptions,
  EmbeddedChunk,
  EmbeddingProvider,
  RagOptions,
  SearchOptions,
  SearchResult,
  SerializedIndex,
} from './types.js';
import { createChunks, createChunksFromDocuments } from './chunker.js';
import { createEmbeddingProvider } from './embeddings.js';
import { VectorStore } from './vector-store.js';

/**
 * RAG - Retrieval Augmented Generation
 * 
 * A simple interface for building RAG applications with transformers.js
 * 
 * @example
 * ```typescript
 * // Dynamic mode - compute embeddings at runtime
 * const rag = new RAG();
 * await rag.addDocument('doc1', 'Your document content here...');
 * const results = await rag.search('query text');
 * 
 * // Precompiled mode - load pre-built index
 * const index = await fetch('/index.json').then(r => r.json());
 * const rag = RAG.fromIndex(index);
 * const results = await rag.search('query text');
 * ```
 */
export class RAG {
  private embedding: EmbeddingProvider;
  private store: VectorStore;
  private chunkOptions: ChunkOptions;
  private searchDefaults: SearchOptions;
  private initialized = false;

  constructor(options: RagOptions = {}) {
    this.embedding = createEmbeddingProvider(options.model);
    this.store = new VectorStore(this.embedding.model, this.embedding.dimensions);
    this.chunkOptions = options.chunkOptions ?? {};
    this.searchDefaults = options.searchOptions ?? {};
  }

  /** The embedding model being used */
  get model(): string {
    return this.embedding.model;
  }

  /** Number of chunks in the index */
  get size(): number {
    return this.store.size;
  }

  /** Embedding dimensions */
  get dimensions(): number {
    return this.store.embeddingDimensions;
  }

  /**
   * Add a document to the index
   * The document will be chunked and embedded
   */
  async addDocument(
    id: string,
    content: string,
    metadata?: Record<string, unknown>
  ): Promise<Chunk[]> {
    const chunks = createChunks(content, id, metadata, this.chunkOptions);
    await this.addChunks(chunks);
    return chunks;
  }

  /**
   * Add multiple documents at once
   */
  async addDocuments(
    documents: Array<{ id: string; content: string; metadata?: Record<string, unknown> }>
  ): Promise<Chunk[]> {
    const chunks = createChunksFromDocuments(documents, this.chunkOptions);
    await this.addChunks(chunks);
    return chunks;
  }

  /**
   * Add pre-created chunks (useful for custom chunking strategies)
   */
  async addChunks(chunks: Chunk[]): Promise<void> {
    if (chunks.length === 0) return;

    // Embed all chunks
    const texts = chunks.map(c => c.content);
    const embeddings = await this.embedding.embed(texts);

    // Store embedded chunks
    const embeddedChunks: EmbeddedChunk[] = chunks.map((chunk, i) => ({
      ...chunk,
      embedding: embeddings[i],
    }));

    this.store.addAll(embeddedChunks);
    this.initialized = true;
  }

  /**
   * Add a single pre-embedded chunk (for manual embedding)
   */
  addEmbeddedChunk(chunk: EmbeddedChunk): void {
    this.store.add(chunk);
    this.initialized = true;
  }

  /**
   * Remove a chunk by ID
   */
  remove(id: string): boolean {
    return this.store.remove(id);
  }

  /**
   * Remove all chunks from a document
   */
  removeDocument(documentId: string): number {
    let removed = 0;
    const toRemove: string[] = [];

    for (const chunk of this.store) {
      if (chunk.id.startsWith(`${documentId}-`)) {
        toRemove.push(chunk.id);
      }
    }

    for (const id of toRemove) {
      if (this.store.remove(id)) removed++;
    }

    return removed;
  }

  /**
   * Clear all chunks from the index
   */
  clear(): void {
    this.store.clear();
    this.initialized = false;
  }

  /**
   * Search for relevant chunks
   */
  async search(
    query: string,
    options?: SearchOptions
  ): Promise<SearchResult[]> {
    const mergedOptions = { ...this.searchDefaults, ...options };

    // Embed the query
    const [queryEmbedding] = await this.embedding.embed([query]);

    // Search the store
    return this.store.search(queryEmbedding, mergedOptions);
  }

  /**
   * Search with a pre-computed query embedding
   */
  searchByEmbedding(
    queryEmbedding: number[],
    options?: SearchOptions
  ): SearchResult[] {
    const mergedOptions = { ...this.searchDefaults, ...options };
    return this.store.search(queryEmbedding, mergedOptions);
  }

  /**
   * Get context for a query (convenience method)
   * Returns concatenated text from top results
   */
  async getContext(
    query: string,
    options?: SearchOptions & { separator?: string }
  ): Promise<string> {
    const { separator = '\n\n---\n\n', ...searchOptions } = options ?? {};
    const results = await this.search(query, searchOptions);
    return results.map(r => r.chunk.content).join(separator);
  }

  /**
   * Serialize the index for storage/precompilation
   */
  serialize(): SerializedIndex {
    return this.store.serialize();
  }

  /**
   * Export to JSON string
   */
  toJSON(): string {
    return this.store.toJSON();
  }

  /**
   * Export to compact binary format
   */
  toBinary(): ArrayBuffer {
    return this.store.toBinary();
  }

  /**
   * Create a RAG instance from a serialized index
   * This is the "precompiled" mode - no embedding computation needed
   */
  static fromIndex(
    data: SerializedIndex,
    options: Omit<RagOptions, 'model'> = {}
  ): RAG {
    const rag = new RAG({
      ...options,
      model: data.model,
    });

    // Replace the store with the loaded one
    rag.store = VectorStore.fromSerialized(data);
    rag.initialized = true;

    return rag;
  }

  /**
   * Create from JSON string
   */
  static fromJSON(
    json: string,
    options: Omit<RagOptions, 'model'> = {}
  ): RAG {
    const data = JSON.parse(json) as SerializedIndex;
    return RAG.fromIndex(data, options);
  }

  /**
   * Create from binary format
   */
  static fromBinary(
    buffer: ArrayBuffer,
    options: Omit<RagOptions, 'model'> = {}
  ): RAG {
    const store = VectorStore.fromBinary(buffer);
    const data = store.serialize();
    return RAG.fromIndex(data, options);
  }

  /**
   * Create a RAG instance with a custom embedding provider
   */
  static withEmbedding(
    provider: EmbeddingProvider,
    options: Omit<RagOptions, 'model'> = {}
  ): RAG {
    const rag = new RAG(options);
    rag.embedding = provider;
    rag.store = new VectorStore(provider.model, provider.dimensions);
    return rag;
  }
}

/**
 * Create a new RAG instance (convenience function)
 */
export function createRAG(options?: RagOptions): RAG {
  return new RAG(options);
}
