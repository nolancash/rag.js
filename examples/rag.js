// src/chunker.ts
var DEFAULT_CHUNK_SIZE = 512;
var DEFAULT_CHUNK_OVERLAP = 50;
var PARAGRAPH_SPLIT = /\n\n+/;
var SENTENCE_SPLIT = /(?<=[.!?])\s+/;
var WORD_SPLIT = /\s+/;
function chunkText(text, options = {}) {
  const {
    chunkSize = DEFAULT_CHUNK_SIZE,
    chunkOverlap = DEFAULT_CHUNK_OVERLAP,
    separator
  } = options;
  if (chunkSize <= 0) {
    throw new Error("chunkSize must be positive");
  }
  if (chunkOverlap < 0) {
    throw new Error("chunkOverlap cannot be negative");
  }
  if (chunkOverlap >= chunkSize) {
    throw new Error("chunkOverlap must be less than chunkSize");
  }
  let segments;
  if (separator) {
    segments = text.split(separator).filter((s) => s.trim());
    if (segments.every((s) => s.length <= chunkSize)) {
      return segments.map((s) => s.trim());
    }
  } else {
    if (text.length <= chunkSize) {
      return text.trim() ? [text.trim()] : [];
    }
    segments = splitByBoundary(text, chunkSize);
  }
  return mergeSegments(segments, chunkSize, chunkOverlap);
}
function splitByBoundary(text, maxSize) {
  let segments = text.split(PARAGRAPH_SPLIT).filter((s) => s.trim());
  const needsSentenceSplit = segments.some((s) => s.length > maxSize);
  if (needsSentenceSplit) {
    segments = segments.flatMap((segment) => {
      if (segment.length <= maxSize) return [segment];
      const sentences = segment.split(SENTENCE_SPLIT).filter((s) => s.trim());
      return sentences.flatMap((sentence) => {
        if (sentence.length <= maxSize) return [sentence];
        return splitByWords(sentence, maxSize);
      });
    });
  }
  return segments;
}
function splitByWords(text, maxSize) {
  const words = text.split(WORD_SPLIT);
  const chunks = [];
  let current = "";
  for (const word of words) {
    if (word.length > maxSize) {
      if (current) {
        chunks.push(current.trim());
        current = "";
      }
      for (let i = 0; i < word.length; i += maxSize) {
        chunks.push(word.slice(i, i + maxSize));
      }
      continue;
    }
    const candidate = current ? `${current} ${word}` : word;
    if (candidate.length <= maxSize) {
      current = candidate;
    } else {
      if (current) chunks.push(current.trim());
      current = word;
    }
  }
  if (current.trim()) {
    chunks.push(current.trim());
  }
  return chunks;
}
function mergeSegments(segments, chunkSize, overlap) {
  if (segments.length === 0) return [];
  const chunks = [];
  let current = "";
  let overlapBuffer = "";
  for (const segment of segments) {
    const separator = current ? "\n\n" : "";
    const candidate = current + separator + segment;
    if (candidate.length <= chunkSize) {
      current = candidate;
    } else {
      if (current) {
        chunks.push(current.trim());
        if (overlap > 0) {
          overlapBuffer = getOverlapText(current, overlap);
        }
      }
      if (overlapBuffer && segment.length + overlapBuffer.length + 1 <= chunkSize) {
        current = overlapBuffer + " " + segment;
      } else {
        current = segment;
      }
      while (current.length > chunkSize) {
        const splitPoint = findSplitPoint(current, chunkSize);
        chunks.push(current.slice(0, splitPoint).trim());
        overlapBuffer = getOverlapText(current.slice(0, splitPoint), overlap);
        const remaining = current.slice(splitPoint).trim();
        current = overlapBuffer ? overlapBuffer + " " + remaining : remaining;
      }
    }
  }
  if (current.trim()) {
    chunks.push(current.trim());
  }
  return chunks;
}
function getOverlapText(text, overlap) {
  if (overlap <= 0 || text.length <= overlap) return "";
  const slice = text.slice(-overlap);
  const wordBoundary = slice.search(/\s/);
  if (wordBoundary > 0 && wordBoundary < slice.length - 1) {
    return slice.slice(wordBoundary + 1);
  }
  return slice;
}
function findSplitPoint(text, target) {
  if (text.length <= target) return text.length;
  const searchStart = Math.max(0, target - 50);
  const searchEnd = Math.min(text.length, target + 50);
  const searchRegion = text.slice(searchStart, searchEnd);
  const sentenceMatch = searchRegion.match(/[.!?]\s+/);
  if (sentenceMatch && sentenceMatch.index !== void 0) {
    return searchStart + sentenceMatch.index + sentenceMatch[0].length;
  }
  const lastSpace = text.lastIndexOf(" ", target);
  if (lastSpace > target * 0.5) {
    return lastSpace + 1;
  }
  return target;
}
function createChunks(text, baseId, metadata, options) {
  const texts = chunkText(text, options);
  return texts.map((content, index) => ({
    id: `${baseId}-${index}`,
    content,
    metadata: {
      ...metadata,
      chunkIndex: index,
      totalChunks: texts.length
    }
  }));
}
function createChunksFromDocuments(documents, options) {
  return documents.flatMap(
    (doc) => createChunks(doc.content, doc.id, doc.metadata, options)
  );
}

// src/embeddings.ts
var DEFAULT_MODEL = "Xenova/all-MiniLM-L6-v2";
var MODEL_DIMENSIONS = {
  "Xenova/all-MiniLM-L6-v2": 384,
  "Xenova/all-MiniLM-L12-v2": 384,
  "Xenova/bge-small-en-v1.5": 384,
  "Xenova/bge-base-en-v1.5": 768,
  "Xenova/e5-small-v2": 384,
  "Xenova/gte-small": 384
};
var TransformersEmbedding = class {
  model;
  dimensions;
  pipeline = null;
  loadPromise = null;
  constructor(model = DEFAULT_MODEL) {
    this.model = model;
    this.dimensions = MODEL_DIMENSIONS[model] ?? 384;
  }
  async ensureLoaded() {
    if (this.pipeline) return;
    if (!this.loadPromise) {
      this.loadPromise = this.loadPipeline();
    }
    await this.loadPromise;
  }
  async loadPipeline() {
    try {
      const { pipeline } = await import("@xenova/transformers");
      this.pipeline = await pipeline("feature-extraction", this.model, {
        quantized: true
      });
    } catch (error) {
      throw new Error(
        `Failed to load transformers.js model "${this.model}". Make sure @xenova/transformers is installed: ${error}`
      );
    }
  }
  async embed(texts) {
    await this.ensureLoaded();
    const results = [];
    for (const text of texts) {
      const output = await this.pipeline(text, {
        pooling: "mean",
        normalize: true
      });
      results.push(Array.from(output.data));
    }
    return results;
  }
};
var CustomEmbedding = class {
  model;
  dimensions;
  embedFn;
  constructor(embedFn, options) {
    this.embedFn = embedFn;
    this.model = options.model ?? "custom";
    this.dimensions = options.dimensions;
  }
  async embed(texts) {
    return this.embedFn(texts);
  }
};
var MockEmbedding = class {
  model = "mock";
  dimensions;
  vectors = /* @__PURE__ */ new Map();
  constructor(dimensions = 384) {
    this.dimensions = dimensions;
  }
  async embed(texts) {
    return texts.map((text) => {
      if (this.vectors.has(text)) {
        return this.vectors.get(text);
      }
      const vector = this.generateVector(text);
      this.vectors.set(text, vector);
      return vector;
    });
  }
  generateVector(text) {
    const vector = [];
    let seed = this.hashCode(text);
    for (let i = 0; i < this.dimensions; i++) {
      seed = seed * 1103515245 + 12345 & 2147483647;
      vector.push(seed / 2147483647 * 2 - 1);
    }
    const magnitude = Math.sqrt(vector.reduce((sum, v) => sum + v * v, 0));
    return vector.map((v) => v / magnitude);
  }
  hashCode(str) {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
      hash = (hash << 5) - hash + str.charCodeAt(i) | 0;
    }
    return Math.abs(hash);
  }
};
function createEmbeddingProvider(modelOrProvider) {
  if (!modelOrProvider) {
    return new TransformersEmbedding();
  }
  if (typeof modelOrProvider === "string") {
    return new TransformersEmbedding(modelOrProvider);
  }
  return modelOrProvider;
}

// src/vector-store.ts
var INDEX_VERSION = 1;
function cosineSimilarity(a, b) {
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
var VectorStore = class _VectorStore {
  chunks = [];
  model;
  dimensions;
  constructor(model = "unknown", dimensions = 384) {
    this.model = model;
    this.dimensions = dimensions;
  }
  /** Number of chunks in the store */
  get size() {
    return this.chunks.length;
  }
  /** Model used for embeddings */
  get embeddingModel() {
    return this.model;
  }
  /** Embedding dimensions */
  get embeddingDimensions() {
    return this.dimensions;
  }
  /**
   * Add a single embedded chunk
   */
  add(chunk) {
    this.validateDimensions(chunk.embedding);
    this.chunks.push(chunk);
  }
  /**
   * Add multiple embedded chunks
   */
  addAll(chunks) {
    for (const chunk of chunks) {
      this.add(chunk);
    }
  }
  /**
   * Remove a chunk by ID
   */
  remove(id) {
    const index = this.chunks.findIndex((c) => c.id === id);
    if (index === -1) return false;
    this.chunks.splice(index, 1);
    return true;
  }
  /**
   * Get a chunk by ID
   */
  get(id) {
    return this.chunks.find((c) => c.id === id);
  }
  /**
   * Check if a chunk exists
   */
  has(id) {
    return this.chunks.some((c) => c.id === id);
  }
  /**
   * Clear all chunks
   */
  clear() {
    this.chunks = [];
  }
  /**
   * Search for similar chunks
   */
  search(queryEmbedding, options = {}) {
    const { topK = 5, threshold = 0, filter } = options;
    this.validateDimensions(queryEmbedding);
    const results = [];
    for (const chunk of this.chunks) {
      if (filter && !filter(chunk.metadata)) {
        continue;
      }
      const score = cosineSimilarity(queryEmbedding, chunk.embedding);
      if (score >= threshold) {
        results.push({
          chunk: {
            id: chunk.id,
            content: chunk.content,
            metadata: chunk.metadata
          },
          score
        });
      }
    }
    results.sort((a, b) => b.score - a.score);
    return results.slice(0, topK);
  }
  /**
   * Serialize the index for storage
   */
  serialize() {
    return {
      version: INDEX_VERSION,
      model: this.model,
      dimensions: this.dimensions,
      chunks: this.chunks
    };
  }
  /**
   * Export to JSON string
   */
  toJSON() {
    return JSON.stringify(this.serialize());
  }
  /**
   * Export to compact binary format
   */
  toBinary() {
    const serialized = this.serialize();
    const json = JSON.stringify({
      version: serialized.version,
      model: serialized.model,
      dimensions: serialized.dimensions,
      chunkMeta: serialized.chunks.map((c) => ({
        id: c.id,
        content: c.content,
        metadata: c.metadata
      }))
    });
    const encoder = new TextEncoder();
    const jsonBytes = encoder.encode(json);
    const headerSize = 16;
    const unalignedOffset = headerSize + jsonBytes.length;
    const padding = (4 - unalignedOffset % 4) % 4;
    const vectorOffset = unalignedOffset + padding;
    const vectorsSize = this.chunks.length * this.dimensions * 4;
    const totalSize = vectorOffset + vectorsSize;
    const buffer = new ArrayBuffer(totalSize);
    const view = new DataView(buffer);
    const uint8 = new Uint8Array(buffer);
    view.setUint32(0, 1380009811, false);
    view.setUint32(4, INDEX_VERSION, false);
    view.setUint32(8, jsonBytes.length, false);
    view.setUint32(12, this.chunks.length, false);
    uint8.set(jsonBytes, headerSize);
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
  static fromSerialized(data) {
    if (data.version !== INDEX_VERSION) {
      throw new Error(`Unsupported index version: ${data.version}`);
    }
    const store = new _VectorStore(data.model, data.dimensions);
    store.addAll(data.chunks);
    return store;
  }
  /**
   * Load from JSON string
   */
  static fromJSON(json) {
    const data = JSON.parse(json);
    return _VectorStore.fromSerialized(data);
  }
  /**
   * Load from binary format
   */
  static fromBinary(buffer) {
    const view = new DataView(buffer);
    const uint8 = new Uint8Array(buffer);
    const magic = view.getUint32(0, false);
    if (magic !== 1380009811) {
      throw new Error("Invalid binary index format");
    }
    const version = view.getUint32(4, false);
    if (version !== INDEX_VERSION) {
      throw new Error(`Unsupported index version: ${version}`);
    }
    const jsonLength = view.getUint32(8, false);
    const chunkCount = view.getUint32(12, false);
    const headerSize = 16;
    const decoder = new TextDecoder();
    const json = decoder.decode(uint8.slice(headerSize, headerSize + jsonLength));
    const meta = JSON.parse(json);
    const unalignedOffset = headerSize + jsonLength;
    const padding = (4 - unalignedOffset % 4) % 4;
    const vectorOffset = unalignedOffset + padding;
    const vectors = new Float32Array(buffer, vectorOffset, chunkCount * meta.dimensions);
    const chunks = [];
    for (let i = 0; i < chunkCount; i++) {
      const chunkMeta = meta.chunkMeta[i];
      const embedding = Array.from(vectors.slice(i * meta.dimensions, (i + 1) * meta.dimensions));
      chunks.push({
        id: chunkMeta.id,
        content: chunkMeta.content,
        metadata: chunkMeta.metadata,
        embedding
      });
    }
    const store = new _VectorStore(meta.model, meta.dimensions);
    store.addAll(chunks);
    return store;
  }
  validateDimensions(embedding) {
    if (embedding.length !== this.dimensions) {
      throw new Error(
        `Embedding dimension mismatch: expected ${this.dimensions}, got ${embedding.length}`
      );
    }
  }
  /**
   * Iterate over all chunks
   */
  [Symbol.iterator]() {
    return this.chunks[Symbol.iterator]();
  }
};

// src/rag.ts
var RAG = class _RAG {
  embedding;
  store;
  chunkOptions;
  searchDefaults;
  initialized = false;
  constructor(options = {}) {
    this.embedding = createEmbeddingProvider(options.model);
    this.store = new VectorStore(this.embedding.model, this.embedding.dimensions);
    this.chunkOptions = options.chunkOptions ?? {};
    this.searchDefaults = options.searchOptions ?? {};
  }
  /** The embedding model being used */
  get model() {
    return this.embedding.model;
  }
  /** Number of chunks in the index */
  get size() {
    return this.store.size;
  }
  /** Embedding dimensions */
  get dimensions() {
    return this.store.embeddingDimensions;
  }
  /**
   * Add a document to the index
   * The document will be chunked and embedded
   */
  async addDocument(id, content, metadata) {
    const chunks = createChunks(content, id, metadata, this.chunkOptions);
    await this.addChunks(chunks);
    return chunks;
  }
  /**
   * Add multiple documents at once
   */
  async addDocuments(documents) {
    const chunks = createChunksFromDocuments(documents, this.chunkOptions);
    await this.addChunks(chunks);
    return chunks;
  }
  /**
   * Add pre-created chunks (useful for custom chunking strategies)
   */
  async addChunks(chunks) {
    if (chunks.length === 0) return;
    const texts = chunks.map((c) => c.content);
    const embeddings = await this.embedding.embed(texts);
    const embeddedChunks = chunks.map((chunk, i) => ({
      ...chunk,
      embedding: embeddings[i]
    }));
    this.store.addAll(embeddedChunks);
    this.initialized = true;
  }
  /**
   * Add a single pre-embedded chunk (for manual embedding)
   */
  addEmbeddedChunk(chunk) {
    this.store.add(chunk);
    this.initialized = true;
  }
  /**
   * Remove a chunk by ID
   */
  remove(id) {
    return this.store.remove(id);
  }
  /**
   * Remove all chunks from a document
   */
  removeDocument(documentId) {
    let removed = 0;
    const toRemove = [];
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
  clear() {
    this.store.clear();
    this.initialized = false;
  }
  /**
   * Search for relevant chunks
   */
  async search(query, options) {
    const mergedOptions = { ...this.searchDefaults, ...options };
    const [queryEmbedding] = await this.embedding.embed([query]);
    return this.store.search(queryEmbedding, mergedOptions);
  }
  /**
   * Search with a pre-computed query embedding
   */
  searchByEmbedding(queryEmbedding, options) {
    const mergedOptions = { ...this.searchDefaults, ...options };
    return this.store.search(queryEmbedding, mergedOptions);
  }
  /**
   * Get context for a query (convenience method)
   * Returns concatenated text from top results
   */
  async getContext(query, options) {
    const { separator = "\n\n---\n\n", ...searchOptions } = options ?? {};
    const results = await this.search(query, searchOptions);
    return results.map((r) => r.chunk.content).join(separator);
  }
  /**
   * Serialize the index for storage/precompilation
   */
  serialize() {
    return this.store.serialize();
  }
  /**
   * Export to JSON string
   */
  toJSON() {
    return this.store.toJSON();
  }
  /**
   * Export to compact binary format
   */
  toBinary() {
    return this.store.toBinary();
  }
  /**
   * Create a RAG instance from a serialized index
   * This is the "precompiled" mode - no embedding computation needed
   */
  static fromIndex(data, options = {}) {
    const rag = new _RAG({
      ...options,
      model: data.model
    });
    rag.store = VectorStore.fromSerialized(data);
    rag.initialized = true;
    return rag;
  }
  /**
   * Create from JSON string
   */
  static fromJSON(json, options = {}) {
    const data = JSON.parse(json);
    return _RAG.fromIndex(data, options);
  }
  /**
   * Create from binary format
   */
  static fromBinary(buffer, options = {}) {
    const store = VectorStore.fromBinary(buffer);
    const data = store.serialize();
    return _RAG.fromIndex(data, options);
  }
  /**
   * Create a RAG instance with a custom embedding provider
   */
  static withEmbedding(provider, options = {}) {
    const rag = new _RAG(options);
    rag.embedding = provider;
    rag.store = new VectorStore(provider.model, provider.dimensions);
    return rag;
  }
};
function createRAG(options) {
  return new RAG(options);
}
export {
  CustomEmbedding,
  DEFAULT_MODEL,
  MockEmbedding,
  RAG,
  TransformersEmbedding,
  VectorStore,
  chunkText,
  cosineSimilarity,
  createChunks,
  createChunksFromDocuments,
  createEmbeddingProvider,
  createRAG
};
