import { describe, it, expect, beforeEach } from 'vitest';
import { VectorStore, cosineSimilarity } from '../src/vector-store.js';
import type { EmbeddedChunk } from '../src/types.js';

describe('cosineSimilarity', () => {
  it('should return 1 for identical vectors', () => {
    const v = [0.5, 0.5, 0.5, 0.5];
    expect(cosineSimilarity(v, v)).toBeCloseTo(1, 5);
  });

  it('should return -1 for opposite vectors', () => {
    const v1 = [1, 0, 0];
    const v2 = [-1, 0, 0];
    expect(cosineSimilarity(v1, v2)).toBeCloseTo(-1, 5);
  });

  it('should return 0 for orthogonal vectors', () => {
    const v1 = [1, 0, 0];
    const v2 = [0, 1, 0];
    expect(cosineSimilarity(v1, v2)).toBeCloseTo(0, 5);
  });

  it('should handle normalized vectors', () => {
    const v1 = [0.6, 0.8, 0];
    const v2 = [0.8, 0.6, 0];
    const sim = cosineSimilarity(v1, v2);
    expect(sim).toBeGreaterThan(0);
    expect(sim).toBeLessThan(1);
  });

  it('should return 0 for zero vectors', () => {
    const v1 = [0, 0, 0];
    const v2 = [1, 0, 0];
    expect(cosineSimilarity(v1, v2)).toBe(0);
  });

  it('should throw on dimension mismatch', () => {
    const v1 = [1, 2, 3];
    const v2 = [1, 2];
    expect(() => cosineSimilarity(v1, v2)).toThrow('dimension mismatch');
  });
});

describe('VectorStore', () => {
  let store: VectorStore;

  const createChunk = (id: string, embedding: number[]): EmbeddedChunk => ({
    id,
    content: `Content for ${id}`,
    embedding,
    metadata: { source: 'test' },
  });

  beforeEach(() => {
    store = new VectorStore('test-model', 3);
  });

  describe('basic operations', () => {
    it('should start empty', () => {
      expect(store.size).toBe(0);
    });

    it('should add a chunk', () => {
      store.add(createChunk('c1', [1, 0, 0]));
      expect(store.size).toBe(1);
    });

    it('should add multiple chunks', () => {
      store.addAll([
        createChunk('c1', [1, 0, 0]),
        createChunk('c2', [0, 1, 0]),
        createChunk('c3', [0, 0, 1]),
      ]);
      expect(store.size).toBe(3);
    });

    it('should get chunk by id', () => {
      store.add(createChunk('c1', [1, 0, 0]));
      const chunk = store.get('c1');
      expect(chunk?.id).toBe('c1');
      expect(chunk?.content).toBe('Content for c1');
    });

    it('should return undefined for unknown id', () => {
      expect(store.get('unknown')).toBeUndefined();
    });

    it('should check if chunk exists', () => {
      store.add(createChunk('c1', [1, 0, 0]));
      expect(store.has('c1')).toBe(true);
      expect(store.has('unknown')).toBe(false);
    });

    it('should remove chunk', () => {
      store.add(createChunk('c1', [1, 0, 0]));
      expect(store.remove('c1')).toBe(true);
      expect(store.size).toBe(0);
      expect(store.has('c1')).toBe(false);
    });

    it('should return false when removing non-existent chunk', () => {
      expect(store.remove('unknown')).toBe(false);
    });

    it('should clear all chunks', () => {
      store.addAll([
        createChunk('c1', [1, 0, 0]),
        createChunk('c2', [0, 1, 0]),
      ]);
      store.clear();
      expect(store.size).toBe(0);
    });

    it('should throw on dimension mismatch when adding', () => {
      expect(() => store.add(createChunk('c1', [1, 0]))).toThrow('dimension mismatch');
    });
  });

  describe('search', () => {
    beforeEach(() => {
      store.addAll([
        createChunk('c1', [1, 0, 0]),
        createChunk('c2', [0.9, 0.1, 0]),
        createChunk('c3', [0, 1, 0]),
        createChunk('c4', [0, 0, 1]),
      ]);
    });

    it('should find most similar chunks', () => {
      const results = store.search([1, 0, 0]);
      expect(results[0].chunk.id).toBe('c1');
      expect(results[0].score).toBeCloseTo(1, 5);
    });

    it('should return results in descending score order', () => {
      const results = store.search([1, 0, 0], { topK: 4 });
      
      for (let i = 1; i < results.length; i++) {
        expect(results[i - 1].score).toBeGreaterThanOrEqual(results[i].score);
      }
    });

    it('should respect topK limit', () => {
      const results = store.search([1, 0, 0], { topK: 2 });
      expect(results.length).toBe(2);
    });

    it('should filter by threshold', () => {
      const results = store.search([1, 0, 0], { threshold: 0.5 });
      
      results.forEach(r => {
        expect(r.score).toBeGreaterThanOrEqual(0.5);
      });
    });

    it('should apply metadata filter', () => {
      store.add({
        id: 'special',
        content: 'Special content',
        embedding: [0.95, 0.05, 0],
        metadata: { type: 'special' },
      });

      const results = store.search([1, 0, 0], {
        filter: meta => meta?.type === 'special',
      });

      expect(results.length).toBe(1);
      expect(results[0].chunk.id).toBe('special');
    });

    it('should return empty array when no matches pass threshold', () => {
      const results = store.search([0, 0, 1], { threshold: 0.99 });
      expect(results.filter(r => r.chunk.id !== 'c4').length).toBe(0);
    });

    it('should not include embeddings in results', () => {
      const results = store.search([1, 0, 0]);
      // @ts-expect-error - checking that embedding is not present
      expect(results[0].chunk.embedding).toBeUndefined();
    });

    it('should throw on dimension mismatch in query', () => {
      expect(() => store.search([1, 0])).toThrow('dimension mismatch');
    });
  });

  describe('serialization', () => {
    beforeEach(() => {
      store.addAll([
        createChunk('c1', [1, 0, 0]),
        createChunk('c2', [0, 1, 0]),
      ]);
    });

    it('should serialize to object', () => {
      const data = store.serialize();
      
      expect(data.version).toBe(1);
      expect(data.model).toBe('test-model');
      expect(data.dimensions).toBe(3);
      expect(data.chunks.length).toBe(2);
    });

    it('should serialize to JSON string', () => {
      const json = store.toJSON();
      const parsed = JSON.parse(json);
      
      expect(parsed.chunks.length).toBe(2);
    });

    it('should deserialize from object', () => {
      const data = store.serialize();
      const restored = VectorStore.fromSerialized(data);
      
      expect(restored.size).toBe(2);
      expect(restored.get('c1')).toBeDefined();
    });

    it('should deserialize from JSON', () => {
      const json = store.toJSON();
      const restored = VectorStore.fromJSON(json);
      
      expect(restored.size).toBe(2);
    });

    it('should serialize/deserialize binary format', () => {
      const binary = store.toBinary();
      const restored = VectorStore.fromBinary(binary);
      
      expect(restored.size).toBe(2);
      expect(restored.get('c1')?.embedding).toEqual([1, 0, 0]);
    });

    it('should preserve search functionality after deserialization', () => {
      const json = store.toJSON();
      const restored = VectorStore.fromJSON(json);
      
      const results = restored.search([1, 0, 0]);
      expect(results[0].chunk.id).toBe('c1');
    });

    it('should throw on invalid version', () => {
      const data = store.serialize();
      data.version = 999;
      
      expect(() => VectorStore.fromSerialized(data)).toThrow('Unsupported index version');
    });

    it('should throw on invalid binary magic', () => {
      const buffer = new ArrayBuffer(16);
      const view = new DataView(buffer);
      view.setUint32(0, 0x12345678, false);
      
      expect(() => VectorStore.fromBinary(buffer)).toThrow('Invalid binary index format');
    });
  });

  describe('iteration', () => {
    it('should be iterable', () => {
      store.addAll([
        createChunk('c1', [1, 0, 0]),
        createChunk('c2', [0, 1, 0]),
      ]);

      const ids = [...store].map(c => c.id);
      expect(ids).toContain('c1');
      expect(ids).toContain('c2');
    });
  });

  describe('properties', () => {
    it('should expose model name', () => {
      expect(store.embeddingModel).toBe('test-model');
    });

    it('should expose dimensions', () => {
      expect(store.embeddingDimensions).toBe(3);
    });
  });
});
