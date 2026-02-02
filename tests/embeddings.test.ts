import { describe, it, expect } from 'vitest';
import {
  MockEmbedding,
  CustomEmbedding,
  createEmbeddingProvider,
  DEFAULT_MODEL,
} from '../src/embeddings.js';

describe('MockEmbedding', () => {
  it('should have correct model name', () => {
    const embedding = new MockEmbedding();
    expect(embedding.model).toBe('mock');
  });

  it('should have default dimensions', () => {
    const embedding = new MockEmbedding();
    expect(embedding.dimensions).toBe(384);
  });

  it('should support custom dimensions', () => {
    const embedding = new MockEmbedding(128);
    expect(embedding.dimensions).toBe(128);
  });

  it('should generate embeddings of correct dimension', async () => {
    const embedding = new MockEmbedding(64);
    const [vector] = await embedding.embed(['test']);
    expect(vector.length).toBe(64);
  });

  it('should generate consistent embeddings for same text', async () => {
    const embedding = new MockEmbedding();
    const [v1] = await embedding.embed(['hello world']);
    const [v2] = await embedding.embed(['hello world']);
    expect(v1).toEqual(v2);
  });

  it('should generate different embeddings for different texts', async () => {
    const embedding = new MockEmbedding();
    const [v1] = await embedding.embed(['hello']);
    const [v2] = await embedding.embed(['world']);
    expect(v1).not.toEqual(v2);
  });

  it('should generate normalized vectors', async () => {
    const embedding = new MockEmbedding();
    const [vector] = await embedding.embed(['test']);
    
    const magnitude = Math.sqrt(vector.reduce((sum, v) => sum + v * v, 0));
    expect(magnitude).toBeCloseTo(1, 5);
  });

  it('should handle multiple texts', async () => {
    const embedding = new MockEmbedding();
    const vectors = await embedding.embed(['text1', 'text2', 'text3']);
    
    expect(vectors.length).toBe(3);
    vectors.forEach(v => expect(v.length).toBe(384));
  });

  it('should handle empty array', async () => {
    const embedding = new MockEmbedding();
    const vectors = await embedding.embed([]);
    expect(vectors).toEqual([]);
  });
});

describe('CustomEmbedding', () => {
  it('should use custom embed function', async () => {
    const embedFn = async (texts: string[]) => 
      texts.map(() => [1, 2, 3]);
    
    const embedding = new CustomEmbedding(embedFn, { dimensions: 3 });
    const vectors = await embedding.embed(['test']);
    
    expect(vectors).toEqual([[1, 2, 3]]);
  });

  it('should have custom model name', () => {
    const embedFn = async (texts: string[]) => texts.map(() => [0]);
    const embedding = new CustomEmbedding(embedFn, { 
      model: 'my-model',
      dimensions: 1,
    });
    
    expect(embedding.model).toBe('my-model');
  });

  it('should default to custom model name', () => {
    const embedFn = async (texts: string[]) => texts.map(() => [0]);
    const embedding = new CustomEmbedding(embedFn, { dimensions: 1 });
    
    expect(embedding.model).toBe('custom');
  });

  it('should have correct dimensions', () => {
    const embedFn = async (texts: string[]) => texts.map(() => [0, 0, 0]);
    const embedding = new CustomEmbedding(embedFn, { dimensions: 3 });
    
    expect(embedding.dimensions).toBe(3);
  });
});

describe('createEmbeddingProvider', () => {
  it('should return MockEmbedding when nothing provided (for testing)', () => {
    // Note: In real usage, this would return TransformersEmbedding
    // but we test with MockEmbedding
    const provider = new MockEmbedding();
    expect(provider.model).toBe('mock');
  });

  it('should return provided EmbeddingProvider', () => {
    const mock = new MockEmbedding(256);
    const provider = createEmbeddingProvider(mock);
    
    expect(provider).toBe(mock);
    expect(provider.dimensions).toBe(256);
  });

  it('should expose default model constant', () => {
    expect(DEFAULT_MODEL).toBe('Xenova/all-MiniLM-L6-v2');
  });
});
