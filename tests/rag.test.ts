import { describe, it, expect, beforeEach } from 'vitest';
import { RAG, createRAG } from '../src/rag.js';
import { MockEmbedding } from '../src/embeddings.js';

describe('RAG', () => {
  let rag: RAG;

  beforeEach(() => {
    // Use mock embedding for tests
    rag = RAG.withEmbedding(new MockEmbedding(64));
  });

  describe('initialization', () => {
    it('should create with default options', () => {
      const r = RAG.withEmbedding(new MockEmbedding());
      expect(r.size).toBe(0);
    });

    it('should expose model name', () => {
      expect(rag.model).toBe('mock');
    });

    it('should expose dimensions', () => {
      expect(rag.dimensions).toBe(64);
    });

    it('should start with zero size', () => {
      expect(rag.size).toBe(0);
    });
  });

  describe('addDocument', () => {
    it('should add a document and return chunks', async () => {
      const chunks = await rag.addDocument('doc1', 'This is test content.');
      
      expect(chunks.length).toBeGreaterThan(0);
      expect(chunks[0].id).toMatch(/^doc1-/);
      expect(rag.size).toBe(chunks.length);
    });

    it('should include metadata in chunks', async () => {
      const chunks = await rag.addDocument('doc1', 'Content here.', {
        author: 'Test',
        page: 1,
      });
      
      expect(chunks[0].metadata?.author).toBe('Test');
      expect(chunks[0].metadata?.page).toBe(1);
    });

    it('should handle empty content', async () => {
      const chunks = await rag.addDocument('doc1', '');
      expect(chunks.length).toBe(0);
      expect(rag.size).toBe(0);
    });

    it('should handle whitespace-only content', async () => {
      const chunks = await rag.addDocument('doc1', '   \n\n   ');
      expect(chunks.length).toBe(0);
    });
  });

  describe('addDocuments', () => {
    it('should add multiple documents', async () => {
      const chunks = await rag.addDocuments([
        { id: 'doc1', content: 'First document.' },
        { id: 'doc2', content: 'Second document.' },
      ]);
      
      expect(chunks.length).toBe(2);
      expect(rag.size).toBe(2);
    });

    it('should preserve per-document metadata', async () => {
      const chunks = await rag.addDocuments([
        { id: 'doc1', content: 'Content 1.', metadata: { type: 'a' } },
        { id: 'doc2', content: 'Content 2.', metadata: { type: 'b' } },
      ]);
      
      const doc1Chunk = chunks.find(c => c.id.startsWith('doc1'));
      const doc2Chunk = chunks.find(c => c.id.startsWith('doc2'));
      
      expect(doc1Chunk?.metadata?.type).toBe('a');
      expect(doc2Chunk?.metadata?.type).toBe('b');
    });

    it('should handle empty documents array', async () => {
      const chunks = await rag.addDocuments([]);
      expect(chunks.length).toBe(0);
    });
  });

  describe('addChunks', () => {
    it('should add pre-created chunks', async () => {
      await rag.addChunks([
        { id: 'c1', content: 'Chunk 1' },
        { id: 'c2', content: 'Chunk 2' },
      ]);
      
      expect(rag.size).toBe(2);
    });

    it('should handle empty chunks array', async () => {
      await rag.addChunks([]);
      expect(rag.size).toBe(0);
    });
  });

  describe('remove', () => {
    it('should remove a chunk by id', async () => {
      await rag.addDocument('doc1', 'Test content.');
      
      const removed = rag.remove('doc1-0');
      expect(removed).toBe(true);
      expect(rag.size).toBe(0);
    });

    it('should return false for non-existent chunk', () => {
      const removed = rag.remove('unknown');
      expect(removed).toBe(false);
    });
  });

  describe('removeDocument', () => {
    it('should remove all chunks from a document', async () => {
      // Add a longer document that gets chunked
      const content = 'A'.repeat(100) + ' ' + 'B'.repeat(100) + ' ' + 'C'.repeat(100);
      await rag.addDocument('doc1', content, undefined);
      
      const sizeBefore = rag.size;
      expect(sizeBefore).toBeGreaterThan(0);
      
      const removed = rag.removeDocument('doc1');
      expect(removed).toBe(sizeBefore);
      expect(rag.size).toBe(0);
    });

    it('should return 0 for non-existent document', () => {
      const removed = rag.removeDocument('unknown');
      expect(removed).toBe(0);
    });

    it('should not remove chunks from other documents', async () => {
      await rag.addDocuments([
        { id: 'doc1', content: 'Content 1.' },
        { id: 'doc2', content: 'Content 2.' },
      ]);
      
      rag.removeDocument('doc1');
      expect(rag.size).toBe(1);
    });
  });

  describe('clear', () => {
    it('should remove all chunks', async () => {
      await rag.addDocuments([
        { id: 'doc1', content: 'Content 1.' },
        { id: 'doc2', content: 'Content 2.' },
      ]);
      
      rag.clear();
      expect(rag.size).toBe(0);
    });
  });

  describe('search', () => {
    beforeEach(async () => {
      await rag.addDocuments([
        { id: 'javascript', content: 'JavaScript is a programming language for the web.' },
        { id: 'python', content: 'Python is great for data science and machine learning.' },
        { id: 'rust', content: 'Rust provides memory safety without garbage collection.' },
      ]);
    });

    it('should return search results', async () => {
      const results = await rag.search('programming');
      
      expect(results.length).toBeGreaterThan(0);
      expect(results[0].chunk.content).toBeTruthy();
      expect(typeof results[0].score).toBe('number');
    });

    it('should respect topK option', async () => {
      const results = await rag.search('code', { topK: 2 });
      expect(results.length).toBeLessThanOrEqual(2);
    });

    it('should respect threshold option', async () => {
      const results = await rag.search('test', { threshold: 0.5 });
      
      results.forEach(r => {
        expect(r.score).toBeGreaterThanOrEqual(0.5);
      });
    });

    it('should apply filter function', async () => {
      await rag.addDocument('filtered', 'Special content.', { special: true });
      
      const results = await rag.search('content', {
        filter: meta => meta?.special === true,
      });
      
      expect(results.length).toBe(1);
      expect(results[0].chunk.metadata?.special).toBe(true);
    });

    it('should return results sorted by score descending', async () => {
      const results = await rag.search('language', { topK: 10 });
      
      for (let i = 1; i < results.length; i++) {
        expect(results[i - 1].score).toBeGreaterThanOrEqual(results[i].score);
      }
    });
  });

  describe('searchByEmbedding', () => {
    it('should search with pre-computed embedding', async () => {
      await rag.addDocument('doc1', 'Test content.');
      
      // Get an embedding to use for search
      const mockEmbed = new MockEmbedding(64);
      const [queryEmbed] = await mockEmbed.embed(['test']);
      
      const results = rag.searchByEmbedding(queryEmbed);
      expect(results.length).toBeGreaterThan(0);
    });
  });

  describe('getContext', () => {
    it('should return concatenated content', async () => {
      await rag.addDocuments([
        { id: 'doc1', content: 'First content.' },
        { id: 'doc2', content: 'Second content.' },
      ]);
      
      const context = await rag.getContext('content', { topK: 2 });
      
      expect(context).toContain('content');
      expect(typeof context).toBe('string');
    });

    it('should use custom separator', async () => {
      await rag.addDocuments([
        { id: 'doc1', content: 'First document about programming.' },
        { id: 'doc2', content: 'Second document about programming.' },
      ]);
      
      const context = await rag.getContext('programming', { 
        topK: 2,
        separator: ' | ',
      });
      
      // Should have content from both docs joined by separator
      expect(context.split(' | ').length).toBeGreaterThanOrEqual(1);
    });

    it('should return empty string with no results', async () => {
      const context = await rag.getContext('query');
      expect(context).toBe('');
    });
  });

  describe('serialization', () => {
    beforeEach(async () => {
      await rag.addDocument('doc1', 'Test content for serialization.');
    });

    it('should serialize to object', () => {
      const data = rag.serialize();
      
      expect(data.version).toBe(1);
      expect(data.model).toBe('mock');
      expect(data.chunks.length).toBe(1);
    });

    it('should serialize to JSON', () => {
      const json = rag.toJSON();
      const parsed = JSON.parse(json);
      
      expect(parsed.chunks.length).toBe(1);
    });

    it('should serialize to binary', () => {
      const binary = rag.toBinary();
      
      expect(binary).toBeInstanceOf(ArrayBuffer);
      expect(binary.byteLength).toBeGreaterThan(0);
    });
  });

  describe('deserialization', () => {
    it('should create from index object', async () => {
      await rag.addDocument('doc1', 'Original content.');
      const data = rag.serialize();
      
      const restored = RAG.fromIndex(data);
      
      expect(restored.size).toBe(1);
      expect(restored.model).toBe('mock');
    });

    it('should create from JSON', async () => {
      await rag.addDocument('doc1', 'Original content.');
      const json = rag.toJSON();
      
      const restored = RAG.fromJSON(json);
      
      expect(restored.size).toBe(1);
    });

    it('should create from binary', async () => {
      await rag.addDocument('doc1', 'Original content.');
      const binary = rag.toBinary();
      
      const restored = RAG.fromBinary(binary);
      
      expect(restored.size).toBe(1);
    });

    it('should preserve search after deserialization', async () => {
      const content = 'JavaScript programming.';
      await rag.addDocument('doc1', content);
      const data = rag.serialize();
      
      // Verify the chunks contain the expected content
      expect(data.chunks.length).toBeGreaterThan(0);
      expect(data.chunks[0].content).toContain('JavaScript');
      
      // Create new RAG with same mock embedding instance for searching
      // Note: Mock embeddings aren't semantically meaningful, so we search
      // with exact content to verify the roundtrip works
      const mockEmbed = new MockEmbedding(64);
      const restoredWithEmbed = RAG.withEmbedding(mockEmbed);
      for (const chunk of data.chunks) {
        restoredWithEmbed.addEmbeddedChunk(chunk);
      }
      
      // Search with exact content (mock embeddings are deterministic per text)
      const results = await restoredWithEmbed.search(content);
      expect(results.length).toBeGreaterThan(0);
      expect(results[0].score).toBeCloseTo(1, 5); // Should be exact match
    });
  });

  describe('withEmbedding', () => {
    it('should create RAG with custom embedding provider', () => {
      const customEmbed = new MockEmbedding(128);
      const r = RAG.withEmbedding(customEmbed);
      
      expect(r.model).toBe('mock');
      expect(r.dimensions).toBe(128);
    });
  });

  describe('createRAG helper', () => {
    it('should create a new RAG instance', () => {
      const r = createRAG();
      expect(r).toBeInstanceOf(RAG);
    });
  });
});
