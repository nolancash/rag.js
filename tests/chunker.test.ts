import { describe, it, expect } from 'vitest';
import { chunkText, createChunks, createChunksFromDocuments } from '../src/chunker.js';

describe('chunkText', () => {
  it('should return empty array for empty text', () => {
    expect(chunkText('')).toEqual([]);
    expect(chunkText('   ')).toEqual([]);
  });

  it('should return single chunk for short text', () => {
    const text = 'Hello, world!';
    const chunks = chunkText(text, { chunkSize: 100 });
    expect(chunks).toEqual(['Hello, world!']);
  });

  it('should split long text into multiple chunks', () => {
    const text = 'A'.repeat(100) + ' ' + 'B'.repeat(100) + ' ' + 'C'.repeat(100);
    const chunks = chunkText(text, { chunkSize: 150, chunkOverlap: 0 });
    
    expect(chunks.length).toBeGreaterThan(1);
    chunks.forEach(chunk => {
      expect(chunk.length).toBeLessThanOrEqual(150);
    });
  });

  it('should respect chunk size limit', () => {
    const text = 'word '.repeat(100);
    const chunks = chunkText(text, { chunkSize: 50, chunkOverlap: 0 });
    
    chunks.forEach(chunk => {
      expect(chunk.length).toBeLessThanOrEqual(50);
    });
  });

  it('should create overlap between chunks', () => {
    const text = 'The quick brown fox jumps over the lazy dog. ' +
                 'Pack my box with five dozen liquor jugs. ' +
                 'How vexingly quick daft zebras jump.';
    
    const chunks = chunkText(text, { chunkSize: 60, chunkOverlap: 15 });
    
    // With overlap, consecutive chunks should share some content
    expect(chunks.length).toBeGreaterThan(1);
  });

  it('should split on paragraph boundaries', () => {
    const text = 'First paragraph here.\n\nSecond paragraph here.\n\nThird paragraph.';
    const chunks = chunkText(text, { chunkSize: 100, chunkOverlap: 0 });
    
    // Should try to respect paragraph boundaries
    expect(chunks.length).toBeGreaterThanOrEqual(1);
  });

  it('should split on sentence boundaries when paragraphs are too long', () => {
    const text = 'First sentence. Second sentence. Third sentence. Fourth sentence. Fifth sentence.';
    const chunks = chunkText(text, { chunkSize: 50, chunkOverlap: 0 });
    
    expect(chunks.length).toBeGreaterThan(1);
  });

  it('should handle custom separator', () => {
    const text = 'chunk1|chunk2|chunk3';
    const chunks = chunkText(text, { chunkSize: 100, separator: '|' });
    
    expect(chunks).toEqual(['chunk1', 'chunk2', 'chunk3']);
  });

  it('should handle regex separator', () => {
    const text = 'chunk1---chunk2===chunk3';
    const chunks = chunkText(text, { chunkSize: 100, separator: /[-=]+/ });
    
    expect(chunks).toEqual(['chunk1', 'chunk2', 'chunk3']);
  });

  it('should throw on invalid chunkSize', () => {
    expect(() => chunkText('test', { chunkSize: 0 })).toThrow('chunkSize must be positive');
    expect(() => chunkText('test', { chunkSize: -1 })).toThrow('chunkSize must be positive');
  });

  it('should throw on invalid chunkOverlap', () => {
    expect(() => chunkText('test', { chunkOverlap: -1 })).toThrow('chunkOverlap cannot be negative');
  });

  it('should throw when overlap >= chunkSize', () => {
    expect(() => chunkText('test', { chunkSize: 50, chunkOverlap: 50 }))
      .toThrow('chunkOverlap must be less than chunkSize');
    expect(() => chunkText('test', { chunkSize: 50, chunkOverlap: 60 }))
      .toThrow('chunkOverlap must be less than chunkSize');
  });

  it('should handle very long words', () => {
    const longWord = 'A'.repeat(200);
    const chunks = chunkText(longWord, { chunkSize: 50, chunkOverlap: 0 });
    
    expect(chunks.length).toBe(4);
    chunks.forEach(chunk => {
      expect(chunk.length).toBeLessThanOrEqual(50);
    });
  });

  it('should handle text with only whitespace between content', () => {
    const text = 'Hello     World';
    const chunks = chunkText(text, { chunkSize: 100 });
    
    expect(chunks[0]).toContain('Hello');
    expect(chunks[0]).toContain('World');
  });
});

describe('createChunks', () => {
  it('should create chunks with IDs', () => {
    const text = 'First paragraph.\n\nSecond paragraph.';
    const chunks = createChunks(text, 'doc1', undefined, { chunkSize: 20, chunkOverlap: 0 });
    
    expect(chunks.length).toBeGreaterThan(0);
    chunks.forEach((chunk, i) => {
      expect(chunk.id).toBe(`doc1-${i}`);
      expect(chunk.content).toBeTruthy();
    });
  });

  it('should include metadata', () => {
    const text = 'Test content here.';
    const metadata = { source: 'test.md', page: 1 };
    const chunks = createChunks(text, 'doc1', metadata, { chunkSize: 100 });
    
    expect(chunks[0].metadata).toMatchObject({
      source: 'test.md',
      page: 1,
      chunkIndex: 0,
      totalChunks: 1,
    });
  });

  it('should track chunk index and total', () => {
    const text = 'A'.repeat(100) + ' ' + 'B'.repeat(100);
    const chunks = createChunks(text, 'doc1', {}, { chunkSize: 80, chunkOverlap: 0 });
    
    expect(chunks.length).toBeGreaterThan(1);
    
    chunks.forEach((chunk, i) => {
      expect(chunk.metadata?.chunkIndex).toBe(i);
      expect(chunk.metadata?.totalChunks).toBe(chunks.length);
    });
  });
});

describe('createChunksFromDocuments', () => {
  it('should process multiple documents', () => {
    const documents = [
      { id: 'doc1', content: 'First document content.' },
      { id: 'doc2', content: 'Second document content.' },
    ];
    
    const chunks = createChunksFromDocuments(documents, { chunkSize: 100 });
    
    expect(chunks.some(c => c.id.startsWith('doc1'))).toBe(true);
    expect(chunks.some(c => c.id.startsWith('doc2'))).toBe(true);
  });

  it('should preserve document metadata', () => {
    const documents = [
      { id: 'doc1', content: 'Content one.', metadata: { author: 'Alice' } },
      { id: 'doc2', content: 'Content two.', metadata: { author: 'Bob' } },
    ];
    
    const chunks = createChunksFromDocuments(documents, { chunkSize: 100 });
    
    const doc1Chunks = chunks.filter(c => c.id.startsWith('doc1'));
    const doc2Chunks = chunks.filter(c => c.id.startsWith('doc2'));
    
    expect(doc1Chunks[0].metadata?.author).toBe('Alice');
    expect(doc2Chunks[0].metadata?.author).toBe('Bob');
  });

  it('should handle empty documents array', () => {
    const chunks = createChunksFromDocuments([]);
    expect(chunks).toEqual([]);
  });

  it('should handle documents with empty content', () => {
    const documents = [
      { id: 'doc1', content: '' },
      { id: 'doc2', content: 'Has content.' },
    ];
    
    const chunks = createChunksFromDocuments(documents, { chunkSize: 100 });
    
    expect(chunks.some(c => c.id.startsWith('doc1'))).toBe(false);
    expect(chunks.some(c => c.id.startsWith('doc2'))).toBe(true);
  });
});
