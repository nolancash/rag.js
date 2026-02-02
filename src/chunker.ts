/**
 * Text chunking utilities for rag.js
 */

import type { Chunk, ChunkOptions } from './types.js';

const DEFAULT_CHUNK_SIZE = 512;
const DEFAULT_CHUNK_OVERLAP = 50;

// Regex patterns for splitting
const PARAGRAPH_SPLIT = /\n\n+/;
const SENTENCE_SPLIT = /(?<=[.!?])\s+/;
const WORD_SPLIT = /\s+/;

/**
 * Split text into chunks with configurable size and overlap
 */
export function chunkText(
  text: string,
  options: ChunkOptions = {}
): string[] {
  const {
    chunkSize = DEFAULT_CHUNK_SIZE,
    chunkOverlap = DEFAULT_CHUNK_OVERLAP,
    separator,
  } = options;

  if (chunkSize <= 0) {
    throw new Error('chunkSize must be positive');
  }
  if (chunkOverlap < 0) {
    throw new Error('chunkOverlap cannot be negative');
  }
  if (chunkOverlap >= chunkSize) {
    throw new Error('chunkOverlap must be less than chunkSize');
  }

  // Split by custom separator first (regardless of size)
  let segments: string[];
  if (separator) {
    segments = text.split(separator).filter(s => s.trim());
    // If custom separator was used and segments fit, return them
    if (segments.every(s => s.length <= chunkSize)) {
      return segments.map(s => s.trim());
    }
    // Segments exist but some are too large, will be split further in mergeSegments
  } else {
    // If text fits in one chunk and no custom separator, return it
    if (text.length <= chunkSize) {
      return text.trim() ? [text.trim()] : [];
    }
    // Split by natural boundaries
    segments = splitByBoundary(text, chunkSize);
  }

  // Merge segments into chunks respecting size limits
  return mergeSegments(segments, chunkSize, chunkOverlap);
}

/**
 * Split text by natural boundaries (paragraphs, sentences, words)
 */
function splitByBoundary(text: string, maxSize: number): string[] {
  // Try paragraph splits first
  let segments = text.split(PARAGRAPH_SPLIT).filter(s => s.trim());
  
  // If any segment is still too large, split by sentences
  const needsSentenceSplit = segments.some(s => s.length > maxSize);
  if (needsSentenceSplit) {
    segments = segments.flatMap(segment => {
      if (segment.length <= maxSize) return [segment];
      const sentences = segment.split(SENTENCE_SPLIT).filter(s => s.trim());
      // If sentences are still too large, split by words
      return sentences.flatMap(sentence => {
        if (sentence.length <= maxSize) return [sentence];
        return splitByWords(sentence, maxSize);
      });
    });
  }

  return segments;
}

/**
 * Split text by words when other boundaries fail
 */
function splitByWords(text: string, maxSize: number): string[] {
  const words = text.split(WORD_SPLIT);
  const chunks: string[] = [];
  let current = '';

  for (const word of words) {
    // Handle words longer than maxSize
    if (word.length > maxSize) {
      if (current) {
        chunks.push(current.trim());
        current = '';
      }
      // Split the long word
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

/**
 * Merge segments into chunks with overlap
 */
function mergeSegments(
  segments: string[],
  chunkSize: number,
  overlap: number
): string[] {
  if (segments.length === 0) return [];

  const chunks: string[] = [];
  let current = '';
  let overlapBuffer = '';

  for (const segment of segments) {
    const separator = current ? '\n\n' : '';
    const candidate = current + separator + segment;

    if (candidate.length <= chunkSize) {
      current = candidate;
    } else {
      // Save current chunk
      if (current) {
        chunks.push(current.trim());
        // Keep the end portion for overlap
        if (overlap > 0) {
          overlapBuffer = getOverlapText(current, overlap);
        }
      }
      
      // Start new chunk with overlap
      if (overlapBuffer && segment.length + overlapBuffer.length + 1 <= chunkSize) {
        current = overlapBuffer + ' ' + segment;
      } else {
        current = segment;
      }
      
      // Handle segment larger than chunk size
      while (current.length > chunkSize) {
        const splitPoint = findSplitPoint(current, chunkSize);
        chunks.push(current.slice(0, splitPoint).trim());
        overlapBuffer = getOverlapText(current.slice(0, splitPoint), overlap);
        const remaining = current.slice(splitPoint).trim();
        current = overlapBuffer ? overlapBuffer + ' ' + remaining : remaining;
      }
    }
  }

  if (current.trim()) {
    chunks.push(current.trim());
  }

  return chunks;
}

/**
 * Get text for overlap from the end of a chunk
 */
function getOverlapText(text: string, overlap: number): string {
  if (overlap <= 0 || text.length <= overlap) return '';
  
  // Try to split on word boundary
  const slice = text.slice(-overlap);
  const wordBoundary = slice.search(/\s/);
  
  if (wordBoundary > 0 && wordBoundary < slice.length - 1) {
    return slice.slice(wordBoundary + 1);
  }
  
  return slice;
}

/**
 * Find a good split point near the target length
 */
function findSplitPoint(text: string, target: number): number {
  if (text.length <= target) return text.length;
  
  // Look for sentence boundary near target
  const searchStart = Math.max(0, target - 50);
  const searchEnd = Math.min(text.length, target + 50);
  const searchRegion = text.slice(searchStart, searchEnd);
  
  const sentenceMatch = searchRegion.match(/[.!?]\s+/);
  if (sentenceMatch && sentenceMatch.index !== undefined) {
    return searchStart + sentenceMatch.index + sentenceMatch[0].length;
  }
  
  // Look for word boundary
  const lastSpace = text.lastIndexOf(' ', target);
  if (lastSpace > target * 0.5) {
    return lastSpace + 1;
  }
  
  return target;
}

/**
 * Create chunks from text with IDs and metadata
 */
export function createChunks(
  text: string,
  baseId: string,
  metadata?: Record<string, unknown>,
  options?: ChunkOptions
): Chunk[] {
  const texts = chunkText(text, options);
  
  return texts.map((content, index) => ({
    id: `${baseId}-${index}`,
    content,
    metadata: {
      ...metadata,
      chunkIndex: index,
      totalChunks: texts.length,
    },
  }));
}

/**
 * Create chunks from multiple documents
 */
export function createChunksFromDocuments(
  documents: Array<{ id: string; content: string; metadata?: Record<string, unknown> }>,
  options?: ChunkOptions
): Chunk[] {
  return documents.flatMap(doc => 
    createChunks(doc.content, doc.id, doc.metadata, options)
  );
}
