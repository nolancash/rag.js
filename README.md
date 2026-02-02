# rag.js

A RAG (Retrieval Augmented Generation) library for [transformers.js](https://github.com/xenova/transformers.js) with dynamic or precompiled retrieval.

Run RAG entirely in the browser — no backend required.

## Features

- 🌐 **Browser-first** — Works entirely client-side with transformers.js
- ⚡ **Two modes** — Dynamic (compute embeddings at runtime) or Precompiled (load pre-built indices)
- 🔌 **Pluggable embeddings** — Use any transformers.js model or bring your own
- 📦 **Smart chunking** — Automatic text chunking with configurable size and overlap
- 🔍 **Vector search** — Efficient cosine similarity search
- 🛠️ **CLI tool** — Build precompiled indices from markdown/docs at deploy time
- 📄 **TypeScript** — Full type definitions included

## Installation

```bash
npm install rag.js @xenova/transformers
```

## Quick Start

### Dynamic Mode

Compute embeddings at runtime (good for small datasets or dynamic content):

```typescript
import { RAG } from 'rag.js';

// Create a RAG instance
const rag = new RAG();

// Add documents
await rag.addDocument('intro', `
  JavaScript is a programming language commonly used for web development.
  It can run in browsers and on servers with Node.js.
`);

await rag.addDocument('typescript', `
  TypeScript is a typed superset of JavaScript that compiles to plain JavaScript.
  It adds optional static typing and class-based object-oriented programming.
`);

// Search for relevant content
const results = await rag.search('What is TypeScript?');

console.log(results[0].chunk.content);
// "TypeScript is a typed superset of JavaScript..."

console.log(results[0].score);
// 0.87
```

### Precompiled Mode

Load a pre-built index for instant startup (good for static documentation):

```typescript
import { RAG } from 'rag.js';

// Load precompiled index (built at deploy time)
const response = await fetch('/docs-index.json');
const index = await response.json();

const rag = RAG.fromIndex(index);

// Search immediately — no embedding computation needed
const results = await rag.search('How do I install?');
```

### Building Precompiled Indices

Use the CLI to build indices at deploy time:

```bash
# Build from a directory of markdown files
npx ragjs build --input ./docs --output ./public/index.json

# With options
npx ragjs build \
  --input ./docs \
  --output ./dist/index.json \
  --model Xenova/all-MiniLM-L6-v2 \
  --chunk-size 512 \
  --chunk-overlap 50 \
  --verbose

# Binary format (smaller file size)
npx ragjs build --input ./docs --output ./index.bin --format binary

# Inspect an index
npx ragjs info ./index.json
```

## API Reference

### RAG Class

```typescript
import { RAG } from 'rag.js';

// Create with options
const rag = new RAG({
  model: 'Xenova/all-MiniLM-L6-v2', // embedding model
  chunkOptions: {
    chunkSize: 512,    // max characters per chunk
    chunkOverlap: 50,  // overlap between chunks
  },
  searchOptions: {
    topK: 5,           // default number of results
    threshold: 0,      // minimum similarity score
  },
});

// Properties
rag.model;      // embedding model name
rag.size;       // number of chunks
rag.dimensions; // embedding dimensions

// Add documents
await rag.addDocument(id, content, metadata?);
await rag.addDocuments([{ id, content, metadata? }]);
await rag.addChunks([{ id, content, metadata? }]);

// Remove
rag.remove(chunkId);           // remove single chunk
rag.removeDocument(documentId); // remove all chunks from document
rag.clear();                    // remove everything

// Search
const results = await rag.search(query, {
  topK: 5,
  threshold: 0.5,
  filter: (metadata) => metadata?.type === 'api',
});

// Get concatenated context for LLM prompts
const context = await rag.getContext(query, { topK: 3 });

// Serialization
const json = rag.toJSON();
const binary = rag.toBinary();
const data = rag.serialize();

// Load from serialized
const rag2 = RAG.fromJSON(json);
const rag3 = RAG.fromBinary(buffer);
const rag4 = RAG.fromIndex(data);
```

### Chunking Utilities

```typescript
import { chunkText, createChunks } from 'rag.js';

// Simple text chunking
const chunks = chunkText(longText, {
  chunkSize: 512,
  chunkOverlap: 50,
  separator: /\n\n/, // optional custom separator
});

// Create chunks with IDs and metadata
const chunks = createChunks(text, 'doc-id', { source: 'file.md' });
```

### Custom Embeddings

```typescript
import { RAG, CustomEmbedding } from 'rag.js';

// Use your own embedding function
const myEmbedding = new CustomEmbedding(
  async (texts) => {
    // Call your embedding API
    const response = await fetch('/api/embed', {
      method: 'POST',
      body: JSON.stringify({ texts }),
    });
    return response.json();
  },
  { dimensions: 384 }
);

const rag = RAG.withEmbedding(myEmbedding);
```

### Vector Store (Low-level)

```typescript
import { VectorStore, cosineSimilarity } from 'rag.js';

const store = new VectorStore('model-name', 384);

store.add({ id: 'c1', content: 'text', embedding: [...] });
store.addAll([...chunks]);

const results = store.search(queryEmbedding, { topK: 5 });

// Serialization
const json = store.toJSON();
const binary = store.toBinary();
const restored = VectorStore.fromJSON(json);
```

## Embedding Models

rag.js works with any transformers.js embedding model. Some good options:

| Model | Dimensions | Size | Notes |
|-------|------------|------|-------|
| `Xenova/all-MiniLM-L6-v2` | 384 | 23MB | Fast, good quality (default) |
| `Xenova/all-MiniLM-L12-v2` | 384 | 34MB | Better quality |
| `Xenova/bge-small-en-v1.5` | 384 | 33MB | High quality |
| `Xenova/bge-base-en-v1.5` | 768 | 109MB | Higher quality |
| `Xenova/gte-small` | 384 | 33MB | Good for retrieval |

## Examples

### Documentation Chatbot

```typescript
import { RAG } from 'rag.js';

// Load pre-built docs index
const rag = RAG.fromJSON(await fetch('/docs-index.json').then(r => r.text()));

async function answerQuestion(question: string) {
  // Get relevant context
  const context = await rag.getContext(question, { topK: 3 });
  
  // Send to LLM with context
  const response = await fetch('/api/chat', {
    method: 'POST',
    body: JSON.stringify({
      messages: [
        { role: 'system', content: `Answer based on this context:\n\n${context}` },
        { role: 'user', content: question },
      ],
    }),
  });
  
  return response.json();
}
```

### Filtering by Metadata

```typescript
await rag.addDocument('api-auth', content, { category: 'api', section: 'auth' });
await rag.addDocument('guide-intro', content, { category: 'guide', section: 'intro' });

// Search only API docs
const results = await rag.search('authentication', {
  filter: (meta) => meta?.category === 'api',
});
```

## Browser Usage

rag.js works in browsers via bundlers (Vite, webpack, etc.) or directly:

```html
<script type="module">
  import { RAG } from 'https://esm.sh/rag.js';
  
  const rag = RAG.fromJSON(await fetch('/index.json').then(r => r.text()));
  const results = await rag.search('hello');
</script>
```

## License

MIT
