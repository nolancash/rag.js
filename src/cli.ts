#!/usr/bin/env node

/**
 * rag.js CLI - Build precompiled vector indices from documents
 * 
 * Usage:
 *   ragjs build --input ./docs --output ./index.json
 *   ragjs build --input ./docs --output ./index.bin --format binary
 *   ragjs info ./index.json
 */

import { readFileSync, writeFileSync, readdirSync, statSync, existsSync } from 'node:fs';
import { join, extname, basename, relative } from 'node:path';
import { RAG } from './rag.js';

interface BuildOptions {
  input: string;
  output: string;
  format: 'json' | 'binary';
  model: string;
  chunkSize: number;
  chunkOverlap: number;
  extensions: string[];
  verbose: boolean;
}

interface InfoResult {
  version: number;
  model: string;
  dimensions: number;
  chunks: number;
  documents: number;
}

/**
 * Recursively find all files with given extensions
 */
function findFiles(dir: string, extensions: string[]): string[] {
  const files: string[] = [];
  
  function walk(currentDir: string) {
    const entries = readdirSync(currentDir);
    
    for (const entry of entries) {
      const fullPath = join(currentDir, entry);
      const stat = statSync(fullPath);
      
      if (stat.isDirectory()) {
        // Skip hidden directories and node_modules
        if (!entry.startsWith('.') && entry !== 'node_modules') {
          walk(fullPath);
        }
      } else if (stat.isFile()) {
        const ext = extname(entry).toLowerCase();
        if (extensions.includes(ext)) {
          files.push(fullPath);
        }
      }
    }
  }
  
  walk(dir);
  return files;
}

/**
 * Extract document ID from file path
 */
function fileToDocId(filePath: string, baseDir: string): string {
  const rel = relative(baseDir, filePath);
  return rel.replace(/\.[^.]+$/, '').replace(/[/\\]/g, '-');
}

/**
 * Build a vector index from documents
 */
async function build(options: BuildOptions): Promise<void> {
  const {
    input,
    output,
    format,
    model,
    chunkSize,
    chunkOverlap,
    extensions,
    verbose,
  } = options;

  const log = verbose ? console.log.bind(console) : () => {};

  // Validate input
  if (!existsSync(input)) {
    console.error(`Error: Input path does not exist: ${input}`);
    process.exit(1);
  }

  const inputStat = statSync(input);
  const files = inputStat.isDirectory()
    ? findFiles(input, extensions)
    : [input];

  if (files.length === 0) {
    console.error(`Error: No files found with extensions: ${extensions.join(', ')}`);
    process.exit(1);
  }

  log(`Found ${files.length} file(s) to process`);
  log(`Using model: ${model}`);

  // Create RAG instance
  const rag = new RAG({
    model,
    chunkOptions: {
      chunkSize,
      chunkOverlap,
    },
  });

  // Process files
  let totalChunks = 0;
  const baseDir = inputStat.isDirectory() ? input : join(input, '..');

  for (const file of files) {
    log(`Processing: ${file}`);
    
    const content = readFileSync(file, 'utf-8');
    const docId = fileToDocId(file, baseDir);
    
    const chunks = await rag.addDocument(docId, content, {
      source: relative(baseDir, file),
      filename: basename(file),
    });
    
    totalChunks += chunks.length;
    log(`  -> ${chunks.length} chunks`);
  }

  log(`Total: ${totalChunks} chunks from ${files.length} documents`);

  // Write output
  log(`Writing ${format} index to: ${output}`);

  if (format === 'binary') {
    const buffer = rag.toBinary();
    writeFileSync(output, Buffer.from(buffer));
  } else {
    writeFileSync(output, rag.toJSON());
  }

  console.log(`✓ Built index: ${files.length} documents, ${totalChunks} chunks`);
}

/**
 * Show info about an index file
 */
function info(indexPath: string): InfoResult {
  if (!existsSync(indexPath)) {
    console.error(`Error: Index file does not exist: ${indexPath}`);
    process.exit(1);
  }

  const buffer = readFileSync(indexPath);
  let data: { version: number; model: string; dimensions: number; chunks: { metadata?: { source?: string } }[] };

  // Detect format
  if (buffer[0] === 0x52 && buffer[1] === 0x41 && buffer[2] === 0x47 && buffer[3] === 0x53) {
    // Binary format (starts with "RAGS")
    // eslint-disable-next-line @typescript-eslint/no-var-requires
    const { VectorStore } = require('./vector-store.js') as typeof import('./vector-store.js');
    const store = VectorStore.fromBinary(buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength));
    data = store.serialize();
  } else {
    // JSON format
    data = JSON.parse(buffer.toString('utf-8'));
  }

  const documents = new Set(data.chunks.map(c => c.metadata?.source || 'unknown'));

  const result: InfoResult = {
    version: data.version,
    model: data.model,
    dimensions: data.dimensions,
    chunks: data.chunks.length,
    documents: documents.size,
  };

  console.log(`Index: ${indexPath}`);
  console.log(`  Version: ${result.version}`);
  console.log(`  Model: ${result.model}`);
  console.log(`  Dimensions: ${result.dimensions}`);
  console.log(`  Documents: ${result.documents}`);
  console.log(`  Chunks: ${result.chunks}`);

  return result;
}

/**
 * Parse command line arguments
 */
function parseArgs(args: string[]): { command: string; options: Record<string, unknown> } {
  const command = args[0] || 'help';
  const options: Record<string, unknown> = {};

  for (let i = 1; i < args.length; i++) {
    const arg = args[i];

    if (arg.startsWith('--')) {
      const key = arg.slice(2);
      const next = args[i + 1];

      if (next && !next.startsWith('--')) {
        options[key] = next;
        i++;
      } else {
        options[key] = true;
      }
    } else if (!options.positional) {
      options.positional = arg;
    }
  }

  return { command, options };
}

/**
 * Print usage information
 */
function printUsage(): void {
  console.log(`
rag.js CLI - Build precompiled vector indices

Usage:
  ragjs build [options]     Build an index from documents
  ragjs info <index>        Show information about an index
  ragjs help                Show this help message

Build Options:
  --input <path>        Input file or directory (required)
  --output <path>       Output index file (default: index.json)
  --format <type>       Output format: json or binary (default: json)
  --model <name>        Embedding model (default: Xenova/all-MiniLM-L6-v2)
  --chunk-size <n>      Chunk size in characters (default: 512)
  --chunk-overlap <n>   Chunk overlap in characters (default: 50)
  --extensions <exts>   File extensions to process (default: .md,.txt,.html)
  --verbose             Show detailed progress

Examples:
  ragjs build --input ./docs --output ./dist/index.json
  ragjs build --input ./content --output ./index.bin --format binary --verbose
  ragjs info ./index.json
`);
}

/**
 * Main entry point
 */
async function main(): Promise<void> {
  const { command, options } = parseArgs(process.argv.slice(2));

  switch (command) {
    case 'build': {
      if (!options.input) {
        console.error('Error: --input is required');
        printUsage();
        process.exit(1);
      }

      const extensions = (options.extensions as string || '.md,.txt,.html')
        .split(',')
        .map(e => e.startsWith('.') ? e : `.${e}`);

      await build({
        input: options.input as string,
        output: (options.output as string) || 'index.json',
        format: (options.format as 'json' | 'binary') || 'json',
        model: (options.model as string) || 'Xenova/all-MiniLM-L6-v2',
        chunkSize: parseInt(options['chunk-size'] as string) || 512,
        chunkOverlap: parseInt(options['chunk-overlap'] as string) || 50,
        extensions,
        verbose: !!options.verbose,
      });
      break;
    }

    case 'info': {
      const indexPath = options.positional as string;
      if (!indexPath) {
        console.error('Error: Index path is required');
        printUsage();
        process.exit(1);
      }
      info(indexPath);
      break;
    }

    case 'help':
    default:
      printUsage();
      break;
  }
}

main().catch(err => {
  console.error('Error:', err.message);
  process.exit(1);
});
