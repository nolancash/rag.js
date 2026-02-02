// Type declarations for @xenova/transformers (optional peer dependency)
declare module '@xenova/transformers' {
  export function pipeline(
    task: string,
    model: string,
    options?: { quantized?: boolean }
  ): Promise<(text: string, options?: { pooling?: string; normalize?: boolean }) => Promise<{ data: Float32Array }>>;
}
