import Database from 'better-sqlite3';
import * as sqliteVec from 'sqlite-vec';
import type { Chunk } from './chunker.js';
import type { StoredChunk } from './store.js';

export class SqliteVectorStore {
  private db: Database.Database;

  constructor(dbPath: string = 'knowledge.db') {
    this.db = new Database(dbPath);
    sqliteVec.load(this.db);       // 加载向量搜索扩展
    this.createTables();
  }

  private createTables() {
    this.db.exec(`
      CREATE TABLE IF NOT EXISTS chunks (
        id TEXT PRIMARY KEY,
        text TEXT NOT NULL,
        source TEXT NOT NULL,
        chunk_index INTEGER NOT NULL,
        embedding TEXT NOT NULL,
        model TEXT NOT NULL DEFAULT 'text-embedding-v3',
        updated_at INTEGER NOT NULL
      );

      CREATE VIRTUAL TABLE IF NOT EXISTS chunks_vec USING vec0(
        id TEXT PRIMARY KEY,
        embedding FLOAT[128]
      );

      CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
        text, id UNINDEXED, source UNINDEXED
      );
    `);
  }

  add(chunk: Chunk, embedding: number[]): void {
    const now = Date.now();
    // 三表联动写入
    this.db.prepare(`INSERT OR REPLACE INTO chunks
      (id, text, source, chunk_index, embedding, updated_at)
      VALUES (?, ?, ?, ?, ?, ?)`)
      .run(chunk.id, chunk.text, chunk.source, chunk.index,
           JSON.stringify(embedding), now);

    this.db.prepare(`INSERT OR REPLACE INTO chunks_vec (id, embedding)
      VALUES (?, ?)`)
      .run(chunk.id, Buffer.from(new Float32Array(embedding).buffer));

    this.db.prepare(`INSERT OR REPLACE INTO chunks_fts (id, text, source)
      VALUES (?, ?, ?)`)
      .run(chunk.id, chunk.text, chunk.source);
  }

  addBatch(items: Array<{ chunk: Chunk; embedding: number[] }>): void {
    const tx = this.db.transaction(() => {
      for (const { chunk, embedding } of items) this.add(chunk, embedding);
    });
    tx();  // 事务批量写入，比逐条快很多
  }

  vectorSearch(queryEmbedding: number[], topK: number): Array<{ chunk: StoredChunk; score: number }> {
    const buf = Buffer.from(new Float32Array(queryEmbedding).buffer);
    const rows = this.db.prepare(`
      SELECT v.id, v.distance, c.text, c.source, c.chunk_index, c.embedding
      FROM chunks_vec v
      JOIN chunks c ON c.id = v.id
      WHERE v.embedding MATCH ?
      ORDER BY v.distance
      LIMIT ?
    `).all(buf, topK) as any[];

    return rows.map(r => ({
      chunk: {
        id: r.id, text: r.text, source: r.source,
        index: r.chunk_index,
        tokenEstimate: Math.ceil(r.text.length * 0.6),
        embedding: JSON.parse(r.embedding),
        addedAt: 0,
      },
      score: 1 - r.distance,  // cosine distance → similarity
    }));
  }

  keywordSearch(query: string, topK: number): Array<{ chunk: StoredChunk; score: number }> {
    const rows = this.db.prepare(`
      SELECT f.id, bm25(chunks_fts) AS rank, c.text, c.source, c.chunk_index, c.embedding
      FROM chunks_fts f
      JOIN chunks c ON c.id = f.id
      WHERE chunks_fts MATCH ?
      ORDER BY rank
      LIMIT ?
    `).all(query, topK) as any[];

    return rows.map(r => ({
      chunk: {
        id: r.id, text: r.text, source: r.source,
        index: r.chunk_index,
        tokenEstimate: Math.ceil(r.text.length * 0.6),
        embedding: JSON.parse(r.embedding),
        addedAt: 0,
      },
      score: r.rank < 0 ? -r.rank / (1 - r.rank) : 1 / (1 + r.rank),
    }));
  }

  size(): number {
    return (this.db.prepare('SELECT COUNT(*) as n FROM chunks').get() as any).n;
  }

  sources(): string[] {
    return (this.db.prepare('SELECT DISTINCT source FROM chunks').all() as any[]).map(r => r.source);
  }

  getAll(): Array<StoredChunk> {
    const rows = this.db.prepare('SELECT * FROM chunks').all() as any[];
    return rows.map(r => ({
      id: r.id,
      text: r.text,
      source: r.source,
      index: r.chunk_index,
      tokenEstimate: Math.ceil(r.text.length * 0.6),
      embedding: JSON.parse(r.embedding),
      addedAt: r.updated_at,
    }));
  }
}
