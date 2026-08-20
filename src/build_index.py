"""RAG 建库：把 knowledge/ 下的 md 语料切块、向量化、写入 Chroma。"""
from pathlib import Path
import chromadb
import ollama

BASE = Path(__file__).resolve().parent.parent
KNOWLEDGE = BASE / "data" / "knowledge"
COLLECTION = "sop_kb"

def load_docs():
    """读所有 .md，按标题/段落切 chunk。"""
    docs = []
    for f in sorted(KNOWLEDGE.glob("*.md")):
        text = f.read_text(encoding="utf-8")
        # 简单切块：按空行分段，合并成 ~500字 的 chunk
        chunks, cur = [], ""
        for para in text.split("\n\n"):
            para = para.strip()
            if not para:
                continue
            if len(cur) + len(para) > 500 and cur:
                chunks.append(cur)
                cur = para
            else:
                cur = f"{cur}\n{para}"
        if cur:
            chunks.append(cur)
        for i, c in enumerate(chunks):
            docs.append({"source": f.name, "chunk": i, "text": c})
    return docs

def embed(texts: list[str]) -> list[list[float]]:
    """用 Ollama 内置 embedding 模型向量化。首次需：ollama pull nomic-embed-text"""
    out = []
    for t in texts:
        r = ollama.embeddings(model="nomic-embed-text", prompt=t)
        out.append(r["embedding"])
    return out

def build():
    docs = load_docs()
    print(f"[build] 共 {len(docs)} 个 chunk")
    texts = [d["text"] for d in docs]
    vecs = embed(texts)

    client = chromadb.PersistentClient(path=str(BASE / "data" / "chroma"))
    col = client.get_or_create_collection(COLLECTION)
    col.upsert(
        ids=[f"{d['source']}#{d['chunk']}" for d in docs],
        embeddings=vecs,
        documents=texts,
        metadatas=[{"source": d["source"]} for d in docs],
    )
    print(f"[build] 已写入 {col.count()} 条向量 -> data/chroma")

if __name__ == "__main__":
    build()
