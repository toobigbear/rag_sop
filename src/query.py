"""RAG 检索 + 生成：给定问题，检索 top-k 知识块，注入 prompt 调用 Ollama。"""
from pathlib import Path
import chromadb
import ollama

BASE = Path(__file__).resolve().parent.parent
COLLECTION = "sop_kb"
MODEL = "qwen2.5:14b"   # 你可按需换 14b
SYSTEM = (
    "你是一名危急重症心功能异常智能预警系统的数据分析助手。"
    "请基于给定的知识片段，用专业但通俗的中文回答；"
    "知识片段未覆盖的内容，请明确说明'根据现有知识库无法确认'，不要臆造。"
)

def retrieve(query: str, top_k: int = 3):
    client = chromadb.PersistentClient(path=str(BASE / "data" / "chroma"))
    col = client.get_collection(COLLECTION)
    r = ollama.embeddings(model="nomic-embed-text", prompt=query)
    res = col.query(query_embeddings=[r["embedding"]], n_results=top_k)
    docs = res["documents"][0]
    metas = res["metadatas"][0]
    return [f"[来源:{m['source']}]\n{d}" for d, m in zip(docs, metas)]

def retrieve_context(query: str, top_k: int = 3) -> str:
    """只做检索：返回知识纯文本；库为空/不可用/异常一律返回空串（不抛异常）。"""
    try:
        chroma_dir = BASE / "data" / "chroma"
        if not chroma_dir.exists():            # 库目录不存在 => 没建库
            return ""
        client = chromadb.PersistentClient(path=str(chroma_dir))
        try:
            col = client.get_collection(COLLECTION)
        except Exception:
            return ""                          # 集合不存在 => 空库
        if col.count() == 0:                   # 集合存在但空
            return ""
        emb = ollama.embeddings(model="nomic-embed-text", prompt=query)
        res = col.query(query_embeddings=[emb["embedding"]], n_results=top_k)
        docs, metas = res["documents"][0], res["metadatas"][0]
        if not docs:
            return ""
        return "\n\n".join(f"[来源:{m['source']}]\n{d}" for d, m in zip(docs, metas))
    except Exception as e:
        print(f"[rag] 检索失败，本轮不注入知识：{e}")
        return ""

def answer(query: str, top_k: int = 3):
    context = "\n\n".join(retrieve(query, top_k))
    prompt = (
        f"以下是相关知识片段：\n<context>\n{context}\n</context>\n\n"
        f"问题：{query}\n\n请结合知识片段回答。"
    )
    resp = ollama.chat(
        model=MODEL,
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": prompt}],
        options={"temperature": 0.2},
    )
    return resp["message"]["content"]

if __name__ == "__main__":
    q = input("请输入问题：") or "SOFA评分怎么算？NT-proBNP升高提示什么？"
    print("\n===== 回答 =====\n")
    print(answer(q))
