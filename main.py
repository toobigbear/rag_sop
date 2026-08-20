"""python main.py build | query | retrieve"""
import sys
from src.build_index import build
from src.query import answer, retrieve_context

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "query"
    if cmd == "build":
        build()
    elif cmd == "retrieve":
        q = sys.argv[2] if len(sys.argv) > 2 else "SOFA评分怎么算？"
        print(retrieve_context(q))
    elif cmd == "query":
        q = sys.argv[2] if len(sys.argv) > 2 else "SOFA评分怎么算？"
        print(answer(q))
    else:
        print("用法: python main.py build | query [问题] | retrieve [问题]")