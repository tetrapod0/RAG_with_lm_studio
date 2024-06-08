# RAG_with_lm_studio

---

### RAG 란?

- RAG는 Retrieval Augmented Generation의 약자로, 지식 검색과 언어 생성 모델을 결합한 형태로 직역하면 검색 증강 생성이다.

- RAG 모델은 다음과 같이 동작한다.
  1. 질문을 입력받음
  2. 질문을 임베딩하여 벡터로 표현
  3. 사전에 벡터저장소에 저장된 문서 벡터들과 질문 벡터 간의 유사도를 계산
  4. 유사도가 높은 상위 k개의 문서를 검색
  5. 검색된 관련 문장들과 원래 질문을 템플릿에 삽입하여 프롬프트를 완성
  5. 프롬프트를 LLM에 넣어 최종 답변 생성

- 이를 통해 기존 언어모델의 지식 부족 문제를 보완할 수 있다.

### RAG 과정

1. 문서의 청크 단위를 임베딩 모델로 벡터화한다.
2. 벡터화된 문서를 벡터 저장소(FAISS 등)에 저장한다.
3. 사용자의 질문 문자열을 임베딩 모델로 벡터화 한다.
4. 벡터화된 질문 문자열을 벡터 저장소에서 유사도 검색한다.
5. 검색된 k개의 문서 청크 문자열을 프롬프트에 삽입한다.
6. 사용자의 질문 문자열을 프롬프트에 삽입한다.
7. 완성된 프롬프트를 LLM에 전달하여 답변을 받는다.

![image](https://github.com/tetrapod0/RAG_with_lm_studio/assets/48349693/3ed53b55-e4e8-4316-b827-a95e3978afd7)

---

### LM-Studio의 LLM에 연결하는 예시 코드

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio",
    model="lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF",
    temperature=0.1,
)
```

### LM-Studio의 Embedding model에 연결하는 예시 코드

```python
from langchain_core.embeddings import Embeddings
from openai import OpenAI
from typing import List

class MyEmbeddings(Embeddings):
    def __init__(self, base_url, api_key="lm-studio"):
        self.client = OpenAI(base_url=base_url, api_key=api_key)

    def embed_documents(self, texts: List[str], model="nomic-ai/nomic-embed-text-v1.5-GGUF") -> List[List[float]]:
        texts = list(map(lambda text:text.replace("\n", " "), texts))
        datas = self.client.embeddings.create(input=texts, model=model).data
        return list(map(lambda data:data.embedding, datas))
        
    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]

emb_model = MyEmbeddings(base_url="http://localhost:1234/v1")
```

---

### RAG 웹 챗봇 구성

- 웹 API : Streamlit
- LLM : lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF
- Embedding Model : nomic-ai/nomic-embed-text-v1.5-GGUF
- LLM, Embedding Model로 LM-Studio 로컬서버만 사용
- 이 외에 모델 다운로드, 프롬프트 다운로드 등등 없음
- TXT, PDF만 파일 업로드 지원
- 문서 청크사이즈 : 200, overlap : 50
- 벡터저장소 : FAISS, top_k : 10
- RAG용 체인, 대화용 체인

---

### 제작한 웹 페이지 리뷰

- streamlit run server.py로 실행후 처음 화면이다.

![image](https://github.com/tetrapod0/RAG_with_lm_studio/assets/48349693/bd7602db-b793-4573-ad51-f9a971424581)

- 느낌표를 앞에 붙이고 질문하면 RAG용 체인으로 작동하여 검색하여 답변한다.
- 여기서 RAG용 체인은 이전대화 포함하지 않고 입력한다.

![image](https://github.com/tetrapod0/RAG_with_lm_studio/assets/48349693/b9061c25-5de4-4f4b-bf42-5ed358a7c737)

- 아래는 LM Studio 서버의 로그이다.

![image](https://github.com/tetrapod0/RAG_with_lm_studio/assets/48349693/e9af6ada-87f1-48a8-b411-fdfa61ce8d17)

- 느낌표를 붙이지 않으면 대화용 체인으로 작동하여 이전대화 기록으로 포함하여 입력한다.
- 이전 RAG 답변 기록도 저장되어있기 때문에 관련 내용을 기억한다.

![image](https://github.com/tetrapod0/RAG_with_lm_studio/assets/48349693/daf821bb-de0d-4a5d-aeda-0aa7af757475)

- 아래는 LM Studio 서버의 로그이다.

![image](https://github.com/tetrapod0/RAG_with_lm_studio/assets/48349693/ad77dfcc-0e11-4bda-b2f2-18abaeb272af)

---

### Reference

- https://velog.io/@kwon0koang/%EB%A1%9C%EC%BB%AC%EC%97%90%EC%84%9C-Llama3-%EB%8F%8C%EB%A6%AC%EA%B8%B0
- https://www.gpters.org/c/llm/embedding-vector-stores
- https://github.com/teddylee777/langserve_ollama/blob/main/example/main.py
- https://docs.streamlit.io/
- https://velog.io/@judy_choi/Steamlit-%EC%98%88%EC%A0%9C
- https://velog.io/@tetrapod0/LM-Studio-RAG-%ED%95%B4%EB%B3%B4%EA%B8%B0-2




