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

---

![image](https://github.com/tetrapod0/RAG_with_lm_studio/assets/48349693/3ed53b55-e4e8-4316-b827-a95e3978afd7)
