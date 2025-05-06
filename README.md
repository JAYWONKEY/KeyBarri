# ☕ MEGA 커피 키오스크: AI 기반 포용적 서비스 개발 프로젝트

## 📌 프로젝트 개요

본 프로젝트는 **디지털 포용법**에 의거하여, 모든 연령대와 다양한 건강 상태를 가진 사용자가 쉽게 이용할 수 있는 **AI 기반 커피 키오스크 시스템**을 개발하였습니다.  
특히 **고령층**과 **디지털 소외계층**을 위한 접근성 향상을 핵심 목표로 하여, **직관적인 인터페이스**와 **지능형 추천 시스템**을 구현했습니다.

---

## 💡 핵심 기능

### 1. 지능형 RAG 기반 응답 시스템
- 메뉴 정보를 벡터화하여 사용자 질문에 **맥락 기반 응답** 제공
- **대화 이력 관리**를 통한 자연스러운 흐름 구현
- 온도/맛/비교 등 **특수 질문 패턴 인식**

### 2. 사용자 맞춤 추천 시스템
- **얼굴 인식**을 통한 연령대 자동 추정
- 연령별 영양소 기준에 따른 **맞춤 메뉴 필터링**
- **질병 정보 기반** 건강 맞춤 메뉴 추천  
  (예: 당뇨, 고혈압, 심장질환 등)

### 3. 포용적 사용자 인터페이스
- **큰 글씨**, **명확한 색상 대비**
- **TTS(Text-to-Speech)**로 시각장애인 접근성 향상
- **단순화된 메뉴 선택 프로세스**

---

## 🛠 기술 스택

- **프론트엔드**: PyQt5  
- **백엔드**: Python  
- **AI 모델**
  - 텍스트 임베딩: `SentenceTransformer`
  - 얼굴 인식: `DeepFace`
  - 텍스트 생성: `Google Gemini API`
- **검색 엔진**: `FAISS`  
- **데이터 처리**: `Pandas`, `NumPy`  
- **TTS 엔진**: `gTTS`, `Pygame`

---

## 🔍 주요 구현 내용

### 1. RAG(Retrieval-Augmented Generation) 파이프라인

```python
def rag_pipeline(query, context_chunks, embedder, index, top_k=8, menu_df=None, conversation_history=None):
    detected_menus, temp_type, is_comparison, is_sweet_question = identify_menu_type(query, menu_df)
    # 대화 이력 및 특수 패턴 처리
    # 벡터 검색 및 컨텍스트 추출
    # LLM으로 응답 생성
    # 응답 전처리 및 로깅
