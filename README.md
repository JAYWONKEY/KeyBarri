# ☕ MEGA 커피 키오스크: 디지털 취약계층을 위한 RAG 기반 키오스크 도우미


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
```

---

### 2. 영양소 기반 건강 필터링

```python
def filter_menu_by_health(menu_df, age_group=None, diseases=None):
    # 연령대별 필터링
    if age_normalized == "어린이":
        filtered_df = filtered_df[filtered_df["단백질(g)"].fillna(0) >= 1]
    elif age_normalized == "청소년":
        filtered_df = filtered_df[filtered_df["단백질(g)"].fillna(0) >= 0.8]

    # 질환별 필터링
    if "당뇨병" in diseases:
        filtered_df = filtered_df[filtered_df["당류(g)"].fillna(999) <= 10]
    if "고혈압" in diseases:
        filtered_df = filtered_df[filtered_df["나트륨(mg)"].fillna(999) <= 100]
```

---

### 3. TTS 음성 변환 기능

```python
def text_to_speech(text, lang='ko'):
    clean_text = clean_text_for_tts(text)
    tts = gTTS(text=clean_text, lang=lang)
    tts.save(filename)

    pygame.mixer.init()
    pygame.mixer.music.load(filename)
    pygame.mixer.music.play()
```

---

### 4. 지속적 학습 시스템

```python
def save_rag_response_log(query, answer, response_path="rag_log.txt"):
    with open(response_path, "a", encoding="utf-8") as f:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"[시간] {timestamp}
[질문] {query}
[답변] {answer}

")

    if os.path.getsize(response_path) > 10000:
        learn_from_log(response_path)
```

---

## ⚙️ 트러블슈팅 및 개선사항

### 1. 메뉴 정보 처리 최적화
- **문제**: CSV에 불필요한 열이 포함되어 복잡도 증가  
- **해결**: 필요한 열만 선택, 인덱스 자동 제거 로직 추가

### 2. TTS 출력 개선
- **문제**: 마크다운 기호로 인한 부자연스러운 발음  
- **해결**: 정규식을 활용한 TTS용 텍스트 정제

### 3. 메뉴 비교 기능 개선
- **문제**: "어떤 게 더 달아요?" 질문에 실패  
- **해결**: 당류 기반 비교 로직 추가

### 4. 대화 맥락 관리 개선
- **문제**: 맥락을 기억하지 못함  
- **해결**: `conversation_history` 구조체 추가 및 관리 로직 구현

---

## 🧑‍💻 사용 방법

### 1. 필요한 패키지 설치

```bash
pip install -r requirements.txt
```

### 2. 메뉴 데이터 준비

`process_data.csv` 형식 예시:

```
카테고리번호,HOT/ICE,분류,가격,이름,칼로리(kcal),탄수화물(g),당류(g),단백질(g),...
```

### 3. 프로그램 실행

```bash
python mega_kiosk_ui.py
```

---

## 🚀 향후 개선 방향

- 다국어 지원을 통한 **외국인 접근성 향상**
- **알레르기 정보 기반 필터링 시스템** 추가
- **사용자 피드백 기반 추천 알고리즘** 고도화
- **온라인 주문 시스템** 연동

---

## 👨‍👩‍👧‍👦 팀원 및 기여

- **홍대길**: 데이터 조사 및 처리, 얼굴 인식 시스템 구현  
- **박지원**: 기획 및 프로젝트 관리, RAG 시스템 구현, UI 설계

---

## 📄 라이센스

본 프로젝트는 **MIT License** 하에 배포됩니다.  
이 프로젝트는 **디지털 포용성 향상**을 위한 연구 목적으로 개발되었으며, **메가커피와는 공식적인 제휴 관계가 없습니다.**
