#MEGA 커피 키오스크: AI 기반 포용적 서비스 개발 프로젝트

1. 프로젝트 개요
본 프로젝트는 디지털 포용법에 의거하여 모든 연령대와 다양한 건강 상태를 가진 사용자가 쉽게 이용할 수 있는 AI 기반 커피 키오스크 시스템을 개발하였습니다. 특히 고령층과 디지털 소외계층을 위한 접근성 향상을 핵심 목표로 하여, 직관적인 인터페이스와 지능형 추천 시스템을 구현했습니다.
2. 핵심 기능
2.1 지능형 RAG 기반 응답 시스템

메뉴 정보를 벡터화하여 사용자 질문에 맥락 기반 응답 제공
대화 이력 관리를 통한 자연스러운 대화 흐름 구현
특수 질문 패턴 인식(온도, 맛, 비교 질문)

2.2 사용자 맞춤 추천 시스템

얼굴 인식을 통한 연령대 자동 추정
연령별 영양소 기준 맞춤 메뉴 필터링
질병 정보 기반 건강 맞춤 메뉴 추천(당뇨, 고혈압, 심장질환 등)

2.3 포용적 사용자 인터페이스

큰 글씨와 명확한 색상 대비의 UI 요소
TTS(Text-to-Speech) 지원으로 시각 장애인 접근성 향상
단순화된 메뉴 선택 프로세스

3. 기술 스택

프론트엔드: PyQt5
백엔드: Python
AI 모델:

텍스트 임베딩: SentenceTransformer
얼굴 인식: DeepFace
텍스트 생성: Google Gemini API


검색 엔진: FAISS 벡터 검색
데이터 처리: Pandas, NumPy
TTS 엔진: gTTS, Pygame

4. 주요 구현 내용
4.1 RAG(Retrieval-Augmented Generation) 파이프라인
pythondef rag_pipeline(query, context_chunks, embedder, index, top_k=8, menu_df=None, conversation_history=None):
    # 메뉴 이름과 HOT/ICE 여부 감지
    detected_menus, temp_type, is_comparison, is_sweet_question = identify_menu_type(query, menu_df)
    
    # 대화 이력 및 특수 패턴 처리
    # 벡터 검색 및 컨텍스트 추출
    # LLM으로 응답 생성
    # 응답 전처리 및 로깅
4.2 영양소 기반 건강 필터링
pythondef filter_menu_by_health(menu_df, age_group=None, diseases=None):
    # 연령대별 필터링
    if age_normalized == "어린이":  # 0~12세
        filtered_df = filtered_df[filtered_df["단백질(g)"].fillna(0) >= 1]
    elif age_normalized == "청소년":  # 13~18세
        filtered_df = filtered_df[filtered_df["단백질(g)"].fillna(0) >= 0.8]
    
    # 질환별 필터링
    if "당뇨병" in diseases:
        filtered_df = filtered_df[filtered_df["당류(g)"].fillna(999) <= 10]
    if "고혈압" in diseases:
        filtered_df = filtered_df[filtered_df["나트륨(mg)"].fillna(999) <= 100]
4.3 TTS 음성 변환 기능
pythondef text_to_speech(text, lang='ko'):
    # TTS용 텍스트 정제
    clean_text = clean_text_for_tts(text)
    
    # 음성 파일 생성 및 재생
    tts = gTTS(text=clean_text, lang=lang)
    tts.save(filename)
    
    pygame.mixer.init()
    pygame.mixer.music.load(filename)
    pygame.mixer.music.play()
4.4 지속적 학습 시스템
pythondef save_rag_response_log(query, answer, response_path="rag_log.txt"):
    # 로그 파일에 질문과 답변 저장
    with open(response_path, "a", encoding="utf-8") as f:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"[시간] {timestamp}\n[질문] {query}\n[답변] {answer}\n\n")
    
    # 주기적으로 로그 파일에서 데이터 학습
    if os.path.getsize(response_path) > 10000:  # 10KB 이상
        learn_from_log(response_path)
5. 트러블슈팅 및 개선사항
5.1 메뉴 정보 처리 최적화

문제: CSV 파일에서 불필요한 열까지 모두 로드되어 메뉴 정보가 복잡해짐
해결: 필요한 열만 선택적으로 로드하고, 인덱스 번호 자동 제거 로직 추가

5.2 TTS 출력 개선

문제: 마크다운 기호와 포맷팅이 음성으로 부자연스럽게 읽힘
해결: 정규식을 사용한 TTS용 텍스트 정제 함수 구현

5.3 메뉴 비교 기능 개선

문제: "어떤 게 더 달아요?" 등의 비교 질문 처리 실패
해결: 메뉴 정보 비교 로직 추가 및 당류 데이터 처리 개선

5.4 대화 맥락 관리 개선

문제: 이전 질문-응답 맥락을 기억하지 못함
해결: 대화 이력 관리 로직 추가 및 conversation_history 구조체 도입

6. 사용 방법

필요한 패키지 설치

bashpip install -r requirements.txt

메뉴 데이터 준비 (CSV 파일)

# process_data.csv 형식
카테고리번호,HOT/ICE,분류,가격,이름,칼로리(kcal),탄수화물(g),당류(g),단백질(g),...

프로그램 실행

bashpython mega_kiosk_ui.py
7. 향후 개선 방향

다국어 지원을 통한 외국인 접근성 향상
알레르기 정보 기반 필터링 시스템 추가
사용자 피드백 기반 추천 알고리즘 고도화
온라인 주문 시스템 연동

8. 팀원 및 기여

홍대길: 데이터 조사 및 처리, 얼굴 인식 시스템 구현
박지원: 기획 및 프로젝트 관리, RAG 시스템 구현, UI 설계

라이센스
이 프로젝트는 MIT 라이센스 하에 배포됩니다.

이 프로젝트는 디지털 포용성을 높이기 위한 연구 목적으로 개발되었으며, 메가 커피와의 공식 제휴 관계는 없습니다.재시도Claude는 실수를 할 수 있습니다. 응답을 반드시 다시 확인해 주세요.