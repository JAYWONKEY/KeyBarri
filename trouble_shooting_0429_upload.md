# 🛠️ Trouble Shooting Log - 2025.04.29

## 🎯 이슈 요약

**Prompt/TTS 출력 개선, CSV 파싱 오류, 메뉴 데이터 활용, 맥락 처리 등 여러 가지 개선 필요**

---

## Trouble Shooting 1: TTS 및 프롬프트 출력 정리

### 1.1 TTS 출력 개선

- 문제: LLM 응답 그대로 TTS 변환 → 마크다운 기호(`**`)도 음성으로 읽힘  
- 해결:

```python
def clean_text_for_tts(text):
    """TTS용 텍스트 정제 함수"""
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    text = re.sub(r'참고:', '참고 사항으로', text)
    text = re.sub(r'[\[\]\(\)\{\}]', '', text)
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'\n', '. ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()
```


1.2 CSV 파싱 개선
문제: load_csv_data()에서 전체 열을 무작정 붙여 사용
```
    # 번호 컬럼 자동 제거
    if df.columns[0].lower() in ['no', 'index', 'id', '번호'] or df.iloc[:, 0].astype(str).str.match(r'^\d+$').all():
        df = df.iloc[:, 1:]
```

1.3 정규식 개선
설명을 제거하기 위한 패턴 정리:
```
text = re.sub(r'\(.*?\)', '', text)
text = re.sub(r'\[.*?\]', '', text)
```

## Trouble Shooting 2: 허브티 응답 및 맥락 처리

### 문제: 메뉴 종류 부족 및 대화 맥락 반영 불가

    ```
    #### 해결 아이디어:

    CSV에 인기도, 추천 연령, 추천 질환 필드 추가

    나이 기반 필터링 로직 도입

    예시 연령 구간:

    0–12세: 성장기 (단백질/칼슘)

    30–49세: 항산화 (비타민 C/E)

    65세 이상: 근력 (단백질/오메가3)

    ```

##  Trouble Shooting 3: 프롬프트 개선 및 반복 응답 문제

### 문제:

   - 프롬프트가 메뉴명 생성 유도를 못함

   - 인기도 정보가 없어 매번 동일한 메뉴 추천

### 해결:

   -  CSV에 인기도 필드 추가

   - 프롬프트에서 2~3개 인기 메뉴만 소개하도록 유도
    ```
        prompt = f"""
        다음 데이터를 바탕으로 질문에 답변해주세요.

        데이터:
        {' '.join(context)}

        질문: {query}
        답변은 자연스럽고 간결하게 해주세요.
        특수문자, 마크다운 기호 없이 TTS에 최적화된 말투로 작성해주세요.
        """
    ```

### 포용적 UX 요소

    ```
    항목	설명
    글자 크기/명암 대비	고령층 배려 18pt 이상, 명확한 배경
    음성 안내	모든 텍스트 음성지원 버튼 제공
    대형 버튼 UI	시니어/지체장애인 위한 큰 터치 영역
    간단한 추천	"추천 받기" 버튼으로 메뉴 2~3개 제안

    ```

 