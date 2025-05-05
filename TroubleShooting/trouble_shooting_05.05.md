문제 발생 사항

"too many values to unpack" 오류

함수가 4개의 값을 반환하는데 2개만 받으려 해서 발생


"아메리카노와 카페라떼 중 어떤 게 더 달아요?" 질문 처리 실패

메뉴 데이터는 정상 감지되었으나 답변이 엉뚱하게 나옴



원인 분석
1) identify_menu_type 함수 수정 사항
python# 기존 (2개 반환)
return menu_name, temp_type

# 수정 (4개 반환)  
return detected_menus, temp_type, is_comparison, is_sweet_question
2) rag_pipeline 함수에서 값을 받는 방식 불일치
python# 오류 발생 코드
menu_name, temp_type = identify_menu_type(query, menu_df)

# 수정된 코드
detected_menus, temp_type, is_comparison, is_sweet_question = identify_menu_type(query, menu_df)
3) 비교 로직 실행 순서 문제

비교 로직이 LLM 프롬프트 로직 뒤에 위치
LLM이 먼저 응답을 생성하여 비교 로직이 실행되지 않음

해결 방법
1단계: 함수 반환값 일치

rag_pipeline의 모든 identify_menu_type 호출 부분을 4개 변수로 수정
기존 menu_name → detected_menus로 변수명 변경

2단계: 로직 실행 순서 조정

비교 질문 처리 로직을 LLM 프롬프트 로직 앞으로 이동
특정 조건(비교 + 단맛 질문)이 맞을 때 먼저 실행

3단계: 데이터 처리 강화
python# 당류 데이터 없을 때 처리
if sugar is None or pd.isna(sugar):
    sugar = 0

# 값이 같을 때 처리    
if max_sugar == min_sugar:
    if max_sugar == 0:
        return "당류 정보가 없어서 정확한 비교가 어렵습니다"
결과

too many values to unpack 오류 해결
메뉴 비교 질문에 대한 정확한 답변 구현
데이터가 없거나 같은 경우에 대한 안정적인 처리

핵심 교훈

함수 시그니처 변경 시 모든 호출부 수정 필수
조건별 로직 실행 순서 중요 (우선순위 있는 로직 먼저)
예외 상황에 대한 안정적인 처리 필요