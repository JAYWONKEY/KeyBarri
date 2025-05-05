import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from dotenv import load_dotenv
import time
import uuid
import pygame
from gtts import gTTS
import re
# 25.04.30일 추가 
import fitz  # PyMuPDF
import os

# 25.05.01 터미널 에러.txt 추가 
import traceback 
from datetime import datetime

def log_error(err_msg):
    with open("err_log.txt", "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now()}] 에러 발생: \n{err_msg}\n\n")


# 25.04.29 기능 추가 : def clean_text_for_tts(text)
def clean_text_for_tts(text):
        """TTS를 위한 텍스트 정제 함수"""
        # 마크다운 굵은 글씨 표시 제거
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
        # 마크다운 이탤릭체 표시 제거
        text = re.sub(r'\*(.*?)\*', r'\1', text)
        # 불필요한 기호 및 텍스트 제거
        text = re.sub(r'참고:', '참고 사항으로', text)
        text = re.sub(r'[\[\]\(\)\{\}]', '', text)
        # 여러 줄바꿈을 하나로 통일
        text = re.sub(r'\n+', '\n', text)
        # 줄바꿈을 자연스러운 문장 구분으로 변환
        text = re.sub(r'\n', '. ', text)
        # 불필요한 공백 제거
        text = re.sub(r'\s+', ' ', text)
        # 숫자 + "원"을 자연스럽게 읽기 
        text = re.sub(r'(\d+)원', r'\1 원', text)  
        # 중복 문장부호 정리
        text = re.sub(r'\.\.+', '.', text) # .. -> .
        text = re.sub(r'[!?.]+\.', '.', text) # !. or ?. -> .
        text = re.sub(r'\s*\.\s*', '. ', text) # 마침표 주변 공백 정리
        return text.strip()

# 25.04.30 def text_to_speech(text, lang='ko'): -> pygmae 재생 : mp3 저장 기능 옵션
def text_to_speech(text, lang='ko'):
    """텍스트를 음성으로 변환하고 재생하는 함수"""
    try:
        # TTS용 텍스트 정제 
        clean_text = clean_text_for_tts(text)
        print(f"\n[TTS용 정제된 텍스트]\n{clean_text}")
        
        filename = f"speech_{str(uuid.uuid4())[:8]}.mp3"
        # 25.04.30 text -> clean_text 수정
        tts = gTTS(text=clean_text, lang=lang)
        tts.save(filename)
        
        pygame.mixer.init()
        pygame.mixer.music.load(filename)
        pygame.mixer.music.play()
        
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)
            
        pygame.mixer.quit()
        os.remove(filename)
        
    except Exception as e:
        print(f"TTS 에러: {str(e)}")
        if os.path.exists(filename):
            try:
                os.remove(filename)
            except:
                pass

csv_path=r"process_data.csv"

# 25.04.29 기능 추가 : load_csv_data 내 인덱스 번호 제거 
def load_csv_data(csv_path):
    """CSV, Excel, PDF 파일 지원"""
    encodings = ['cp949', 'euc-kr', 'utf-8']
    
    for encoding in encodings:
        try:
            print(f"{encoding} 인코딩으로 CSV 파일 읽기 시도...")
            df = pd.read_csv(csv_path, encoding=encoding)

            # 🔧 컬럼명 공백 제거 25.05.01 추가
            df.columns = df.columns.str.strip()

            # 25.04.29 기능 추가 : 첫 번째 컬럼이 번호면 제거
            if df.columns[0].lower() in ['no', 'index', 'id', '번호', 'id'] or df.iloc[:,0].astype(str).str.match(r'^\d+$').all():
                print("번호 컬럼 제거")
                df = df.iloc[:, 1:]

            # 25.04.29 기능 추가 : 필요한 열만 선택 -> 05.02 분류 내 디카페인메뉴 수정 및 코드내 '분류'추가 
            selected_cols = ['이름', '가격', '분류', 'HOT/ICE']
            df = df[selected_cols]

            # 텍스트 결합 
            # 25.05.02 분류정보 추가
            texts = df.apply(lambda row: f"{row['이름']} {row['가격']}원. 분류: {row['분류']}.", axis=1).tolist()
            print(f"CSV 파일 로드 성공: {len(texts)}개의 행")
            return texts
        
        except UnicodeDecodeError:
            continue
        except Exception as e:
            print(f"CSV 읽기 오류: {str(e)}")
            return None
    
    print("\nCSV 파일을 읽을 수 없습니다. 다음 방법 중 하나를 시도해보세요:")
    print("1. Excel에서:")
    print("   - '다른 이름으로 저장' 선택")
    print("   - 파일 형식을 'CSV (쉼표로 분리) (*.csv)' 선택")
    print("2. 메모장에서:")
    print("   - '다른 이름으로 저장' 선택")
    print("   - 인코딩을 'ANSI' 선택")
    return None, None

def load_data(any_path):
    """CSV, Excel, PDF 파일 지원"""
    ext = os.path.splitext(any_path)[-1].lower()

    if ext == '.csv':
        return load_csv_data(any_path)
    elif ext in ['.xls', '.xlsx']:
        try:
            df = pd.read_excel(any_path, engine='openpyxl')
            print("Excel 파일 로드 성공")
            return process_menu_dataframe(df)
        except Exception as e:
            print(f"Excel 파일 읽기 오류: {str(e)}")
            return None
    elif ext == '.pdf':
        try:
            text =""
            doc = fitz.open(any_path)
            for page in doc:
                text += page.get_text()
            print("PDF 파일 텍스트 추출 완료")
            return [text]
        except Exception as e:
            print(f"PDF 처리 오류: {str(e)}")
            return None
    else:
        print("지원하지 않는 파일 형식입니다.")
        return None
    
def process_menu_dataframe(df):
     """메뉴용 DataFrame을 처리하여 텍스트 리스트로 변환"""
     if df.columns[0].lower() in ['no', 'index', 'id', '번호'] or df.iloc[:, 0].astype(str).str.match(r'^\d+$').all():
         df = df.iloc[:, 1:]
     # 분류, HOT/ICE 추가 
     selected_cols = ['이름', '가격', '분류', 'HOT/ICE']
     df = df[selected_cols]
    
     # 텍스트 조합 분류 추가 
     texts = df.apply(lambda row: f"{row['이름']} {row['가격']}원. 분류: {row['분류']}.", axis=1).tolist()
     print(f"CSV 파일 로드 성공: {len(texts)}개의 행")
     return texts


def split_into_chunks(texts, chunk_size=1000):
    """텍스트를 청크로 분할"""
    chunks = []
    for text in texts:
        words = str(text).split()
        current_chunk = []
        current_size = 0
        
        for word in words:
            current_chunk.append(word)
            current_size += len(word) + 1
            if current_size >= chunk_size:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_size = 0
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
    
    return chunks

# 25.05.01일 기능 추가: 데이터 타입 변환 개선 코드 
def preprocess_dataframe(df):
    """
    데이터프레임 전처리 함수 - 데이터 타입 변환 및 정리
    """
    import pandas as pd
    import numpy as np
    
    # 컬럼명 공백 제거 및 정규화
    df.columns = df.columns.str.strip()
    
    # 숫자형 열 목록
    numeric_columns = [
        "단백질(g)", "당류(g)", "지방(g)", "포화지방(g)", "트랜스지방(g)",
        "나트륨(mg)", "콜레스테롤(mg)", "카페인(mg)", "칼로리(kcal)", "탄수화물(g)"
    ]
    
    # 숫자형 열 변환
    for col in numeric_columns:
        if col in df.columns:
            # 원본 값 백업
            original_values = df[col].copy()
            
            try:
                # 문자열로 변환 후 정제
                df[col] = df[col].astype(str).str.strip()
                # 콤마 제거
                df[col] = df[col].str.replace(',', '')
                # 숫자가 아닌 문자 제거 (소수점 유지)
                df[col] = df[col].str.replace(r'[^\d.]', '', regex=True)
                # 빈 문자열을 NaN으로 변환
                df[col] = df[col].replace('', np.nan)
                # 숫자로 변환 (오류 허용)
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # 변환 실패한 부분 확인 및 기록
                nan_count = df[col].isna().sum()
                if nan_count > 0:
                    print(f"경고: '{col}' 열에서 {nan_count}개의 값이 숫자로 변환되지 않았습니다.")
                    
            except Exception as e:
                print(f"'{col}' 열 변환 중 오류 발생: {str(e)}")
                # 오류 발생 시 원본 데이터 복원
                df[col] = original_values
    
    return df

# 25.04.30일 기능추가 : 사용자 연령대와 질환 정보를 기반으로 메뉴를 필터링
def filter_menu_by_health(menu_df, age_group=None, diseases=None):
    """
    사용자 연령대와 질환 정보를 기반으로 메뉴를 필터링 - 개선된 버전
    """
    try:
        filtered_df = menu_df.copy()
        # 25.05.01 숫자형 열을 float으로 변환
        numeric_columns = [
        "단백질(g)", "당류(g)", "지방(g)", "포화지방(g)", "트랜스지방(g)",
            "나트륨(mg)", "카페인(mg)"
        ]
        for col in numeric_columns:
            if col in filtered_df.columns:
                try:
                    filtered_df[col] = pd.to_numeric(filtered_df[col], errors='coerce')
                except Exception as e:
                    print(f"[경고] 열 '{col}'을 numeric으로 변환할 수 없습니다: {e}")

        # 25.05.01 질환에 따른 필터링 -> - NaN 값 처리 추가 .fillna(999)
        if diseases:
            if "당뇨병" in diseases:
                filtered_df = filtered_df[filtered_df["당류(g)"].fillna(999) <= 10]
            if "고혈압" in diseases:
                filtered_df = filtered_df[filtered_df["나트륨(mg)"].fillna(999) <= 100]
            if "심장질환" in diseases:
                filtered_df = filtered_df[(filtered_df["지방(g)"].fillna(999) <= 3) & (filtered_df["카페인(mg)"].fillna(999) <= 100)]
            if "고지혈증" in diseases:
                filtered_df =filtered_df[(filtered_df["포화지방(g)"].fillna(999) <= 1.5) & (filtered_df["지방(g)"].fillna(999) <= 3) & (filtered_df["트랜스지방(g)"].fillna(999) <= 0.1)]
            
         # 25.04.30일 기능추가 :  연령대에 따른 영양 고려 필터링 - NaN 값 처리 추가
        if age_group:
            # 25.05.01 추가수정 : 연령대 그룹 정규화
            age_normalized = age_group.lower() if isinstance(age_group, str) else None

            if age_normalized == "어린이": # 0 ~ 12
                filtered_df = filtered_df[filtered_df["단백질(g)"].fillna(0) >=1]
            elif age_normalized == "청소년": # 13 ~ 18
                filtered_df = filtered_df[filtered_df["단백질(g)"].fillna(0) >=0.8]

            # 성인 일 때 청년, 중년, 장년이 들어가서 kiosk 
            # 성인 그룹 처리 (청년, 중년, 장년이 모두 성인에 포함)
            elif age_normalized in ["청년", "중년", "장년", "성인"]: # age_normalized == "성인": 추가
                # 25.05.02 공통 성인 필터 추가
                filtered_df = filtered_df[filtered_df["단백질(g)"].fillna(0) >= 0.5]
               
                # 세부 연령대별 추가 필터
                if age_normalized == "성인" or age_normalized == "청년":
                    # 25.05.02 청년 또는 일반 성인일 경우 카페인 제한만 적용
                    filtered_df = filtered_df[filtered_df["카페인(mg)"].fillna(999) <= 200]
                elif age_normalized == "중년":
                    # 중년은 지방 제한 추가
                    filtered_df = filtered_df[(filtered_df["지방(g)"].fillna(999) <= 3) & (filtered_df["포화지방(g)"].fillna(999) <= 1.5)]
                elif age_normalized == "장년":
                    # 장년은 나트륨과 카페인 제한 추가
                    filtered_df = filtered_df[(filtered_df["나트륨(mg)"].fillna(999) <= 100) & (filtered_df["카페인(mg)"].fillna(999) <= 150)]
            elif age_normalized == "노인":
                # 노인은 단백질 높고 카페인 낮은 메뉴
                filtered_df = filtered_df[(filtered_df["단백질(g)"].fillna(0) >= 1.5) & (filtered_df["카페인(mg)"].fillna(999) <= 150)]

        # 25.05.01 추가 연령 컬럼이 있으면 해당 연령대 포함된 항목 필터링 (문자열 포함 검색) 
        if "연령" in filtered_df.columns and age_group:
            # 25.05.02 추가 : 성인 검색 시 청년, 중년, 장년 포함 처리 
            if age_normalized == "성인":
                adult_filter = (filtered_df["연령"].str.contains("성인", case=False, na=False) | 
                                filtered_df["연령"].str.contains("청년", case=False, na=False) | 
                                filtered_df["연령"].str.contains("중년", case=False, na=False) | 
                                filtered_df["연령"].str.contains("장년", case=False, na=False))
                filtered_df = filtered_df[adult_filter]
            else:
                filtered_df = filtered_df[filtered_df["연령"].str.contains(age_group, case=False, na=False)]
        
        # 결과가 없으면 원본 데이터의 일부 반환
        if filtered_df.empty and menu_df.shape[0] > 0:
            print(f"필터링 결과가 없어 기본 메뉴를 반환합니다.")
            return menu_df.head(3)  # 기본적으로 상위 3개 메뉴 반환
        return filtered_df
    
    except Exception as e:
        import traceback
        log_error(traceback.format_exc())
        print(f"필터링 중 오류 발생: {str(e)}")
        return menu_df  # 오류 발생 시 원본 데이터 반환
 
# 25.04.30일 기능추가 : 빠른 추천 -> 05.01 일부 수정 -> 05.02 추가
def recommend_by_age(menu_df, age_group: str, top_n: int = 3):
    """
    연령대 기반 빠른 추천 함수
    """
    try:
         # 05.01 일부 수정 : 연령대 정규화
        age_normalized = age_group.strip().lower() if isinstance(age_group, str) else None
        # 필터링된 메뉴 가져오기
        filtered_df = filter_menu_by_health(menu_df, age_group=age_group)
        # 결과 있는지 확인
        if filtered_df.empty:
            return f"{age_group}에게 적합한 메뉴를 찾지 못했어요."
        
        # 인기순 또는 랜덤 추천 if, else 추가 
        if len(filtered_df) <= top_n:
            recommended = filtered_df.sample(n=min(top_n, len(filtered_df))) # 필터링된 결과가 top_n보다 적으면 전체 반환
        else:
            recommended = filtered_df.sample(n=top_n)  # 랜덤 추천

        # 추천 메뉴 목록 생성
        items = recommended['이름'].tolist()
        return f"{age_group}에게 추천되는 메뉴입니다: " + ", ".join(items) + "."
   
    except Exception as e:
        import traceback
        log_error(traceback.format_exc())
        return "추천 도중 오류 발생, 관리자에게 문의 바랍니다."

 # 25.04.30일 기능추가 : 연령 + 질환을 모두 고려한 건강 맞춤 추천 함수
def recommend_by_age_and_disease(menu_df, age_gorup: str = None, diseases: list =None, top_n: int =3):
    """
    연령 + 질환을 모두 고려한 건강 맞춤 추천 함수
    """
    try:
         # 필터링된 메뉴 가져오기
        filtered_df = filter_menu_by_health(menu_df, age_group=age_gorup, diseases=diseases)
        # 결과가 있는지 확인
        if filtered_df.empty:
            return "조건에 맞는 건강 메뉴를 찾지 못했어요"
        
        # 인기순 또는 랜덤 추천
        recommended = filtered_df.sample(n=min(top_n, len(filtered_df)))  
         # 추천 메뉴 목록 생성
        items = recommended['이름'].tolist()

        cond_text = ""
        if age_gorup:
            cond_text += f"{age_gorup}" + " "
        if diseases:
            cond_text += f"{', '.join(diseases)} 질환을 고려한 "
        return f"{cond_text.strip()} 맞춤 추천 메뉴입니다: " + ", ".join(items) + "."
    # 25.05.02 try, Exception 추가 
    except Exception as e:
        import traceback
        log_error(traceback.format_exc())
        return f"추천 도중 오류 발생: {str(e)}. 관리자에게 문의 바랍니다."

# 외부환경에서 빠르게 메뉴명만 뽑기
def recommend_menu_only(menu_df, age_group=None, diseases=None, top_n=3):
    """
    사용자 조건에 따라 추천 메뉴 이름만 반환한다. 
    """
    filtered = filter_menu_by_health(menu_df, age_group, diseases)
    if filtered.empty:
        return ["추천할 메뉴가 없습니다."]
    return filtered.sort_values(by="단백질(g)", ascending=False)["이름"].head(top_n).tolist()

# 25.04.29 기능 추가 : 'RAG 응답 전처리 및 포맷팅 함수
def preprocess_rag_response(response_text, menu_df=None):
    '''RAG 응답 전처리 및 포맷팅 함수'''
     # 메뉴가 없는 경우 간결하게 응답 생성 => 조건이 좁은 관계로 수정한다.
    if any(keyword in response_text for keyword in ["없는 메뉴", "제공하지 않는", "없습니다", "죄송합니다"]):
    # if "없는 메뉴" in response_text or "제공하지 않는 메뉴" in response_text:
        clean_response = "죄송합니다. 주문하신 메뉴는 현재 제공하고 있지 않습니다. "

        # 25.05.02 기능 고도화 : 랜덤 추천, 다양한 카테고리 추천, 간결한 추천 
        # 메뉴 정보가 있다면 간결한 추천 추가 
        # 랜덤하게 다양한 추천을 제공
        # 커피, 차/티, 스무디 등 다양한 카테고리에서 추천
        if menu_df is not None:
            if '분류' in menu_df.columns:
                categories = menu_df['분류'].unique()
                sample_categories = categories if len(categories) <= 3 else pd.Series(categories).sample(3).tolist()
           
                sample_menus = []
                for category in sample_categories:
                    category_items = menu_df[menu_df['분류'] == category].sample(1)
                    if not category_items.empty:
                        sample_menus.append(category_items.iloc[0]['이름'])

                if sample_menus:
                    clean_response += "대신 다음 메뉴는 어떨까요? "
                    clean_response += ", ".join(sample_menus) + "." 
            else:         
               # 분류가 없는 경우 랜덤 추천 
                coffee_items = menu_df[menu_df['이름'].str.contains('커피|아메리카노|라떼', case=False, na=False)].head(3)
                if not coffee_items.empty:
                     clean_response += "대신 다음 메뉴는 어떨까요? "
                     menu_list = coffee_items['이름'].tolist()
                     clean_response += ", ".join(menu_list) + "."
        return clean_response
        # 일반 응답 정제

    # 04.30 추가 : 말투 다듬기
    replacements = {
        "많이들 찾으세요" : "많이들 좋아하시더라고요",
        "괜찮으실 거예요" : "좋아하실 거예요",
        "드셔보세요" : "한 번 드셔보시는 것도 좋아요",
        # 25.05.02 정제말투 추가 
        "시도해 보세요": "시도해 보시는 건 어떨까요",
        "맛있을 거예요": "맛있을 거라고 생각해요",
        "주문하시겠어요": "주문해 보시는 건 어떨까요",
        "대표적인 메뉴": "대표 메뉴",
        "인기 메뉴": "인기 있는 메뉴",
        "추천해 드립니다": "추천해 드려요",
    }
    for k, v in replacements.items():
        response_text = response_text.replace(k, v)

    # 띄어쓰기 및 문장 정리
    response_text = re.sub(r'\s+', ' ', response_text)  # 중복 공백 제거
    response_text = re.sub(r'\.+', '.', response_text)  # 중복 마침표 제거
    response_text = re.sub(r'\.\s*\.', '.', response_text)  # 마침표 간격 정리
    
    return response_text

# 25.05.01 추가 -> 05.05 질문 패턴 인식 및 검색된 컨텍스트 가격 정보 + 외에 정보 포함 수정
def identify_menu_type(query, menu_df):
    """
    사용자 질문에서 메뉴 이름을 감지하고 HOT/ICE 여부 확인
    """
    print(f"[DEBUG] identify_menu_type - query: {query}")
    print(f"[DEBUG] menu_df shape: {menu_df.shape}")
    print(f"[DEBUG] menu_df columns: {menu_df.columns.tolist()}")

    # 메뉴 이름 목록 수정 -> 길이 순으로 정렬 ( 더 긴 메뉴 부터 검사)
    #menu_names = menu_df['이름'].unique().tolist()
    menu_names = sorted(menu_df['이름'].unique().tolist(), key=len, reverse=True)
    
    hot_keywords =['뜨거운', '핫', 'hot', 'Hot', 'HOT', '따듯한', '따뜻하게', '뜨겁게', '핫뜨', '뜨뜨', "따따", '아주 뜨겁게']
    ice_keywords = ['차가운', '시원한', '아이스', 'ice', 'Ice', 'ICE', '시원하게', '차갑게', '아이스로', '쓰원', '시원', '시이원', '아주 차갑게']

    # 메뉴 이름 감지 (길이 순으로 정렬된 메뉴를 체크)
    # detected_menu = None
    # 25.05.05 [] 수정
    detected_menus = []  
    for menu in menu_names:
        if menu in query:
        # detected_menu = menu
            detected_menus.append(menu)
    # 메뉴 이름이 감지되지 않았다면 퍼지 매칭 시도
    if not detected_menus:
        query_words = query.split()
        for menu in menu_names:
            menu_words = menu.split()
            # 메뉴 이름의 주요 단어가 포함되어 있는지 확인
            if any(word in query_words for word in menu_words if len(word) > 1):
                possible_match = True
                for word in menu_words:
                    if len(word) > 1 and word not in query:
                        possible_match = False
                        break
                if possible_match:
                    detected_menus.append(menu) # 리스트 추가 
                    break # 퍼지 매칭은 첫 번째 매치만 
    # HOT/ICE 여부 감지
    is_hot = any(keyword in query for keyword in hot_keywords)
    is_ice = any(keyword in query for keyword in ice_keywords)
    
    #  25.05.05 추가 비교 질문 패턴 감지
    comparison_patterns = ['어떤 게 더', '어느 게 더', '뭐가 더', '무슨 게 더', '어떤 것이 더']
    is_comparison = any(pattern in query for pattern in comparison_patterns)
    
    # 25.05.05 추가 단맛 관련 키워드 감지
    sweet_keywords = ['달아요', '달달한', '달콤한', '단', '당도', '설탕']
    is_sweet_question = any(keyword in query for keyword in sweet_keywords)
    
    # 25.05.05 추가 디버깅 정보 추가
    print(f"[DEBUG] detected_menus: {detected_menus}")
    print(f"[DEBUG] is_comparison: {is_comparison}")
    print(f"[DEBUG] is_sweet_question: {is_sweet_question}")
    # 결과 반환
    if detected_menus:
        # HOT/ICE 처리는 첫 번째 메뉴만 확인
        first_menu = detected_menus[0]

        #if "(HOT)" in first_menu or "핫" in first_menu:
        #    return first_menu, "HOT", is_comparison, is_sweet_question
         # 이 부분에 문제가 있습니다
        # 실제 메뉴가 HOT/ICE를 명시하고 있는지 확인
        menu_data = menu_df[menu_df['이름'] == first_menu]
        if not menu_data.empty:
        # 메뉴 데이터에서 HOT/ICE 정보를 확인
            menu_temp_types = menu_data['HOT/ICE'].unique()
        
        # 하나의 온도 타입만 있는 경우
        if len(menu_temp_types) == 1 and not pd.isna(menu_temp_types[0]):
            return detected_menus, menu_temp_types[0], is_comparison, is_sweet_question
         
        # 온도 타입이 없는 경우에도 4개의 값을 반환
        return detected_menus, None, is_comparison, is_sweet_question
        # 질문에서 HOT/ICE 선호도 파악
        # if is_hot:
        #     return detected_menus, "HOT", is_comparison, is_sweet_question
        # elif is_ice:
        #     return detected_menus, "ICE", is_comparison, is_sweet_question
        # else:
        #     return None, None, is_comparison, is_sweet_question # 2개 -> 4개 수정 
    else:
        return None, None, is_comparison, is_sweet_question  # 메뉴 감지 실패
# 25.05.01 대화 이력 처리 추가 , top_k = 8로 수정
def rag_pipeline(query: str, context_chunks: list, embedder, index, top_k: int = 8, menu_df=None, conversation_history=None) -> str:

    """RAG 파이프라인 실행 - 대화 이력 처리 추가  - 대화 이력 처리 및 응답 품질 개선"""
    
    # 대화 이력이 있는 경우만 처리
    if conversation_history and menu_df is not None:
        # 메뉴 이름과 HOT/ICE 여부 감지 2개 + 25.05.05 당도 추가 2개 == 총 4개 
        #menu_name, temp_type = identify_menu_type(query, menu_df)
        #menu_name, temp_type, is_comparison, is_sweet_question = identify_menu_type(query, menu_df)
        detected_menus, temp_type, is_comparison, is_sweet_question = identify_menu_type(query, menu_df)
        
        # 대화 이력에 보류 중인 메뉴가 있고, 현재 질문에서 온도 선호도가 언급된 경우
        if conversation_history.get('pending_menu') and not detected_menus:
            prev_menu = conversation_history['pending_menu']
            
            # 온도 선호도 감지 25.05.02 일부 단어 추가 
            hot_keywords = ['따뜻한', '뜨거운', '핫', 'hot', 'Hot', 'HOT', '핫뜨', '뜨뜨' ,"따따" '따듯한', '따뜻하게', '뜨겁게', '아주 뜨겁게' ]
            ice_keywords = ['차가운', '시원한', '아이스', 'ice', 'Ice', 'ICE', '쓰원', '시원', '시이원', '차갑게', '아이스로', '아주 차갑게']
            
            if any(keyword in query.lower() for keyword in hot_keywords):
                temp_type = "HOT"
                filtered_menu = menu_df[(menu_df['이름'] == prev_menu) & 
                                       ((menu_df['HOT/ICE'] == temp_type) | 
                                        (menu_df['HOT/ICE'].isna()))]  # HOT/ICE 컬럼이 없거나 NaN인 경우도 포함
                if not filtered_menu.empty:
                    price = filtered_menu.iloc[0]['가격']
                    conversation_history['pending_menu'] = None  # 보류 메뉴 처리 완료
                    return f"네, 따뜻한 {prev_menu} 가격은 {price}원입니다."
                    
            elif any(keyword in query.lower() for keyword in ice_keywords):
                temp_type = "ICE"
                filtered_menu = menu_df[(menu_df['이름'] == prev_menu) & ((menu_df['HOT/ICE'] == temp_type) | 
                                        (menu_df['HOT/ICE'].isna()))]  # HOT/ICE 컬럼이 없거나 NaN인 경우도 포함
                if not filtered_menu.empty:
                    price = filtered_menu.iloc[0]['가격']
                    conversation_history['pending_menu'] = None  # 보류 메뉴 처리 완료
                    return f"네, 차가운 {prev_menu} 가격은 {price}원입니다."
        
         # 메뉴 감지됐으나 HOT/ICE 여부가 미지정인 경우 => menu_name => deteㄹcted_menus 변경 (25.05.05)
        if detected_menus and not temp_type:
            first_menu = detected_menus[0]  # 첫 번째 메뉴 가져오기
            # HOT/ICE 모두 있는지 확인 (수정된 코드)
            hot_available = len(menu_df[(menu_df['이름'] == first_menu) &  # menu_name을 first_menu로 변경
                                      ((menu_df['HOT/ICE'] == 'HOT') | 
                                        (menu_df['HOT/ICE'].isna()))]) > 0
            
            ice_available = len(menu_df[(menu_df['이름'] == first_menu) & 
                                      ((menu_df['HOT/ICE'] == 'ICE') | 
                                        (menu_df['HOT/ICE'].isna()))]) > 0
            
            if hot_available and ice_available: # menu_name => first_menu 변경 
                # 대화 이력에 보류 메뉴 저장
                conversation_history['pending_menu'] = first_menu
                return f"{first_menu} 주문하셨네요. 따뜻한 것으로 드릴까요, 아니면 차가운 것으로 드릴까요?"
            elif hot_available:
                price = menu_df[(menu_df['이름'] == first_menu) & 
                               ((menu_df['HOT/ICE'] == 'HOT') | 
                                (menu_df['HOT/ICE'].isna()))].iloc[0]['가격']
                return f"{first_menu}는 따뜻한 메뉴로만 제공됩니다. 가격은 {price}원입니다."
            elif ice_available:
                price = menu_df[(menu_df['이름'] == first_menu) & 
                               ((menu_df['HOT/ICE'] == 'ICE') | 
                                (menu_df['HOT/ICE'].isna()))].iloc[0]['가격']
                return f"{first_menu}는 차가운 메뉴로만 제공됩니다. 가격은 {price}원입니다."
  
    # 25.05.02 특별한 질문 패턴 처리
    if any(keyword in query.lower() for keyword in ["단 음료", "달달한", "달콤한", "단 메뉴", "당도가 높은", "아주 단"]):
        # 단 음료 추천 로직
        if menu_df is not None and "당류(g)" in menu_df.columns:
            # 당류 데이터가 있으면 당류 기준으로 필터링
            sweet_menu = menu_df.sort_values(by="당류(g)", ascending=False).head(3)
            if not sweet_menu.empty:
                menu_list = sweet_menu['이름'].tolist()
                return f"달콤한 메뉴를 찾으시는군요! 가장 달콤한 메뉴는 {', '.join(menu_list)}입니다. 이 중에서 어떤 메뉴가 좋으실까요?"
        # 일반 검색 로직
        # 검색 대상 확장 (top_k 증가)
        query_embed = embedder.encode([query])
        distances, indices = index.search(query_embed, min(top_k, len(context_chunks)))
        context = [context_chunks[i] for i in indices[0]]
         # 프롬프트 개선 - 더 자세한 컨텍스트 제공
        model = genai.GenerativeModel('gemini-2.0-flash')
    
    if any(keyword in query.lower() for keyword in ["차가운 메뉴", "아이스 메뉴", "시원한 음료", "시원한 메뉴"]):
        # 차가운 메뉴 추천 로직
        if menu_df is not None and "HOT/ICE" in menu_df.columns:
            ice_menu = menu_df[menu_df['HOT/ICE'] == 'ICE'].sample(min(3, len(menu_df[menu_df['HOT/ICE'] == 'ICE'])))
            if not ice_menu.empty:
                menu_list = ice_menu['이름'].tolist()
                return f"시원한 음료를 찾으시는군요! 추천 아이스 메뉴는 {', '.join(menu_list)}입니다. 어떤 메뉴가 좋으실까요?"
    if any(keyword in query.lower() for keyword in ["뜨거운 메뉴", "HOT 메뉴", "따듯한 음료", "따듯한 메뉴", "핫 메뉴"]):
        # 따듯한 메뉴 추천 로직
        if menu_df is not None and "HOT/ICE" in menu_df.columns:
            hot_menu = menu_df[menu_df['HOT/ICE'] == 'HOT'].sample(min(3, len(menu_df[menu_df['HOT/ICE'] == 'HOT'])))
            if not hot_menu.empty:
                menu_list = hot_menu['이름'].tolist()
                return f"따듯한 음료를 찾으시는군요! 추천 따듯한 메뉴는 {', '.join(menu_list)}입니다. 어떤 메뉴가 좋으실까요?"

    # 일반 검색 로직
    # 검색 대상 확장 (top_k 증가)
    query_embed = embedder.encode([query])
    distances, indices = index.search(query_embed, min(top_k, len(context_chunks)))
    context = [context_chunks[i] for i in indices[0]]

    # 프롬프트 개선 - 더 자세한 컨텍스트 제공
    model = genai.GenerativeModel('gemini-2.0-flash')

    # 25.05.02 이전 대화 컨텍스트 추가
    conversation_context = ""
    if conversation_history and conversation_history.get('previous_query') and conversation_history.get('previous_response'):
        conversation_context = f"""
        이전 질문: {conversation_history['previous_query']}
        이전 답변: {conversation_history['previous_response']}
        """

     # 25.05.02 : 메뉴 정보를 프롬프트에 추가
    menu_info = ""
    if menu_df is not None:
        # HOT/ICE 옵션이 있는 메뉴 목록
        if "HOT/ICE" in menu_df.columns:
            hot_menus = menu_df[menu_df['HOT/ICE'] == 'HOT']['이름'].unique().tolist()
            ice_menus = menu_df[menu_df['HOT/ICE'] == 'ICE']['이름'].unique().tolist()
            menu_info += f"""
            HOT 메뉴: {', '.join(hot_menus[:5])} 등 {len(hot_menus)}개
            ICE 메뉴: {', '.join(ice_menus[:5])} 등 {len(ice_menus)}개
            """
    # 25.05.02 분류별 메뉴 정보 추가
        if "분류" in menu_df.columns:
            categories = menu_df['분류'].unique()
            for category in categories:
                category_items = menu_df[menu_df['분류'] == category]['이름'].unique().tolist()
                menu_info += f"{category}: {', '.join(category_items[:3])} 등 {len(category_items)}개\n"
     # 25.05.05 추가 :  메뉴 비교 질문 처리 추가
    #detected_menus, _, _, is_comparison, is_sweet_question = identify_menu_type(query, menu_df)
    #detected_menus = menu_name  # menu_name은 리스트임
    # 25.05.05 추가 : 당류 추가 
    if detected_menus is not None and is_comparison and is_sweet_question:
        # 비교할 메뉴들의 당류 정보 찾기
        comparison_results = []
        print(f"[DEBUG] 비교하려는 메뉴들: {detected_menus}")
        for menu in detected_menus:
            menu_data = menu_df[menu_df['이름'] == menu]
            print(f"[DEBUG] 메뉴 '{menu}' 검색 결과: {len(menu_data)}개")

            if not menu_data.empty:
                #for idx, row in menu_data.iterrows():
                    #sugar = row.get('당류(g)', None)
                sugar = menu_data.iloc[0].get('당류(g)', None)
                print(f"[DEBUG] {menu}의 당류 원본: {sugar} (타입: {type(sugar)})")

                if sugar is None or pd.isna(sugar):
                    sugar = 0
                elif isinstance(sugar, str):
                    import re
                    sugar = re.sub(r'[^0-9.]', '', sugar)
                    sugar = float(sugar) if sugar else 0
                else:
                    try:
                        sugar = float(sugar)
                    except:
                        sugar = 0
                        
                print(f"[DEBUG] {menu}의 당류: {sugar}")
                comparison_results.append((menu, sugar))
                  # 값이 숫자가 아닌 경우 0으로 처리
        print(f"[DEBUG] 비교 결과: {comparison_results}")

        if comparison_results:
            # comparison_results.sort(key=lambda x: x[1], reverse=True)
            # sweeter_menu = comparison_results[0][0]
            # sweeter_sugar = comparison_results[0][1]
            sugars = [result[1] for result in comparison_results]
            max_sugar = max(sugars)
            min_sugar = min(sugars)
        
            # 모든 값이 같은 경우
            if max_sugar == min_sugar:
                if max_sugar == 0:
                    return f"{' '.join(detected_menus)} 모두 당류 정보가 없어서 정확한 비교가 어렵습니다. 둘 다 비슷한 정도의 단맛이라고 생각하시면 됩니다."
                else:
                    return f"{' '.join(detected_menus)} 모두 당류 함량이 같아서({max_sugar}g) 단맛이 비슷합니다."
            # 가장 단 메뉴 찾기
            comparison_results.sort(key=lambda x: x[1], reverse=True)
            sweeter_menu = comparison_results[0][0]
            sweeter_sugar = comparison_results[0][1]
        
            print(f"[DEBUG] 가장 단 메뉴: {sweeter_menu}, 당류량: {sweeter_sugar}")
        
            # 당류 값이 0인 경우
            if sweeter_sugar == 0:
                return f"두 메뉴 모두 당류 데이터가 없어서 정확한 비교가 어렵습니다. 일반적으로 아메리카노는 단맛이 거의 없고, 카페라떼는 우유로 인해 약간의 단맛이 있습니다."
                    
            # 정상적인 비교 답변
            return f"{sweeter_menu}이 더 달아요. {sweeter_menu}의 당류 함량은 {sweeter_sugar}g입니다."
        else:
            return "선택하신 메뉴의 당류 정보를 찾을 수 없습니다. 다시 확인해주세요."
            
        #return f"{comparison_results[0][0]}이 더 달아요. {sweeter_menu}의 당류 함량은 {sweeter_sugar}g입니다."
    # 25.05.02 개선된 프롬프트 추가 -> 25.05.05 추가 : 9, 10, 11번 추가 
    prompt = f"""다음 데이터와 이전 대화 맥락을 바탕으로 카페 메뉴 질문에 답변해주세요.

        메뉴 데이터:
        {' '.join(context)}

        메뉴 분류 정보:
        {menu_info}

        {conversation_context}

        질문: {query}

        답변 가이드라인:
        1. 사람이 직접 대화하는 것처럼 자연스럽고 간결하게 답변하세요.
        2. 특수문자나 마크다운 포맷은 사용하지 말고, 음성으로 변환되었을 때 자연스럽게 들리도록 해주세요.
        3. 메뉴 추천을 할 때는 가장 적합한 2-3개 항목만 간결하게 언급해주세요.
        4. 가격 정보는 정확히 제공하고, 메뉴 종류나 개수를 물어보는 질문에는 정확한 수치와 몇 가지 예시를 알려주세요.
        예시: "커피(HOT) 메뉴는 총 13종류입니다. 아메리카노, 카페라떼, 바닐라라떼 등이 있어요."
        5. 개수와 가격을 물어보는 질문에 정확한 수치와 예시를 알려주세요.
        예시: "아메리카노 3개의 가격은 7500원입니다."
        6. 손님이 HOT/ICE를 명시하지 않고 단순히 메뉴 이름만 말했다면, HOT/ICE 여부를 물어보세요.
        예시: "아메리카노 주문하셨네요. 따뜻한 것으로 드릴까요, 아니면 차가운 것으로 드릴까요?"
        7. 손님이 "따뜻한 거요" 또는 "차가운 거요"라고 대답하면, 해당하는 메뉴와 가격을 알려주세요.
        예시: "네, 따뜻한 아메리카노 가격은 2500원입니다."
        8. 음료의 특성(단 음료, 쓴 음료, 시원한 음료 등)을 물어보면 적절한 메뉴를 추천해주세요.
        예시: "달콤한 음료를 찾으시나요? 카라멜마키아또, 바닐라라떼, 초콜릿라떼가 가장 달콤한 메뉴입니다."
        9. 비교 질문("어떤 게 더 달아요?")의 경우, 각 메뉴의 당류 함량을 비교하여 답변하세요.
        10. 메뉴 정보가 없는 경우, 일반적인 상식으로 답변하되 정확하게 전달하세요.
        11. 달콤한 정도를 비교할 때는 당류(g) 수치를 기준으로 하세요.

        답변:"""
    
    # 25.04.30 중복 회피 및 답변 다양성 증가 : 응답생성
    response = model.generate_content(prompt, generation_config={
        "temperature": 0.9,
        "top_p": 0.95,
        "top_k": 40,
    })

    # 25.05.02 로깅 및 학습을 위한 응답 저장
    raw_response = response.text

    # 25.04.29 응답 전처리 및 포맷팅
    processed_response = preprocess_rag_response(response.text, menu_df)
    # 학습 데이터 로깅 - 원본 질문, 응답, 처리된 응답
    save_rag_response_log(query, processed_response)
    
    return processed_response

# 25.05.02 기능추가 :  rag_log.txt 파일을 활용한 지속 학습 로직
def save_rag_response_log(query, answer, response_path="rag_log.txt"):
    """
    RAG 질의응답 로그를 저장하고 학습에 활용하는 함수 - 개선된 버전
    """
    # 로그 파일에 질문과 답변 저장
    with open(response_path, "a", encoding="utf-8") as f:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"[시간] {timestamp}\n[질문] {query}\n[답변] {answer}\n\n")
    
    # 주기적으로 로그 파일에서 데이터 학습
    try:
        # 로그 파일 크기가 일정 크기 이상이면 학습 데이터로 활용
        if os.path.getsize(response_path) > 10000:  # 10KB 이상
            learn_from_log(response_path)
    except Exception as e:
        print(f"로그 파일 학습 중 오류: {str(e)}")

# 25.05.02 : 로그 파일에서 질문-답변 쌍 추출 및 인덱스 강화 함수
def learn_from_log(log_path="rag_log.txt"):
    """
    로그 파일에서 질문-답변 쌍을 추출하여 임베딩 및 검색 인덱스 강화
    """
    try:
        # 로그 파일 읽기
        with open(log_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # 질문-답변 쌍 추출
        qa_pairs = []
        sections = content.split("\n\n")
        
        for section in sections:
            if "[질문]" in section and "[답변]" in section:
                lines = section.strip().split("\n")
                question = None
                answer = None
                
                for line in lines:
                    if line.startswith("[질문]"):
                        question = line[5:].strip()
                    elif line.startswith("[답변]"):
                        answer = line[5:].strip()
                
                if question and answer:
                    qa_pairs.append((question, answer))
        
        # 학습 데이터 준비 완료 알림
        print(f"로그에서 {len(qa_pairs)}개의 질문-답변 쌍을 추출했습니다.")
        
        # 여기에 실제 학습 로직 구현 가능
        # 예: fine-tuning 또는 임베딩 인덱스 업데이트
        
        # 로그 파일 백업 및 초기화 (선택 사항)
        if len(qa_pairs) > 100:  # 충분한 데이터가 모이면 백업 후 초기화
            backup_file = f"{log_path}.{datetime.now().strftime('%Y%m%d%H%M%S')}"
            os.rename(log_path, backup_file)
            print(f"로그 파일을 {backup_file}로 백업했습니다.")
            
    except Exception as e:
        print(f"로그 파일 학습 처리 중 오류: {str(e)}")

# 25.05.01 대화 이력 관리 부분 추가  top_k => 8로 수정
def main(csv_path: str, chunk_size: int = 1000, top_k: int = 8):
    """메인 실행 함수"""

    # 대화 이력 초기화
    conversation_history = {
        'pending_menu': None,  # HOT/ICE 여부를 기다리는 메뉴 이름
        'previous_query': None,  # 이전 질문
        'previous_response': None  # 이전 답변
    }
    
    prev_answer = ""
    # 환경 설정
    load_dotenv()
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("Error: GEMINI_API_KEY가 설정되지 않았습니다.")
        return
    genai.configure(api_key=api_key)

    # CSV 데이터 로드 
    print(f"CSV 파일 '{csv_path}' 처리 중...")
    
    # 25.05.01 menu_df- 모든 컬럼 포함하여 로드 
    try:
        menu_df = pd.read_csv(csv_path, encoding='utf-8')
        print(f"메뉴 데이터 로드 완료: {len(menu_df)}개 항목")
    except Exception as e:
        print(f"메뉴 데이터 로드 오류: {str(e)}")
        return

    # 텍스트 데이터 준비
    texts = load_csv_data(csv_path)
    if not texts:
        return
    
    # 텍스트 청크 분할
    chunks = split_into_chunks(texts, chunk_size)
    print(f"총 {len(chunks)}개의 청크로 분할됨")

    # 임베딩 모델 준비
    embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embeddings = embedder.encode(chunks)
    
    # FAISS 인덱스 구축
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings, dtype=np.float32))

    # 대화형 질의응답 루프
    print("\nRAG 시스템이 준비되었습니다. 질문을 입력해주세요.")
    while True:
        input_str = input("\n질문을 입력하세요 (종료: q): ")
        if input_str.lower() == 'q':
            break
            
        user_query = input_str.strip()
        if len(user_query) < 2 or not re.search(r'[가-힣a-zA-Z0-9]', user_query):
            print("\n죄송합니다, 다시 말씀해 주세요.")
            continue
        print("\n답변 생성 중...")

        # 4.30일 응답 중복 방지 (이전 응답을 기억하고 필터링한다)
        # 25.05.02 코드 수정
        answer = rag_pipeline(user_query, chunks, embedder, index, top_k, menu_df, conversation_history)     
        # 25.05.02 추가 : 대화 이력 업데이트
        conversation_history['previous_query'] = user_query
        conversation_history['previous_response'] = answer

        # 이전 응답과 너무 비슷한 경우 재생성
        if prev_answer and answer.strip() == prev_answer.strip():
            print("같은 응답 반복 방지를 위해 다시 생성 중")
            answer = rag_pipeline(user_query + "다른 추천도 알려줘", chunks, embedder, index, top_k)

        prev_answer = answer 

        # 질문 받기 
        print(f"\n질문: {user_query}")
        print(f"답변: {answer}")
        
        print("\nTTS 변환 중...")
        text_to_speech(answer)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='CSV 기반 RAG 시스템')
    parser.add_argument('--csv', type=str, default="process_data.csv", help='CSV 파일 경로')
    parser.add_argument('--chunk_size', type=int, default=1000, help='청크 크기')
    parser.add_argument('--top_k', type=int, default=4, help='검색할 상위 k개 문서 사용')
    
    args = parser.parse_args()
    try:
        main(csv_path=args.csv, chunk_size=args.chunk_size, top_k=args.top_k)
    except Exception:
        log_error(traceback.format_exc())
        print("⚠ 시스템 실행 중 오류가 발생했습니다. error_log.txt 파일을 확인해주세요.")