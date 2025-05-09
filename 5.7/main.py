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

csv_path='123/por0502/process_data.csv'
# import pandas as pd
# import fitz  # PyMuPDF
# import os

# 저장 경로 설정 -> task 던지는 함수 설정 -> 저장한거 자연스럽게 학습되어 답변한다.

# 25.04.29 기능 추가 : load_csv_data 내 인덱스 번호 제거 
def load_csv_data(csv_path):
    """CSV, Excel, PDF 파일 지원"""
    encodings = ['cp949', 'euc-kr', 'utf-8']
    
    for encoding in encodings:
        try:
            print(f"{encoding} 인코딩으로 CSV 파일 읽기 시도...")
            df = pd.read_csv(csv_path, encoding=encoding)

            # 25.04.29 기능 추가 : 첫 번째 컬럼이 번호면 제거
            if df.columns[0].lower() in ['no', 'index', 'id', '번호', 'id'] or df.iloc[:,0].astype(str).str.match(r'^\d+$').all():
                print("번호 컬럼 제거")
                df = df.iloc[:, 1:]

            # 25.04.29 기능 추가 : 필요한 열만 선택
            selected_cols = ['이름', '가격']
            df = df[selected_cols]

            # 텍스트 결합 
            #texts = df.apply(lambda row: f"{row['이름']} {row['가격']}원. {row['설명']}", axis=1).tolist()
            #texts = df.apply(lambda x: ' '.join(x.astype(str)), axis=1).tolist()
            texts = df.apply(lambda row: f"{row['이름']} {row['가격']}원.", axis=1).tolist()
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
    사용자 연령대와 질환 정보를 기반으로 메뉴를 필터링
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
                    # # 문자열로 변환 후 정제 작업 수행
                    # filtered_df[col] = filtered_df[col].astype(str).strip()
                    #  # 콤마 제거
                    # filtered_df[col] = filtered_df[col].str.replace(',', '')
                    #  # 숫자가 아닌 문자 제거 (소수점 유지)
                    # filtered_df[col] = filtered_df[col].str.replace(r'[^\d.]', '', regex=True)
                    # # 빈 문자열을 NaN으로 변환
                    # filtered_df[col] = filtered_df[col].replace('', np.nan)
                    # # 숫자로 변환 (오류 허용)
                    # filtered_df[col] = pd.to_numeric(filtered_df[col], errors='coerce')
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
            df = df[df["연령"].str.contains(age_group)]
            # 25.05.01 추가수정 : 연령대 그룹 정규화
            age_normalized = age_group.lower() if isinstance(age_group, str) else None

            if age_normalized == "어린이": # 0 ~ 12
                filtered_df = filtered_df[filtered_df["단백질(g)"].fillna(0) >=1]
            elif age_normalized == "청소년": # 13 ~ 18
                filtered_df = filtered_df[filtered_df["단백질(g)"].fillna(0) >=0.8]

            # 성인 일 때 청년, 중년, 장년이 들어가서 kiosk 
            # 성인 그룹 처리 (청년, 중년, 장년이 모두 성인에 포함)
            elif age_normalized in ["청년", "중년", "장년", "성인"]:
                if age_normalized == "성인" or age_normalized == "청년": 
                    filtered_df = filtered_df[filtered_df["카페인(mg)"].fillna(999) <= 200]
                elif age_normalized == "중년":
                    filtered_df = filtered_df[(filtered_df["지방(g)"].fillna(999) <= 3) & (filtered_df["포화지방(g)"].fillna(999) <= 1.5)]
                elif age_normalized == "장년":
                    filtered_df = filtered_df[(filtered_df["나트륨(mg)"].fillna(999) <= 100) & (filtered_df["카페인(mg)"].fillna(999) <= 150)]
            elif age_normalized == "노인":
                filtered_df = filtered_df[(filtered_df["단백질(g)"].fillna(0) >= 1.5) & (filtered_df["카페인(mg)"].fillna(999) <= 150)]

        # 25.05.01 추가 
        if "연령" in filtered_df.columns and age_group:
            filtered_df = filtered_df[filtered_df["연령"].str.contains(age_group, case=False, na=False)]
        return filtered_df
    
    except Exception as e:
        import traceback
        log_error(traceback.format_exc())
        print(f"필터링 중 오류 발생: {str(e)}")
        return menu_df  # 오류 발생 시 원본 데이터 반환
 
# 25.04.30일 기능추가 : 빠른 추천 -> 05.01 일부 수정 
def recommend_by_age(menu_df, age_group: str, top_n: int = 3):
    """
    연령대 기반 빠른 추천 함수
    """
    try:

         # 05.01 일부 수정 : 연령대 정규화
        age_normalized = age_group.strip().lower() if isinstance(age_group, str) else None
       
        filtered_df = filter_menu_by_health(menu_df, age_group=age_group)

        if filtered_df.empty:
            return f"{age_group}에게 적합한 메뉴를 찾지 못했어요."
        
        # 인기순 또는 랜덤 추천 if, else 추가 
        if len(filtered_df) <= top_n:
            recommended = filtered_df.sample(n=min(top_n, len(filtered_df))) # 필터링된 결과가 top_n보다 적으면 전체 반환
        else:
            recommended = filtered_df.sample(n=top_n)  # 랜덤 추천
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
    filtered_df = filter_menu_by_health(menu_df, age_group=age_gorup, diseases=diseases)

    if filtered_df.empty:
        return "조건에 맞는 건강 메뉴를 찾지 못했어요"
    
    # 인기순 또는 랜덤 추천
    recommended = filtered_df.sample(n=min(top_n, len(filtered_df)))  
    items = recommended['이름'].tolist()

    cond_text = ""
    if age_gorup:
        cond_text += f"{age_gorup}"
    if diseases:
        cond_text += f"{', '.join(diseases)} 질환을 고려한 "
    return f"{cond_text.strip()} 맞춤 추천 메뉴입니다: " + ", ".join(items) + "."

# 외부환경에서 빠르게 메뉴명만 뽑기
def recommend_menu_only(menu_df, age_group=None, diseases=None, top_n=3):
    """
    사용자 조건에 따라 추천 메뉴 이름만 반환한다. 
    """
    filtered = filter_menu_by_health(menu_df, age_group, diseases)
    if filtered.empty:
        return ["추천할 메뉴가 없습니다."]
    return filtered.sort_values(by="단백질(g)", ascending=False)["이름"].head(top_n).tolist()

def save_rag_response_log(query, answer, reponse_path="rag_log.txt"):
    with open(reponse_path, "a", encoding="utf-8") as f:
        f.write(f"[질문] {query}\n[답변] {answer}\n\n")

# 25.05.01 추가 
def identify_menu_type(query, menu_df):
    """
    사용자 질문에서 메뉴 이름을 감지하고 HOT/ICE 여부 확인
    """
    # 메뉴 이름 목록
    menu_names = menu_df['이름'].unique().tolist()
    
    # HOT/ICE 관련 키워드
    hot_keywords = ['따뜻한', '뜨거운', '핫', 'hot', 'Hot', 'HOT']
    ice_keywords = ['차가운', '시원한', '아이스', 'ice', 'Ice', 'ICE']
    
    # 메뉴 이름 감지
    detected_menu = None
    for menu in menu_names:
        if menu in query:
            detected_menu = menu
            break
    
    # HOT/ICE 여부 감지
    is_hot = any(keyword in query for keyword in hot_keywords)
    is_ice = any(keyword in query for keyword in ice_keywords)
    
    # 결과 반환
    if detected_menu:
        if is_hot:
            return detected_menu, "HOT"
        elif is_ice:
            return detected_menu, "ICE"
        else:
            return detected_menu, None  # HOT/ICE 여부 미지정
    else:
        return None, None  # 메뉴 감지 실패
    

# 25.04.29 기능 추가 : 'RAG 응답 전처리 및 포맷팅 함수
def preprocess_rag_response(response_text, menu_df=None):
    '''RAG 응답 전처리 및 포맷팅 함수'''
     # 메뉴가 없는 경우 간결하게 응답 생성 => 조건이 좁은 관계로 수정한다.
    if any(keyword in response_text for keyword in ["없는 메뉴", "제공하지 않는", "없습니다", "죄송합니다"]):
    # if "없는 메뉴" in response_text or "제공하지 않는 메뉴" in response_text:
        clean_response = "죄송합니다. 주문하신 메뉴는 현재 제공하고 있지 않습니다. "

        # 메뉴 정보가 있다면 간결한 추천 추가 
        if menu_df is not None:
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
    }
    for k, v in replacements.items():
        response_text = response_text.replace(k, v)

    return response_text

# 25.05.01 내용 수정 - 대화 이력 처리 추가 
#def rag_pipeline(query: str, context_chunks: list, embedder, index, top_k: int = 4, menu_df=None) -> str:
def rag_pipeline(query: str, context_chunks: list, embedder, index, top_k: int = 4, menu_df=None, conversation_history=None) -> str:

    """RAG 파이프라인 실행 - 대화 이력 처리 추가 """

    # 대화 이력이 있는 경우만 처리
    if conversation_history and menu_df is not None:
        # 메뉴 이름과 HOT/ICE 여부 감지
        menu_name, temp_type = identify_menu_type(query, menu_df)
        
        # 대화 이력에 보류 중인 메뉴가 있고, 현재 질문에서 온도 선호도가 언급된 경우
        if conversation_history.get('pending_menu') and not menu_name:
            prev_menu = conversation_history['pending_menu']
            
            # 온도 선호도 감지
            hot_keywords = ['따뜻한', '뜨거운', '핫', 'hot', 'Hot', 'HOT']
            ice_keywords = ['차가운', '시원한', '아이스', 'ice', 'Ice', 'ICE']
            
            if any(keyword in query.lower() for keyword in hot_keywords):
                temp_type = "HOT"
                filtered_menu = menu_df[(menu_df['이름'] == prev_menu) & (menu_df['HOT/ICE'] == temp_type)]
                if not filtered_menu.empty:
                    price = filtered_menu.iloc[0]['가격']
                    conversation_history['pending_menu'] = None  # 보류 메뉴 처리 완료
                    return f"네, 따뜻한 {prev_menu} 가격은 {price}원입니다."
                    
            elif any(keyword in query.lower() for keyword in ice_keywords):
                temp_type = "ICE"
                filtered_menu = menu_df[(menu_df['이름'] == prev_menu) & (menu_df['HOT/ICE'] == temp_type)]
                if not filtered_menu.empty:
                    price = filtered_menu.iloc[0]['가격']
                    conversation_history['pending_menu'] = None  # 보류 메뉴 처리 완료
                    return f"네, 차가운 {prev_menu} 가격은 {price}원입니다."
        
        # 메뉴 감지됐으나 HOT/ICE 여부가 미지정인 경우
        if menu_name and not temp_type:
            # HOT/ICE 모두 있는지 확인
            has_hot = len(menu_df[(menu_df['이름'] == menu_name) & (menu_df['HOT/ICE'] == 'HOT')]) > 0
            has_ice = len(menu_df[(menu_df['이름'] == menu_name) & (menu_df['HOT/ICE'] == 'ICE')]) > 0
            
            if has_hot and has_ice:
                # 대화 이력에 보류 메뉴 저장
                conversation_history['pending_menu'] = menu_name
                return f"{menu_name} 주문하셨네요. 따뜻한 것으로 드릴까요, 아니면 차가운 것으로 드릴까요?"
            

    query_embed = embedder.encode([query])
    distances, indices = index.search(query_embed, min(top_k, len(context_chunks)))
    context = [context_chunks[i] for i in indices[0]]
    
    model = genai.GenerativeModel('gemini-2.0-flash')
    prompt = f"""다음 데이터를 바탕으로 질문에 답변해주세요.

데이터:
{' '.join(context)}

질문: {query}
    답변은 사람이 직접 대화하는 것처럼 자연스럽고 간결하게 해주세요.
    특수문자나 마크다운 포맷은 사용하지 말고, 음성으로 변환되었을 때 자연스럽게 들리도록 해주세요.
    메뉴 추천을 할 때는 가장 인기 있는 2-3개 항목만 간결하게 언급해주세요.

    메뉴 종류나 개수를 물어보는 질문(예: "커피 종류가 몇 개야?")에는 정확한 수치와 함께 몇 가지 예시를 알려주세요.
    예시 답변: "커피(HOT) 메뉴는 총 13종류입니다. 아메리카노, 카페라떼, 바닐라라떼 등이 있어요."

    또한 개수와 가격을 물어보는 질문에 정확한 수치와 예시를 알려주세요
    예시 답변: "아메리카노 3개의 가격은 7500원 입니다. 

    만약 손님이 HOT/ICE를 명시하지 않고 단순히 메뉴 이름만 말했다면(예: "아메리카노 주세요"), 
    HOT/ICE 여부를 물어보세요:
    예시 답변: "아메리카노 주문하셨네요. 따뜻한 것으로 드릴까요, 아니면 차가운 것으로 드릴까요?"

    만약 손님이 "따뜻한 거요" 또는 "차가운 거요"라고 대답하면, 해당하는 메뉴와 가격을 알려주세요:
    예시 답변: "네, 따뜻한 아메리카노 가격은 2500원입니다."

답변:"""
    # 25.04.30 중복 회피 및 답변 다양성 증가
    response = model.generate_content(prompt, generation_config={
        "temperature": 0.9,
        "top_p": 0.95,
        "top_k": 40,
    })
    # 25.04.29 응답 전처리 및 포맷팅
    preprocess_response =  preprocess_rag_response(response.text, menu_df)
    save_rag_response_log(query, preprocess_response)
    return preprocess_response
    # return response.text

# 25.05.01 대화 이력 관리 부분 추가 
def main(csv_path: str, chunk_size: int = 1000, top_k: int = 4):
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
        answer = rag_pipeline(user_query, chunks, embedder, index, top_k)
            
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

        

        # 4.30일 추가 : 입력값 유효성 검사 (비정상 입력 대응)
        # if len(user_query) < 2 or not re.search(r'[가-힣-a-zA-Z0-9]',user_query):
        #     print("\n죄송합니다, 다시 말씀해 주세요.")
        #     continue

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='CSV 기반 RAG 시스템')
    parser.add_argument('--csv', type=str, default="123\por0502\process_data.csv", help='CSV 파일 경로')
    parser.add_argument('--chunk_size', type=int, default=1000, help='청크 크기')
    parser.add_argument('--top_k', type=int, default=4, help='검색할 상위 k개 문서 사용')
    
    args = parser.parse_args()
    try:
        main(csv_path=args.csv, chunk_size=args.chunk_size, top_k=args.top_k)
    except Exception:
        log_error(traceback.format_exc())
        print("⚠ 시스템 실행 중 오류가 발생했습니다. error_log.txt 파일을 확인해주세요.")
    
