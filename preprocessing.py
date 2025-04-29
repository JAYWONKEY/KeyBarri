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

        return text.strip()

def text_to_speech(text, lang='ko'):
    """텍스트를 음성으로 변환하고 재생하는 함수"""
    try:
        # TTS용 텍스트 정제 
        clean_text = clean_text_for_tts(text)
        print(f"\n[TTS용 정제된 텍스트]\n{clean_text}")
        
        filename = f"speech_{str(uuid.uuid4())[:8]}.mp3"
        tts = gTTS(text=text, lang=lang)
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

csv_path=r"data.csv"

# 25.04.29 기능 추가 : load_csv_data 내 인덱스 번호 제거 
def load_csv_data(csv_path):
    """CSV 파일에서 데이터 로드"""
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
            selected_cols = ['이름', '가격', '설명']
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
# 25.04.29 기능 추가 : 'RAG 응답 전처리 및 포맷팅 함수
def preprocess_rag_response(response_text, menu_df=None):
    '''RAG 응답 전처리 및 포맷팅 함수'''
     # 메뉴가 없는 경우 간결하게 응답 생성
    if "없는 메뉴" in response_text or "제공하지 않는 메뉴" in response_text:
        clean_response = "죄송합니다. 주문하신 메뉴는 현재 제공하고 있지 않습니다. "

        # 메뉴 정보가 있다면 간결한 추천 추가 
        if menu_df is not None:
            coffee_items = menu_df[menu_df['상품명'].str.contains('커피|아메리카노|라떼', case=False, na=False)].head(3)
            if not coffee_items.empty:
                 clean_response += "대신 다음 메뉴는 어떨까요? "
                 menu_list = coffee_items['상품명'].tolist()
                 clean_response += ", ".join(menu_list) + "."
        return clean_response
        # 일반 응답 정제
    return response_text


def rag_pipeline(query: str, context_chunks: list, embedder, index, top_k: int = 4, menu_df=None) -> str:
    """RAG 파이프라인 실행"""
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

답변:"""
    
    response = model.generate_content(prompt)
    # 25.04.29 응답 전처리 및 포맷팅
    preprocess_response =  preprocess_rag_response(response.text, menu_df)
    return preprocess_response
    # return response.text

def main(csv_path: str, chunk_size: int = 1000, top_k: int = 4):
    """메인 실행 함수"""
    # 환경 설정
    load_dotenv()
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("Error: GEMINI_API_KEY가 설정되지 않았습니다.")
        return
    genai.configure(api_key=api_key)

    # CSV 데이터 로드
    print(f"CSV 파일 '{csv_path}' 처리 중...")
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
        print("\n답변 생성 중...")
        answer = rag_pipeline(user_query, chunks, embedder, index, top_k)
        
        print(f"\n질문: {user_query}")
        print(f"답변: {answer}")
        
        print("\nTTS 변환 중...")
        text_to_speech(answer)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='CSV 기반 RAG 시스템')
    parser.add_argument('--csv', type=str, default="data.csv", help='CSV 파일 경로')
    parser.add_argument('--chunk_size', type=int, default=1000, help='텍스트 청크 크기')
    parser.add_argument('--top_k', type=int, default=4, help='검색할 상위 문서 수')
    
    args = parser.parse_args()
    main(args.csv, args.chunk_size, args.top_k)