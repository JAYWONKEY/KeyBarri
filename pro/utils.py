import os
import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image

# TensorFlow 경고 메시지 비활성화
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def put_korean_text(frame, text, position, font_path='C:/Windows/Fonts/malgun.ttf', 
                   font_size=24, color=(255,255,255)):
    """OpenCV 이미지에 한글 텍스트를 추가하는 함수"""
    try:
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        
        # 폰트 파일 존재 확인
        if not os.path.exists(font_path):
            raise FileNotFoundError(f"폰트 파일을 찾을 수 없습니다: {font_path}")
            
        font = ImageFont.truetype(font_path, font_size)
        draw.text(position, text, font=font, fill=color)
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"텍스트 렌더링 오류: {str(e)}")
        return frame