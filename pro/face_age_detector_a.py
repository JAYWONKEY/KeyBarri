import cv2
from deepface import DeepFace
from PIL import ImageFont, ImageDraw, Image
import numpy as np
import time

def put_korean_text(frame, text, position, font_path='C:/Windows/Fonts/malgun.ttf', font_size=24, color=(255,255,255)):
    """한글 텍스트를 이미지에 출력하는 함수"""
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.truetype(font_path, font_size)
    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def find_camera():
    """사용 가능한 카메라 자동 탐색"""
    for index in range(5):
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if cap.isOpened():
            print(f"[INFO] 카메라 연결 성공 - 인덱스 {index}")
            return cap
        cap.release()
    raise RuntimeError("사용 가능한 카메라를 찾을 수 없습니다.")

def main():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    cap = find_camera()

    print("\n=== 나이 감지 시스템 ===")
    print("카메라에 얼굴을 가까이 대면 나이를 분석합니다.")
    print("종료하려면 'q'를 누르세요.\n")

    detection_start_time = None
    last_face = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("카메라에서 프레임을 가져올 수 없습니다.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                current_face = (x, y, w, h)

                if last_face and abs(x - last_face[0]) < 50 and abs(y - last_face[1]) < 50:
                    if detection_start_time and (time.time() - detection_start_time > 3):
                        try:
                            face_roi = frame[y:y+h, x:x+w]
                            result = DeepFace.analyze(face_roi, actions=['age'], enforce_detection=False)
                            age = int(result[0]['age'])

                            frame = put_korean_text(frame, f"예상 나이: {age}세", (x, y-30), font_size=30, color=(50, 255, 50))
                            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        except Exception as e:
                            print(f"[분석 오류] {e}")
                    else:
                        frame = put_korean_text(frame, "분석 중...", (x, y-30), font_size=24, color=(0, 255, 0))
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                else:
                    detection_start_time = time.time()
                    last_face = current_face
                    frame = put_korean_text(frame, "얼굴 감지됨", (x, y-30), font_size=24, color=(255, 255, 0))
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            cv2.imshow('Age Detection', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
