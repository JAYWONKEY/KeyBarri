#5.7STT 파일 추가
import sounddevice as sd
from scipy.io.wavfile import write
import whisper
from main import rag_pipeline  # 기존 RAG 함수

def run_stt_pipeline(menu_texts, embedder, index, menu_df, conversation_history, duration=5):
    """마이크로 녹음 → Whisper STT → RAG 응답 반환"""
    fs = 16000
    filename = "voice_input.wav"

    print("🎙 음성 녹음 중...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    write(filename, fs, audio)
    print("✅ 녹음 완료")

    print("🧠 Whisper 처리 중...")
    model = whisper.load_model("small")
    result = model.transcribe(filename, language="Korean")
    question = result["text"]

    print("❓ 질문 인식:", question)

    # RAG 응답
    answer = rag_pipeline(
        question, menu_texts, embedder, index,
        menu_df=menu_df,
        conversation_history=conversation_history
    )
    print("🤖 응답:", answer)

    return question, answer