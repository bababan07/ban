import time
import logging
from pathlib import Path

import numpy as np
import pyaudio
import torch
import whisper
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# --- 1. 설정 (이 부분에서 모든 것을 제어) ---
# 경로 설정
BASE_DIR = Path(__file__).resolve().parent
LOG_FILE = BASE_DIR / "recorder.log"
OUTPUT_FILE = BASE_DIR / "filtered_sentences.txt"

# 오디오 설정
SAMPLING_RATE = 16000
CHUNK_SIZE = 1024
CHANNELS = 1
SILENCE_THRESHOLD = 500  # 마이크와 환경에 맞게 조절 필요
SILENCE_DURATION = 1.5  # 이 시간(초) 이상 침묵 시 처리 시작

# Whisper 모델 설정
MODEL_TYPE = "base"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 임베딩 및 필터링 설정
EMBEDDING_MODEL = 'jhgan/ko-sbert-nli'
TARGET_SENTENCES = [
    "학교 급식 메뉴와 맛에 대한 이야기",
    "화장실이나 복도 등 학교 시설에 대한 건의",
    "친구 사이의 다툼이나 학교 폭력에 대한 대화",
    "교실 수업 환경이나 선생님에 대한 의견"
]
SIMILARITY_THRESHOLD = 0.45 # 임계값 (조절 가능)

# --- 2. 로깅 설정 ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler() # 터미널에도 로그 출력
    ]
)

# --- 3. 모델 로드 ---
logging.info("모델을 로드합니다...")
try:
    whisper_model = whisper.load_model(MODEL_TYPE, device=DEVICE)
    embedding_model = SentenceTransformer(EMBEDDING_MODEL, device=DEVICE)
    target_embeddings = embedding_model.encode(TARGET_SENTENCES)
    logging.info("모든 모델 로드를 완료했습니다.")
except Exception as e:
    logging.error(f"모델 로드 중 오류 발생: {e}")
    exit()

# --- 4. 메인 로직 ---
def main():
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=CHANNELS, rate=SAMPLING_RATE,
                    input=True, frames_per_buffer=CHUNK_SIZE)
    
    logging.info("음성 인식을 시작합니다. (Ctrl+C로 종료)")
    
    audio_buffer = []
    silence_start_time = None
    is_speaking = False

    try:
        while True:
            data = stream.read(CHUNK_SIZE)
            audio_chunk = np.frombuffer(data, dtype=np.int16)
            rms = np.sqrt(np.mean(audio_chunk.astype(float)**2))

            if rms > SILENCE_THRESHOLD:
                if not is_speaking:
                    logging.info("말하는 중...")
                    is_speaking = True
                audio_buffer.append(data)
                silence_start_time = None
            elif is_speaking:
                if silence_start_time is None:
                    silence_start_time = time.time()
                elif time.time() - silence_start_time > SILENCE_DURATION:
                    process_audio_buffer(audio_buffer)
                    audio_buffer, is_speaking, silence_start_time = [], False, None
                    logging.info("다시 말씀해주세요...")

    except KeyboardInterrupt:
        logging.info("사용자에 의해 프로그램이 종료되었습니다.")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()
        logging.info("오디오 스트림을 안전하게 종료했습니다.")

def process_audio_buffer(buffer):
    if not buffer:
        return
        
    logging.info("침묵 감지, 음성 처리를 시작합니다.")
    audio_data = b''.join(buffer)
    audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

    try:
        # Whisper 텍스트 변환
        result = whisper_model.transcribe(audio_np, language="ko", fp16=(DEVICE=="cuda"))
        text = result.get('text', '').strip()

        if not text:
            logging.warning("인식된 텍스트가 없습니다.")
            return

        logging.info(f"인식 결과: {text}")

        # 임베딩 유사도 측정
        new_embedding = embedding_model.encode([text])
        similarities = cosine_similarity(new_embedding, target_embeddings)
        max_similarity = np.max(similarities)
        logging.info(f"최고 유사도: {max_similarity:.4f}")

        # 필터링 및 저장
        if max_similarity >= SIMILARITY_THRESHOLD:
            logging.info(f"주제 매칭! 파일에 저장합니다. (유사도: {max_similarity:.2f})")
            with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
                f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [유사도: {max_similarity:.2f}] {text}\n")

    except Exception as e:
        logging.error(f"음성 처리 중 오류 발생: {e}")

if __name__ == "__main__":
    main()
