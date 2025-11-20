import argparse
import tempfile
import traceback
import cv2 
import numpy as np
from PIL import Image
# Gradio/Spaces 관련 임포트 제거됨

# --- image_gen_aux 라이브러리에서 핵심 클래스를 임포트 ---
from image_gen_aux import UpscaleWithModel 
from image_gen_aux.utils import load_image
import torch # 라이브러리 의존성 때문에 유지 (이전 논의 참고)

# --- Model Configuration ---
SINGLE_MODEL_PATH = "4xDF2K_plksr_tiny_fp16_500k.onnx" 

LOADED_MODELS_CACHE = {}

def get_upscaler(model_path: str):
    """지정된 ONNX 모델 경로를 사용하여 Upscaler 객체를 로드하고 캐시합니다."""
    if model_path not in LOADED_MODELS_CACHE:
        print(f"Loading local ONNX model: {model_path}")
        # ONNX 모델 로딩 (image_gen_aux가 내부적으로 ONNX Runtime 사용)
        upscaler = UpscaleWithModel.from_pretrained(model_path)
        LOADED_MODELS_CACHE[model_path] = upscaler
    return LOADED_MODELS_CACHE[model_path]


# --- Core Upscaling Function (Video) ---
def upscale_video(video_path, output_path):
    print(f"\n--- 비디오 업스케일링 시작 ---")
    print(f"입력 파일: {video_path}")
    print(f"출력 파일: {output_path}")

    try:
        upscaler = get_upscaler(SINGLE_MODEL_PATH)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"비디오 파일 열기 실패: {video_path}")

        # 메타데이터 추출
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        scale_factor = 4
        new_width = frame_width * scale_factor
        new_height = frame_height * scale_factor

        # VideoWriter 객체 초기화
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        out = cv2.VideoWriter(output_path, fourcc, fps, (new_width, new_height))

        # 프레임별 처리 루프
        processed_frames = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # 진행 상황 출력 (CLI용)
            if processed_frames % 50 == 0:
                 print(f"처리 중... 프레임 {processed_frames}/{frame_count}")
            
            # BGR -> RGB -> PIL
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)

            # 업스케일링 (Tiling 사용)
            upscaled_pil_image = upscaler(pil_image, tiling=True, tile_width=1024, tile_height=1024)

            # PIL -> NumPy -> BGR로 변환 후 VideoWriter에 쓰기
            upscaled_numpy = np.array(upscaled_pil_image)
            bgr_frame = cv2.cvtColor(upscaled_numpy, cv2.COLOR_RGB2BGR)
            out.write(bgr_frame)
            
            processed_frames += 1

        # 자원 해제
        cap.release()
        out.release()
        
        print(f"\n--- 성공적으로 업스케일링 완료. {processed_frames} 프레임 처리 ---")

    except Exception as e:
        print(f"\nFATAL ERROR: 비디오 처리 중 오류 발생: {e}")
        traceback.print_exc()


# ----------------------------------------------------
# 메인 실행 구문 (명령줄 인터페이스)
# ----------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ONNX 모델을 사용한 비디오 업스케일링 CLI 툴")
    
    parser.add_argument("input_file", type=str, help="업스케일할 비디오 파일 경로 (예: video.mp4)")
    parser.add_argument("--output", type=str, default="upscaled_output.mp4", 
                        help="업스케일된 비디오를 저장할 경로 및 이름 (기본값: upscaled_output.mp4)")

    args = parser.parse_args()
    
    # 비디오 처리 함수 직접 호출
    upscale_video(args.input_file, args.output)
