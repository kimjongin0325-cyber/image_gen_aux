import argparse
import tempfile
import traceback
import cv2 
import numpy as np
from PIL import Image
import onnxruntime as ort # ONNX Runtime 직접 임포트

# --- Model Configuration ---
# 사용자님의 파일 구조에 있는 ONNX 파일명 사용
SINGLE_MODEL_PATH = "4xDF2K_plksr_tiny_fp16_500k.onnx" 

LOADED_MODELS_CACHE = {}

# --- ONNX 세션 로딩 함수 ---
def get_upscaler(model_path: str):
    """지정된 ONNX 모델 경로를 사용하여 ONNX InferenceSession 객체를 로드합니다."""
    if model_path not in LOADED_MODELS_CACHE:
        print(f"Loading ONNX Runtime session: {model_path}")
        
        # GPU 사용을 위한 세션 옵션 설정 (CUDAExecutionProvider를 우선 사용)
        sess_options = ort.SessionOptions()
        LOADED_MODELS_CACHE[model_path] = ort.InferenceSession(
            model_path, 
            sess_options, 
            # Colab GPU 환경을 위해 CUDAExecutionProvider 사용
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider'] 
        )
    return LOADED_MODELS_CACHE[model_path]


# --- 핵심 비디오 업스케일링 함수 (루프 포함) ---
def upscale_video(video_path, output_path):
    print(f"\n--- 비디오 업스케일링 시작 ---")
    print(f"입력 파일: {video_path}")
    print(f"출력 파일: {output_path}")

    try:
        # 1. 모델 로드 및 정보 추출
        session = get_upscaler(SINGLE_MODEL_PATH)
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        
        # 2. 비디오 캡처 초기화
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"비디오 파일 열기 실패: {video_path}")

        # 3. 메타데이터 추출 및 VideoWriter 설정
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        scale_factor = 4 # 4배 업스케일링 모델로 가정
        new_width = frame_width * scale_factor
        new_height = frame_height * scale_factor

        # VideoWriter 객체 초기화 (코덱: mp4v)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        out = cv2.VideoWriter(output_path, fourcc, fps, (new_width, new_height), isColor=True)

        # 4. 프레임별 처리 루프 (영상 업스케일의 핵심)
        processed_frames = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # 진행 상황 출력
            if processed_frames % 50 == 0:
                 print(f"처리 중... 프레임 {processed_frames}/{frame_count}")
            
            # --- ONNX 전처리 ---
            # 1. BGR(cv2 기본) -> RGB로 색상 순서 변환
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)

            # 2. 값 범위 정규화 (0-255 -> 0.0-1.0) 및 NCHW 텐서 형식 변환
            input_array = np.array(pil_image).astype(np.float32) / 255.0
            input_array = np.transpose(input_array, (2, 0, 1)) # HWC -> CHW
            input_tensor = np.expand_dims(input_array, axis=0) # CHW -> NCHW

            # 3. ONNX 추론 실행
            ort_output = session.run([output_name], {input_name: input_tensor})[0]

            # --- ONNX 후처리 ---
            # 4. NCHW -> HWC 변환 및 값 범위 복원 (0.0-1.0 -> 0-255)
            output_array = np.squeeze(ort_output, axis=0) # NCHW -> CHW
            output_array = np.transpose(output_array, (1, 2, 0)) # CHW -> HWC
            output_array = np.clip(output_array * 255.0, 0, 255).astype(np.uint8)

            # 5. VideoWriter에 쓰기 위한 RGB -> BGR 변환
            bgr_frame = cv2.cvtColor(output_array, cv2.COLOR_RGB2BGR)
            out.write(bgr_frame)
            
            processed_frames += 1

        # 6. 자원 해제
        cap.release()
        out.release()
        
        if processed_frames > 0:
            print(f"\n--- 성공적으로 업스케일링 완료. {processed_frames} 프레임 처리 ---")
            print(f"결과 파일: {output_path}")
        else:
            print(f"\n--- 업스케일링 실패 ---")


    except Exception as e:
        print(f"\nFATAL ERROR: 비디오 처리 중 치명적인 오류 발생: {e}")
        traceback.print_exc()

# --- 메인 CLI 실행 블록 ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ONNX 모델을 사용한 비디오 업스케일링 CLI 툴")
    
    parser.add_argument("input_file", type=str, help="업스케일할 비디오 파일 경로 (예: video.mp4)")
    parser.add_argument("--output", type=str, default="upscaled_output.mp4", 
                        help="업스케일된 비디오를 저장할 경로 및 이름 (기본값: upscaled_output.mp4)")

    args = parser.parse_args()
    
    upscale_video(args.input_file, args.output)
