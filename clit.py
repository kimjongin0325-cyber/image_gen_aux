import argparse
import tempfile
import traceback
import cv2 
import numpy as np
from PIL import Image
import onnxruntime as ort

# --- Model Configuration ---
# TensorRT 엔진 파일 경로 및 이름으로 변경 (사용자 요청 반영)
SINGLE_MODEL_PATH = "/content/image_gen_aux/plksr_4x.engine" 
LOADED_MODELS_CACHE = {}

# --- ONNX/TensorRT 세션 로딩 함수 ---
def get_upscaler(model_path: str):
    """
    지정된 엔진 또는 ONNX 경로를 사용하여 ONNX InferenceSession 객체를 로드하며,
    TensorRTExecutionProvider를 최우선으로 지정합니다.
    """
    if model_path not in LOADED_MODELS_CACHE:
        print(f"Loading TensorRT/ONNX Runtime session: {model_path}")
        
        sess_options = ort.SessionOptions()
        # TensorRT Execution Provider를 최우선으로 지정하여 TensorRT를 통해 가속합니다.
        LOADED_MODELS_CACHE[model_path] = ort.InferenceSession(
            model_path, 
            sess_options, 
            providers=['TensorRTExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'] 
        )
    return LOADED_MODELS_CACHE[model_path]


# --- Tiling (타일링) 처리 함수 (성능 최적화 핵심) ---
def tiled_upscale(session, pil_image: Image.Image, tile_size: int, scale_factor: int, tile_overlap: int = 32) -> Image.Image:
    """
    타일링을 사용하여 PIL 이미지를 ONNX/TensorRT 모델로 업스케일링합니다.
    GPU 메모리 부족 문제를 해결하고 고해상도 처리를 가능하게 합니다.
    """
    
    # 1. 메타데이터 계산 및 결과 이미지 객체 생성
    img_width, img_height = pil_image.size
    output_width = img_width * scale_factor
    output_height = img_height * scale_factor
    output_img = Image.new('RGB', (output_width, output_height))
    
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    # 2. 타일 순회 및 처리
    for i in range(0, img_height, tile_size - tile_overlap):
        for j in range(0, img_width, tile_size - tile_overlap):
            
            # 입력 타일 영역 정의
            x_start = j
            y_start = i
            x_end = min(j + tile_size, img_width)
            y_end = min(i + tile_size, img_height)

            input_tile = pil_image.crop((x_start, y_start, x_end, y_end))

            # --- ONNX/TensorRT 전처리 ---
            input_array = np.array(input_tile).astype(np.float32) / 255.0
            input_array = np.transpose(input_array, (2, 0, 1)) # HWC -> CHW
            input_tensor = np.expand_dims(input_array, axis=0) # CHW -> NCHW
            
            # --- ONNX/TensorRT 추론 실행 ---
            ort_output = session.run([output_name], {input_name: input_tensor})[0]

            # --- ONNX/TensorRT 후처리 ---
            output_array = np.squeeze(ort_output, axis=0)
            output_array = np.transpose(output_array, (1, 2, 0)) # CHW -> HWC
            output_array = np.clip(output_array * 255.0, 0, 255).astype(np.uint8)
            output_tile_pil = Image.fromarray(output_array)

            # 3. 결과 타일 병합 (오버랩 제거)
            out_tile_w, out_tile_h = output_tile_pil.size
            
            # 오버랩 제거 영역 계산
            crop_top = (y_start != 0) * (tile_overlap * scale_factor // 2)
            crop_left = (x_start != 0) * (tile_overlap * scale_factor // 2)
            crop_bottom = (y_end != img_height) * (tile_overlap * scale_factor // 2)
            crop_right = (x_end != img_width) * (tile_overlap * scale_factor // 2)
            
            final_crop_box = (
                crop_left,
                crop_top,
                out_tile_w - crop_right,
                out_tile_h - crop_bottom,
            )
            
            cropped_output_tile = output_tile_pil.crop(final_crop_box)

            # 결과 이미지에 최종 타일 붙여넣기 위치 계산
            output_x = x_start * scale_factor + crop_left
            output_y = y_start * scale_factor + crop_top
            
            output_img.paste(cropped_output_tile, (output_x, output_y))

    return output_img


# --- 핵심 비디오 업스케일링 함수 (루프 포함) ---
def upscale_video(video_path, output_path):
    print(f"\n--- 비디오 업스케일링 시작 ---")
    print(f"입력 파일: {video_path}")
    print(f"출력 파일: {output_path}")

    try:
        session = get_upscaler(SINGLE_MODEL_PATH)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"비디오 파일 열기 실패: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        scale_factor = 4 
        new_width = frame_width * scale_factor
        new_height = frame_height * scale_factor

        # VideoWriter 객체 초기화 (코덱: mp4v)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        out = cv2.VideoWriter(output_path, fourcc, fps, (new_width, new_height), isColor=True)

        processed_frames = 0
        TILE_SIZE = 512 

        # 비디오 프레임 루프
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if processed_frames % 50 == 0:
                 print(f"처리 중... 프레임 {processed_frames}/{frame_count}")
            
            # BGR -> RGB -> PIL (전처리)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)

            # --- TensorRT 추론 실행: 타일링 함수 사용 ---
            upscaled_pil_image = tiled_upscale(
                session=session, 
                pil_image=pil_image,
                tile_size=TILE_SIZE,
                scale_factor=scale_factor
            )
            # ----------------------------------------

            # PIL -> NumPy -> BGR로 변환 후 VideoWriter에 쓰기 (후처리)
            upscaled_numpy = np.array(upscaled_pil_image)
            bgr_frame = cv2.cvtColor(upscaled_numpy, cv2.COLOR_RGB2BGR)
            out.write(bgr_frame)
            
            processed_frames += 1

        cap.release()
        out.release()
        
        if processed_frames > 0:
            print(f"\n--- 성공적으로 업스케일링 완료. {processed_frames} 프레임 처리 ---")
            print(f"결과 파일: {output_path}")
        else:
            print(f"\n--- 업스케일링 실패: 처리된 프레임 없음 ---")


    except Exception as e:
        print(f"\nFATAL ERROR: 비디오 처리 중 치명적인 오류 발생: {e}")
        traceback.print_exc()

# --- 메인 CLI 실행 블록 ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TensorRT 모델을 사용한 비디오 업스케일링 CLI 툴")
    
    parser.add_argument("input_file", type=str, help="업스케일할 비디오 파일 경로 (예: /content/video.mp4)")
    parser.add_argument("--output", type=str, default="tensorrt_upscaled_output.mp4", 
                        help="업스케일된 비디오를 저장할 경로 및 이름 (기본값: tensorrt_upscaled_output.mp4)")

    args = parser.parse_args()
    
    upscale_video(args.input_file, args.output)
