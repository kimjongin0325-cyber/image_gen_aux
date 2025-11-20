import gradio as gr
import spaces
from gradio_imageslider import ImageSlider
from image_gen_aux import UpscaleWithModel
from image_gen_aux.utils import load_image
import tempfile
import traceback
import cv2  # 비디오 처리를 위해 추가
import numpy as np
from PIL import Image

# --- Model Configuration ---
# 사용할 단일 모델 파일 경로 정의
SINGLE_MODEL_PATH = "4xDF2K_plksr_tiny_fp16_500k.onnx"
SINGLE_MODEL_NAME = "4xDF2K_plksr_tiny_fp16_500k (4x Upscale - ONNX/TensorRT)"

# --- Efficient Model Loading and Caching ---
LOADED_MODELS_CACHE = {}

def get_upscaler(model_path: str):
    """지정된 로컬 ONNX 모델 경로를 사용하여 Upscaler 객체를 로드하고 캐시합니다."""
    if model_path not in LOADED_MODELS_CACHE:
        print(f"Loading local model: {model_path}")
        try:
            # from_pretrained API가 로컬 파일 경로도 처리할 수 있도록 가정
            # image_gen_aux 라이브러리는 내부적으로 ONNX 파일을 로드할 때 
            # ONNX Runtime을 사용하며, 환경에 따라 TensorRT를 백엔드로 사용할 수 있습니다.
            upscaler = UpscaleWithModel.from_pretrained(model_path)
            LOADED_MODELS_CACHE[model_path] = upscaler
        except Exception as e:
            # 로컬 파일 로드 실패 시 오류 출력
            print(f"Error loading model from path {model_path}: {e}")
            raise gr.Error(f"Failed to load model from {model_path}")

    # to("cuda") 호출은 UpscaleWithModel 객체가 PyTorch 모델을 로드했을 때만 유효함
    # ONNX 모델은 .to("cuda") 대신 ONNX Runtime 백엔드 설정을 통해 GPU를 사용
    return LOADED_MODELS_CACHE[model_path]


# --- Core Upscaling Function (Video) ---
@spaces.GPU
def upscale_video(video_path, progress=gr.Progress(track_tqdm=True)):
    if video_path is None:
        raise gr.Error("No video uploaded. Please upload a video to upscale.")

    try:
        progress(0, desc=f"Loading model: {SINGLE_MODEL_NAME}...")
        upscaler = get_upscaler(SINGLE_MODEL_PATH)
        
        # 1. 비디오 캡처 객체 초기화
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise gr.Error("Failed to open video file.")

        # 비디오 메타데이터 추출
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 업스케일 비율 (모델 이름에서 4x를 가정)
        scale_factor = 4
        new_width = frame_width * scale_factor
        new_height = frame_height * scale_factor

        # 2. 출력 비디오 파일 설정
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            output_filepath = temp_file.name
        
        # VideoWriter 객체 초기화 (MP4 컨테이너, H.264 코덱 사용 - FFmpeg 필요)
        # Note: 'mp4v' (MPEG-4) 또는 'XVID'는 더 범용적일 수 있지만, 'H264'가 고품질에 적합함
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        out = cv2.VideoWriter(output_filepath, fourcc, fps, (new_width, new_height))

        # 3. 프레임별 처리 루프
        processed_frames = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # 진행률 업데이트
            progress((processed_frames / frame_count), desc=f"Upscaling frame {processed_frames}/{frame_count}...")
            
            # BGR (OpenCV) -> RGB (PIL) 변환
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)

            # 업스케일링 수행 (기존 이미지 로직 재사용, Tiling 유지)
            # image_gen_aux 라이브러리가 여기서 TensorRT 엔진을 사용하도록 내부적으로 최적화되어야 함
            upscaled_pil_image = upscaler(pil_image, tiling=True, tile_width=1024, tile_height=1024)

            # RGB (PIL) -> BGR (OpenCV) 변환 및 쓰기
            upscaled_numpy = np.array(upscaled_pil_image)
            bgr_frame = cv2.cvtColor(upscaled_numpy, cv2.COLOR_RGB2BGR)
            out.write(bgr_frame)
            
            processed_frames += 1

        # 4. 자원 해제
        cap.release()
        out.release()
        
        # 미리보기는 첫 프레임의 원본 및 업스케일된 버전으로 제공 (옵션)
        # 여기서는 복잡성을 줄이기 위해 단순 파일 출력만 반환
        return output_filepath

    except Exception as e:
        print(f"An error occurred: {traceback.format_exc()}")
        raise gr.Error(f"An error occurred during video processing: {e}")

def clear_outputs():
    # 비디오 출력에 맞게 수정
    return None

# --- Gradio Interface Definition ---
title = f"""<h1 align="center">TensorRT Optimized Video Upscaler ({SINGLE_MODEL_NAME})</h1>
<div align="center">
Upload a video to upscale using the dedicated ONNX model, optimized for GPU performance.<br>
This requires a Google Colab session with **TensorRT**, **FFmpeg**, and **OpenCV** installed.
</div>
"""

with gr.Blocks(delete_cache=(3600, 3600)) as demo:
    gr.HTML(title)
    with gr.Row():
        with gr.Column(scale=1):
            # 💡 입력 타입을 gr.Video로 변경
            input_video = gr.Video(label="Input Video (MP4, AVI, etc.)")
            
            # 모델 선택 드롭다운 대신 사용 모델 정보를 표시
            gr.Markdown(f"**사용 모델:** `{SINGLE_MODEL_NAME}`")
            
            run_button = gr.Button("Start Video Upscale", variant="primary")
            
        with gr.Column(scale=2):
            # 💡 이미지 슬라이더 대신 비디오 출력 컴포넌트 사용
            output_video = gr.Video(label="Upscaled Video Output (MP4)")
            
            gr.Markdown(
                "<center><i>Note: The processing time depends heavily on the video length and GPU availability.</i></center>"
            )

    # --- Event Handling ---
    run_button.click(
        fn=clear_outputs,
        inputs=None,
        outputs=[output_video], # 출력 변경
        queue=False 
    ).then(
        fn=upscale_video, # 함수 변경
        inputs=[input_video], # 입력 변경
        outputs=[output_video],
    )

# --- Pre-load the single model for a faster first-time user experience ---
try:
    print("Pre-loading single model...")
    # 단일 모델 경로를 사용하여 미리 로드합니다.
    get_upscaler(SINGLE_MODEL_PATH) 
    print("Model loaded successfully.")
except Exception as e:
    print(f"Could not pre-load the model. The app will still work. Error: {e}")

demo.queue()
demo.launch(share=False)
