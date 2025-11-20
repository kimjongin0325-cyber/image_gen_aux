import argparse
import traceback
import cv2
import numpy as np
import os
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit # PyCUDA Context initialization
import time 
from typing import Dict, Optional, List, Tuple
import threading
import queue
import sys

# --- Configuration ---
# NOTE: 이 경로는 FP16으로 빌드된 엔진 파일 경로로 변경해야 합니다!
SINGLE_MODEL_PATH = "/content/image_gen_aux/plksr_4x_fp16.engine" 
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
FRAME_QUEUE_SIZE = 10 # Buffer size for CPU decoding/encoding
TILE_SIZE = 960 # Tiling size
OVERLAP = 16 # Overlap size for tiling

# --- PLKSRTinyEngine Class (Robust Initialization) ---
class PLKSRTinyEngine:
    """TensorRT engine wrapper handling dynamic I/O and PyCUDA memory management."""
    def __init__(self, engine_path):
        
        # 안전한 초기화를 위해 모든 주요 속성을 None으로 설정합니다.
        self.engine = None
        self.context = None
        self.stream = None
        self.input_device: Optional[cuda.DeviceAllocation] = None
        self.output_device: Optional[cuda.DeviceAllocation] = None
        self.current_input_size = 0
        self.current_output_size = 0
        self.dtype_size = 2 # FP16은 2 bytes

        if not os.path.exists(engine_path):
            raise FileNotFoundError(f"TensorRT engine file not found: {engine_path}")

        with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
            
        # 엔진 로딩이 성공한 후에만 CUDA 객체를 생성합니다.
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        
        # Tensor names
        self.input_name = None
        self.output_name = None
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            # Check if the tensor is on the device (required for I/O in TRT 8+)
            if self.engine.get_tensor_location(name) == trt.TensorLocation.DEVICE:
                if "input" in name.lower() or self.input_name is None:
                    self.input_name = name
                elif "output" in name.lower() or self.output_name is None:
                    self.output_name = name
        
        if not self.input_name or not self.output_name:
            raise ValueError("Could not find input/output tensor names in the engine.")
        
        # Check if the engine is truly FP16 by checking the output precision
        self.output_dtype = self.engine.get_tensor_dtype(self.output_name)
        if self.output_dtype != trt.DataType.HALF:
            print(f"[WARNING] Engine output dtype is {self.output_dtype}, expected FP16 (HALF). Allocation uses 2 bytes.", file=sys.stderr)
        
    def __del__(self):
        """Clean up allocated device memory."""
        # self.stream이 생성되었는지 확인합니다.
        if self.stream:
            try:
                self.stream.synchronize()
            except Exception:
                pass
        
        if self.input_device:
            try:
                self.input_device.free()
            except Exception:
                pass
        if self.output_device:
            try:
                self.output_device.free()
            except Exception:
                pass

    def _allocate_buffers(self, input_size: int, output_size: int):
        """Allocates or reallocates device buffers if size changes. Uses 2 bytes (FP16) for both I/O."""
        
        # Reallocate input buffer if needed. Using 2 bytes for float16
        if input_size != self.current_input_size:
            if self.input_device:
                self.input_device.free()
            self.input_device = cuda.mem_alloc(input_size * self.dtype_size) # FP16 = 2 bytes
            self.current_input_size = input_size
            print(f"[DEBUG] Reallocated Input Device Buffer: {input_size * self.dtype_size} bytes")

        # Reallocate output buffer if needed. Assuming FP16 output (2 bytes)
        if output_size != self.current_output_size:
            if self.output_device:
                self.output_device.free()
            self.output_device = cuda.mem_alloc(output_size * self.dtype_size) # FP16 = 2 bytes
            self.current_output_size = output_size
            print(f"[DEBUG] Reallocated Output Device Buffer: {output_size * self.dtype_size} bytes")
            
    # Function now returns the actual calculated output shape from TRT context
    def infer_async(self, img_np: np.ndarray, output_host: np.ndarray) -> Tuple[int, int, int, int]:
        """
        Asynchronously infers a single image tile and schedules result copy to output_host (FP16).
        Returns the actual calculated output shape.
        """
        
        # 1. Modcrop (ensure 4x divisibility)
        h, w = img_np.shape[:2]
        h_mod = h - (h % 4)
        w_mod = w - (w % 4)
        
        if h_mod <= 0 or w_mod <= 0:
             raise ValueError("Input tile size is too small after modcrop to be processed.")
             
        img_np = img_np[:h_mod, :w_mod]
        
        # 2. Preprocessing: Normalization and NCHW conversion
        x = img_np.astype(np.float16) / 255.0 # (FP16 Data Type)
        x = np.transpose(x, (2, 0, 1))[np.newaxis, ...]  # (1, 3, H, W)
        
        # 3. Set dynamic shape
        input_shape = x.shape
        self.context.set_input_shape(self.input_name, input_shape)

        # 4. CRITICAL: Get actual output shape from TensorRT context
        output_shape = self.context.get_tensor_shape(self.output_name)
        
        # 5. Calculate buffer size
        input_size = trt.volume(input_shape)
        output_size = trt.volume(output_shape)
        
        # Verify output_host size matches the required output_size
        if output_host.size != output_size:
            raise RuntimeError(f"Host buffer size mismatch. Expected: {output_size}, Actual: {output_host.size}. This is a critical error in tiling setup.")
        
        # Host input buffer (CPU) - Page-locked for fast asynchronous transfer (FP16)
        input_host = cuda.pagelocked_empty(input_size, np.float16) # (FP16 Data Type)
        np.copyto(input_host, x.ravel())

        # Allocate/Reallocate Device Buffers (FP16 based allocation)
        self._allocate_buffers(input_size, output_size)
        
        # 6. Set Bindings (v3 required)
        self.context.set_tensor_address(self.input_name, int(self.input_device))
        self.context.set_tensor_address(self.output_name, int(self.output_device))

        # 7. Execute (Asynchronous schedule)
        # HtoD Copy
        cuda.memcpy_htod_async(self.input_device, input_host, self.stream)
        # Inference
        if not self.context.all_tensors_know_shapes:
             raise RuntimeError("TensorRT context does not know all output shapes after setting input shape.")
             
        self.context.execute_async_v3(stream_handle=self.stream.handle) 
        # DtoH Copy schedule (to be copied to output_host, which is now np.float16)
        cuda.memcpy_dtoh_async(output_host, self.output_device, self.stream)
        
        return output_shape # Return the calculated shape for later reshaping


# --- Tiling function (Adjusted to use FP16 output host buffer) ---
def tiled_upscale(engine: PLKSRTinyEngine, img_np: np.ndarray, tile_size: int = TILE_SIZE, overlap: int = OVERLAP) -> np.ndarray:
    """
    Performs NumPy-based Tiling using asynchronous infer_async. 
    (Single synchronization per frame)
    """
    h, w, c = img_np.shape
    scale = 4
    
    new_h, new_w = h * scale, w * scale
    out_np = np.zeros((new_h, new_w, c), dtype=np.uint8)

    overlap_scaled = overlap * scale 
    
    host_output_buffers: List[np.ndarray] = []
    tile_info: List[Dict] = []

    for y in range(0, h, tile_size - overlap):
        for x in range(0, w, tile_size - overlap):
            r_x = min(x + tile_size, w)
            r_y = min(y + tile_size, h)
            
            # Tile dimension check
            tile_h, tile_w = r_y - y, r_x - x
            if tile_h < 4 or tile_w < 4: 
                continue 

            tile = img_np[y:r_y, x:r_x, :]

            # Pre-calculate the required size for the host buffer
            output_size = (tile_h * scale) * (tile_w * scale) * c
            
            # Use np.float16 for output host buffer
            output_host = cuda.pagelocked_empty(output_size, np.float16) # (FP16 Data Type)
            host_output_buffers.append(output_host)
            
            # Asynchronously schedule inference for this tile and get the actual output shape
            try:
                # The returned shape is (N, C, H, W)
                actual_output_shape = engine.infer_async(tile, output_host)
            except RuntimeError as e:
                print(f"[ERROR] Failed to infer tile at ({x}, {y}): {e}", file=sys.stderr)
                # If an error occurs, skip this tile
                host_output_buffers.pop()
                continue
            
            tile_info.append({
                'x': x, 'y': y, 'r_x': r_x, 'r_y': r_y,
                'out_shape': actual_output_shape, # Store the actual shape from TRT
                'output_host_index': len(host_output_buffers) - 1
            })

    # Synchronize the stream once all tiles for the frame are scheduled
    engine.stream.synchronize() 

    # Post-process and blend tiles
    for info in tile_info:
        x, y, r_x, r_y = info['x'], info['y'], info['r_x'], info['r_y']
        actual_output_shape = info['out_shape'] # (N, C, H_out, W_out)
        tile_out_h = actual_output_shape[2]
        tile_out_w = actual_output_shape[3]
        
        output_host = host_output_buffers[info['output_host_index']]
        
        # Post-processing steps: 
        # 1. Reshape using the actual shape obtained from TRT
        result = output_host.reshape(actual_output_shape)
        
        # 2. Reshape and Transpose (The result array is already np.float16)
        out_tile = result[0].transpose(1, 2, 0) # CHW -> HWC
        # 3. Clip and Denormalize (FP16 output)
        out_tile = np.clip(out_tile, 0, 1) 
        out_tile = (out_tile * 255.0).round().astype(np.uint8)
        
        # Overlap cropping calculation (based on original size)
        cl = (x > 0) * overlap * scale // 2
        ct = (y > 0) * overlap * scale // 2
        cr = (r_x < w) * overlap * scale // 2
        cb = (r_y < h) * overlap * scale // 2

        cropped_np = out_tile[ct : tile_out_h - cb, cl : tile_out_w - cr, :]
        
        # Placement calculation
        out_x_pos_start = x * scale + cl
        out_y_pos_start = y * scale + ct
        out_x_pos_end = out_x_pos_start + cropped_np.shape[1]
        out_y_pos_end = out_y_pos_start + cropped_np.shape[0]

        # Splice into the final output array
        out_np[out_y_pos_start:out_y_pos_end, out_x_pos_start:out_x_pos_end, :] = cropped_np
            
    return out_np

# --- 1. VideoDecoder Thread (Producer) ---

class VideoDecoder(threading.Thread):
    """Reads video frames, converts BGR -> RGB, and puts them into the input queue."""
    def __init__(self, video_path, frame_queue):
        super().__init__(name="VideoDecoder")
        self.video_path = video_path
        self.frame_queue = frame_queue
        self.stop_event = threading.Event()
        self.frames_read = 0

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"\n[DECODER ERROR] Failed to open video file: {self.video_path}", file=sys.stderr)
            self.stop_event.set()
            return

        while not self.stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                break
            
            # BGR -> RGB conversion is done here
            rgb_np = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            try:
                # Put frame into queue, waiting if queue is full (Backpressure)
                self.frame_queue.put((self.frames_read, rgb_np), timeout=1) 
                self.frames_read += 1
            except queue.Full:
                continue 
            except Exception:
                break 

        cap.release()
        # End signal
        self.frame_queue.put((-1, None)) 
        print("\n[DECODER] Video decoding complete.")

# --- 3. VideoWriter Thread (Final Consumer) ---

class VideoWriterThread(threading.Thread):
    """Consumes upscaled frames and encodes/writes them to the video file."""
    def __init__(self, output_path, width, height, fps, output_queue):
        super().__init__(name="VideoWriter")
        self.output_path = output_path
        self.output_queue = output_queue
        self.stop_event = threading.Event()
        
        # VideoWriter initialization (using mp4v codec for compatibility)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        self.out_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=True)
        if not self.out_writer.isOpened():
            print(f"\n[WRITER ERROR] Failed to initialize Video Writer: {output_path}", file=sys.stderr)
            self.stop_event.set()

    def run(self):
        frames_written = 0
        while not self.stop_event.is_set():
            try:
                # Wait for the next frame from the queue
                frame_idx, bgr_np = self.output_queue.get(timeout=5)
            except queue.Empty:
                continue

            if bgr_np is None:
                # End signal
                break 

            # File write (CPU encoding task)
            self.out_writer.write(bgr_np)
            frames_written += 1
            self.output_queue.task_done()

        self.out_writer.release()
        print(f"\n[WRITER] Video writing complete. Total {frames_written} frames.")


def upscale_video_threaded(video_path, output_path):
    print(f"\n--- 비디오 업스케일링 시작 (3-Stage Threaded Pipeline, FP16 Optimized) ---")
    print(f"입력 파일: {video_path}")
    print(f"출력 파일: {output_path}")

    try:
        # 1. Information extraction and initialization
        cap_info = cv2.VideoCapture(video_path)
        if not cap_info.isOpened():
            raise FileNotFoundError(f"Failed to open video file: {video_path}")

        fps = cap_info.get(cv2.CAP_PROP_FPS)
        w = int(cap_info.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap_info.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap_info.release()
        
        new_w = w * 4
        new_h = h * 4

        # Initialize TensorRT engine
        engine = PLKSRTinyEngine(SINGLE_MODEL_PATH)
        
        # Define queues
        input_queue = queue.Queue(maxsize=FRAME_QUEUE_SIZE) # Decoder -> Inferencer
        output_queue = queue.Queue(maxsize=FRAME_QUEUE_SIZE) # Inferencer -> Writer
        
        # 2. Start Producer/Consumer threads
        decoder_thread = VideoDecoder(video_path, input_queue)
        writer_thread = VideoWriterThread(output_path, new_w, new_h, fps, output_queue)
        
        decoder_thread.start()
        writer_thread.start()
        print(f"[MAIN] 3-Stage Pipeline started. Input/Output Queue size: {FRAME_QUEUE_SIZE}")

        # 3. Inferencer Loop (Main Thread, acting as intermediary)
        i = 0
        total_frame_processing_time = 0.0
        
        while True:
            frame_start_time = time.time()

            try:
                # Stage 1: Wait for the next frame from the decoder queue
                frame_idx, rgb_np = input_queue.get(timeout=5)
            except queue.Empty:
                if not decoder_thread.is_alive():
                    break
                continue

            if rgb_np is None:
                # End signal from decoder
                input_queue.task_done()
                break 

            if i % 10 == 0:
                print(f"Inferring frame... {i}")
            
            # Stage 2: GPU Inference (Core task of the Inferencer)
            try:
                up_np = tiled_upscale(engine, rgb_np)
            except Exception as e:
                print(f"\n[FATAL] Tile Upscale failed for frame {i}: {e}", file=sys.stderr)
                traceback.print_exc()
                # Skip the failed frame but keep the pipeline running if possible
                input_queue.task_done()
                i += 1
                continue
            
            # Stage 3: RGB -> BGR conversion and pass to output queue (Writer handles encoding)
            bgr_np = cv2.cvtColor(up_np, cv2.COLOR_RGB2BGR)
            
            try:
                output_queue.put((frame_idx, bgr_np), timeout=1)
            except queue.Full:
                 # Should rarely happen, but wait if the Writer is stalled
                 output_queue.put((frame_idx, bgr_np))
                
            frame_end_time = time.time()
            elapsed = frame_end_time - frame_start_time
            total_frame_processing_time += elapsed
            
            if i % 10 == 9:
                print(f"  > Frame {i} Inference/Transfer Time: {elapsed:.3f}s")
            
            i += 1
            input_queue.task_done() # Signal input queue task completion

        # 4. Cleanup and signal termination
        print("\n[MAIN] Inference complete. Waiting for Writer thread to finish...")
        # Send termination signal to Writer thread
        output_queue.put((-1, None)) 

        # Wait for all threads to join
        decoder_thread.join()
        writer_thread.join()
        
        if i > 0:
            avg_frame_time = total_frame_processing_time / i
            avg_fps = i / total_frame_processing_time
            print("\n================================================")
            print(f"Successfully processed {i} frames (3-Stage Threaded, FP16 Optimized).")
            print(f"Total Frame Inference/Transfer Time (Total Infer Time): {total_frame_processing_time:.2f}s")
            print(f"Average Processing Time per Frame: {avg_frame_time:.3f}s")
            print(f"Average Processing Speed (Average FPS): {avg_fps:.2f} FPS")
            print("\nCPU Encoding/Writing also completed in parallel. Check the final FP16 optimized FPS results.")
            print("================================================")
        else:
            print("No frames were processed.")


    except Exception as e:
        # 이 부분이 파일이 없을 때 발생하는 오류의 주 원인입니다.
        print(f"\nFATAL ERROR: A critical error occurred during video processing: {e}", file=sys.stderr)
        traceback.print_exc()

# --- Main CLI Execution Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLI tool for video upscaling using TensorRT engine.")
    
    parser.add_argument("input_file", type=str, nargs='?', default="/content/m.mp4",
                        help="Path to the video file to upscale (e.g., /content/m.mp4)")
    parser.add_argument("--output", type=str, default="PLKSR_LEGEND_4K_optimized_fp16.mp4", 
                        help="Path and name to save the upscaled video (Default: PLKSR_LEGEND_4K_optimized_fp16.mp4)")

    args = parser.parse_args()
    
    if args.input_file == "/content/m.mp4" and not os.path.exists(args.input_file):
        print("Warning: Default input file (/content/m.mp4) does not exist. Please enter the actual video file path.", file=sys.stderr)
        print("Example: python clit16.py /content/my_video.mp4", file=sys.stderr)
        sys.exit(1)
        
    upscale_video_threaded(args.input_file, args.output)
