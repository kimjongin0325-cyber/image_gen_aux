import cv2
import numpy as np
from PIL import Image
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

SINGLE_MODEL_PATH = "/content/image_gen_aux/plksr_4x.engine"
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

class PLKSRTinyEngine:
    def __init__(self, engine_path):
        with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()

        # 텐서 이름 미리 가져오기
        self.input_name = None
        self.output_name = None
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_location(name) == trt.TensorLocation.DEVICE:
                if "input" in name.lower() or self.input_name is None:
                    self.input_name = name
                else:
                    self.output_name = name

    def infer(self, img_np):
        h, w = img_np.shape[:2]
        h = h - (h % 4)
        w = w - (w % 4)
        if h <= 0 or w <= 0:
            return np.zeros((h*4, w*4, 3), dtype=np.uint8)
        img_np = img_np[:h, :w]

        # 입력 준비
        x = img_np.astype(np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))[np.newaxis, ...]  # (1,3,H,W)

        # 동적 shape 설정
        self.context.set_input_shape(self.input_name, x.shape)

        # 입력/출력 버퍼 정확히 재할당
        input_size = trt.volume(x.shape)
        output_shape = (1, 3, h*4, w*4)
        output_size = trt.volume(output_shape)

        # 입력 버퍼
        input_host = cuda.pagelocked_empty(input_size, np.float32)
        input_device = cuda.mem_alloc(input_host.nbytes)
        np.copyto(input_host, x.ravel())

        # 출력 버퍼
        output_host = cuda.pagelocked_empty(output_size, np.float32)
        output_device = cuda.mem_alloc(output_host.nbytes)

        # 바인딩 설정 (v3 필수!)
        self.context.set_tensor_address(self.input_name, int(input_device))
        self.context.set_tensor_address(self.output_name, int(output_device))

        # 실행
        cuda.memcpy_htod_async(input_device, input_host, self.stream)
        self.context.execute_async_v3(stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(output_host, output_device, self.stream)
        self.stream.synchronize()

        # 결과
        result = output_host.reshape(output_shape)
        result = np.clip(result[0], 0, 1).transpose(1, 2, 0)
        return (result * 255.0).round().astype(np.uint8)


def tiled_upscale(engine, pil_img, tile_size=512, overlap=32):
    w, h = pil_img.size
    out = Image.new('RGB', (w*4, h*4))
    scale = 4

    for y in range(0, h, tile_size - overlap):
        for x in range(0, w, tile_size - overlap):
            r = min(x + tile_size, w)
            b = min(y + tile_size, h)
            tile = pil_img.crop((x, y, r, b))
            out_tile = engine.infer(np.array(tile))
            out_pil = Image.fromarray(out_tile)

            cl = (x > 0) * (overlap * scale // 2)
            ct = (y > 0) * (overlap * scale // 2)
            cr = (r < w) * (overlap * scale // 2)
            cb = (b < h) * (overlap * scale // 2)

            cropped = out_pil.crop((cl, ct, out_tile.shape[1]-cr, out_tile.shape[0]-cb))
            out.paste(cropped, (x*scale + cl, y*scale + ct))
    return out


# 실행
engine = PLKSRTinyEngine(SINGLE_MODEL_PATH)
cap = cv2.VideoCapture("/content/m.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter("PLKSR_LEGEND_4K.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (w*4, h*4))

i = 0
while True:
    ret, frame = cap.read()
    if not ret: break
    if i % 10 == 0:
        print(f"프레임 처리 중... {i}")
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    up = tiled_upscale(engine, Image.fromarray(rgb))
    bgr = cv2.cvtColor(np.array(up), cv2.COLOR_RGB2BGR)
    out.write(bgr)
    i += 1

cap.release()
out.release()
print("완료!!! 한국 최초 PLKSR-tiny 실시간 4K 업스케일 영상 탄생!!!")
