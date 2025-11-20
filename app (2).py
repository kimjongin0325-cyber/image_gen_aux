import gradio as gr
import spaces
from gradio_imageslider import ImageSlider
from image_gen_aux import UpscaleWithModel
from image_gen_aux.utils import load_image
import tempfile
from PIL import Image
import traceback
import torch

# --- Model Dictionary ---
# A complete dictionary of your self-trained models.
MODELS = {
    "1xDeH264_realplksr": "Phips/1xDeH264_realplksr",
    "1xDeJPG_HAT": "Phips/1xDeJPG_HAT",
    "1xDeJPG_OmniSR": "Phips/1xDeJPG_OmniSR",
    "1xDeJPG_realplksr_otf": "Phips/1xDeJPG_realplksr_otf",
    "1xDeJPG_SRFormer_light": "Phips/1xDeJPG_SRFormer_light",
    "1xDeNoise_realplksr_otf": "Phips/1xDeNoise_realplksr_otf",
    "1xExposureCorrection_compact": "Phips/1xExposureCorrection_compact",
    "1xOverExposureCorrection_compact": "Phips/1xOverExposureCorrection_compact",
    "1xUnderExposureCorrection_compact": "Phips/1xUnderExposureCorrection_compact",
    "2xAoMR_mosr": "Phips/2xAoMR_mosr",
    "2xEvangelion_compact": "Phips/2xEvangelion_compact",
    "2xEvangelion_dat2": "Phips/2xEvangelion_dat2",
    "2xEvangelion_omnisr": "Phips/2xEvangelion_omnisr",
    "2xHFA2k_compact_multijpg": "Phips/2xHFA2k_compact_multijpg",
    "2xHFA2k_LUDVAE_compact": "Phips/2xHFA2k_LUDVAE_compact",
    "2xHFA2k_LUDVAE_SPAN": "Phips/2xHFA2k_LUDVAE_SPAN",
    "2xHFA2kAVCCompact": "Phips/2xHFA2kAVCCompact",
    "2xHFA2kAVCOmniSR": "Phips/2xHFA2kAVCOmniSR",
    "2xHFA2kAVCSRFormer_light": "Phips/2xHFA2kAVCSRFormer_light",
    "2xHFA2kCompact": "Phips/2xHFA2kCompact",
    "2xHFA2kOmniSR": "Phips/2xHFA2kOmniSR",
    "2xHFA2kReal-CUGAN": "Phips/2xHFA2kReal-CUGAN",
    "2xHFA2kShallowESRGAN": "Phips/2xHFA2kShallowESRGAN",
    "2xHFA2kSPAN": "Phips/2xHFA2kSPAN",
    "2xHFA2kSwinIR-S": "Phips/2xHFA2kSwinIR-S",
    "2xLexicaRRDBNet": "Phips/2xLexicaRRDBNet",
    "2xLexicaRRDBNet_Sharp": "Phips/2xLexicaRRDBNet_Sharp",
    "2xNomosUni_compact_multijpg": "Phips/2xNomosUni_compact_multijpg",
    "2xNomosUni_compact_multijpg_ldl": "Phips/2xNomosUni_compact_multijpg_ldl",
    "2xNomosUni_compact_otf_medium": "Phips/2xNomosUni_compact_otf_medium",
    "2xNomosUni_esrgan_multijpg": "Phips/2xNomosUni_esrgan_multijpg",
    "2xNomosUni_span_multijpg": "Phips/2xNomosUni_span_multijpg",
    "2xNomosUni_span_multijpg_ldl": "Phips/2xNomosUni_span_multijpg_ldl",
    "2xParimgCompact": "Phips/2xParimgCompact",
    "4x4xTextures_GTAV_rgt-s": "Phips/4xTextures_GTAV_rgt-s",
    "4xArtFaces_realplksr_dysample": "Phips/4xArtFaces_realplksr_dysample",
    "4xBHI_dat2_multiblur": "Phips/4xBHI_dat2_multiblur",
    "4xBHI_dat2_multiblurjpg": "Phips/4xBHI_dat2_multiblurjpg",
    "4xBHI_dat2_otf": "Phips/4xBHI_dat2_otf",
    "4xBHI_dat2_real": "Phips/4xBHI_dat2_real",
    "4xBHI_realplksr_dysample_multi": "Phips/4xBHI_realplksr_dysample_multi",
    "4xBHI_realplksr_dysample_multiblur": "Phips/4xBHI_realplksr_dysample_multiblur",
    "4xBHI_realplksr_dysample_otf": "Phips/4xBHI_realplksr_dysample_otf",
    "4xBHI_realplksr_dysample_otf_nn": "Phips/4xBHI_realplksr_dysample_otf_nn",
    "4xBHI_realplksr_dysample_real": "Phips/4xBHI_realplksr_dysample_real",
    "4xFaceUpDAT": "Phips/4xFaceUpDAT",
    "4xFaceUpLDAT": "Phips/4xFaceUpLDAT",
    "4xFaceUpSharpDAT": "Phips/4xFaceUpSharpDAT",
    "4xFaceUpSharpLDAT": "Phips/4xFaceUpSharpLDAT",
    "4xFFHQDAT": "Phips/4xFFHQDAT",
    "4xFFHQLDAT": "Phips/4xFFHQLDAT",
    "4xHFA2k": "Phips/4xHFA2k",
    "4xHFA2k_ludvae_realplksr_dysample": "Phips/4xHFA2k_ludvae_realplksr_dysample",
    "4xHFA2kLUDVAEGRL_small": "Phips/4xHFA2kLUDVAEGRL_small",
    "4xHFA2kLUDVAESRFormer_light": "Phips/4xHFA2kLUDVAESRFormer_light",
    "4xHFA2kLUDVAESwinIR_light": "Phips/4xHFA2kLUDVAESwinIR_light",
    "4xLexicaDAT2_otf": "Phips/4xLexicaDAT2_otf",
    "4xLSDIRCompact2": "Phips/4xLSDIRCompact2",
    "4xLSDIRCompact": "Phips/4xLSDIRCompact",
    "4xLSDIRCompactC3": "Phips/4xLSDIRCompactC3",
    "4xLSDIRCompactC": "Phips/4xLSDIRCompactC",
    "4xLSDIRCompactCR3": "Phips/4xLSDIRCompactCR3",
    "4xLSDIRCompactN3": "Phips/4xLSDIRCompactN3",
    "4xLSDIRCompactR3": "Phips/4xLSDIRCompactR3",
    "4xLSDIRCompactR": "Phips/4xLSDIRCompactR",
    "4xLSDIRDAT": "Phips/4xLSDIRDAT",
    "4xNature_realplksr_dysample": "Phips/4xNature_realplksr_dysample",
    "4xNomos2_hq_atd": "Phips/4xNomos2_hq_atd",
    "4xNomos2_hq_dat2": "Phips/4xNomos2_hq_dat2",
    "4xNomos2_hq_drct-l": "Phips/4xNomos2_hq_drct-l",
    "4xNomos2_hq_mosr": "Phips/4xNomos2_hq_mosr",
    "4xNomos2_otf_esrgan": "Phips/4xNomos2_otf_esrgan",
    "4xNomos2_realplksr_dysample": "Phips/4xNomos2_realplksr_dysample",
    "4xNomos8k_atd_jpg": "Phips/4xNomos8k_atd_jpg",
    "4xNomos8kDAT": "Phips/4xNomos8kDAT",
    "4xNomos8kHAT-L_bokeh_jpg": "Phips/4xNomos8kHAT-L_bokeh_jpg",
    "4xNomos8kHAT-L_otf": "Phips/4xNomos8kHAT-L_otf",
    "4xNomos8kSC": "Phips/4xNomos8kSC",
    "4xNomos8kSCHAT-L": "Phips/4xNomos8kSCHAT-L",
    "4xNomos8kSCHAT-S": "Phips/4xNomos8kSCHAT-S",
    "4xNomos8kSCSRFormer": "Phips/4xNomos8kSCSRFormer",
    "4xNomosUni_rgt_multijpg": "Phips/4xNomosUni_rgt_multijpg",
    "4xNomosUni_rgt_s_multijpg": "Phips/4xNomosUni_rgt_s_multijpg",
    "4xNomosUni_span_multijpg": "Phips/4xNomosUni_span_multijpg",
    "4xNomosUniDAT2_box": "Phips/4xNomosUniDAT2_box",
    "4xNomosUniDAT2_multijpg_ldl": "Phips/4xNomosUniDAT2_multijpg_ldl",
    "4xNomosUniDAT2_multijpg_ldl_sharp": "Phips/4xNomosUniDAT2_multijpg_ldl_sharp",
    "4xNomosUniDAT_bokeh_jpg": "Phips/4xNomosUniDAT_bokeh_jpg",
    "4xNomosUniDAT_otf": "Phips/4xNomosUniDAT_otf",
    "4xNomosWebPhoto_atd": "Phips/4xNomosWebPhoto_atd",
    "4xNomosWebPhoto_esrgan": "Phips/4xNomosWebPhoto_esrgan",
    "4xNomosWebPhoto_RealPLKSR": "Phips/4xNomosWebPhoto_RealPLKSR",
    "4xReal_SSDIR_DAT_GAN": "Phips/4xReal_SSDIR_DAT_GAN",
    "4xRealWebPhoto_v3_atd": "Phips/4xRealWebPhoto_v3_atd",
    "4xRealWebPhoto_v4_dat2": "Phips/4xRealWebPhoto_v4_dat2",
    "4xRealWebPhoto_v4_drct-l": "Phips/4xRealWebPhoto_v4_drct-l",
    "4xSSDIRDAT": "Phips/4xSSDIRDAT",
    "4xTextureDAT2_otf": "Phips/4xTextureDAT2_otf",
    "4xTextures_GTAV_rgt-s": "Phips/4xTextures_GTAV_rgt-s",
    "4xTextures_GTAV_rgt-s_dither": "Phips/4xTextures_GTAV_rgt-s_dither",
}

# --- Efficient Model Loading and Caching ---
LOADED_MODELS_CACHE = {}

def get_upscaler(model_name: str):
    if model_name not in LOADED_MODELS_CACHE:
        print(f"Loading model: {model_name}")
        LOADED_MODELS_CACHE[model_name] = UpscaleWithModel.from_pretrained(
            MODELS[model_name]
        ).to("cuda")
    return LOADED_MODELS_CACHE[model_name]

# --- Core Upscaling Function ---
@spaces.GPU
def upscale_image(image, model_selection: str, progress=gr.Progress(track_tqdm=True)):
    if image is None:
        raise gr.Error("No image uploaded. Please upload an image to upscale.")

    try:
        progress(0, desc="Loading image and model...")
        original = load_image(image)
        upscaler = get_upscaler(model_selection)

        progress(0.5, desc="Upscaling image... (this may take a moment)")
        upscaled_pil_image = upscaler(original, tiling=True, tile_width=1024, tile_height=1024)

        progress(0.9, desc="Saving result...")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
            upscaled_pil_image.save(temp_file.name, "PNG")
            output_filepath = temp_file.name

        return (original, upscaled_pil_image), output_filepath

    except Exception as e:
        print(f"An error occurred: {traceback.format_exc()}")
        raise gr.Error(f"An error occurred during processing: {e}")

def clear_outputs():
    return None, None

# --- Gradio Interface Definition ---
title = """<h1 align="center">Image Upscaler</h1>
<div align="center">
Use this Space to upscale your images with a collection of custom-trained models.<br>
This app uses the <a href="https://github.com/asomoza/image_gen_aux">Image Generation Auxiliary Tools</a> library and <a href="https://github.com/Phhofm/models">my models</a>.<br>
Tiling is fixed at 1024x1024 for optimal performance. An <a href="https://huggingface.co/spaces/Phips/Upscaler/resolve/main/input_example1.png">example input image</a> is available to try.
</div>
"""

with gr.Blocks(delete_cache=(3600, 3600)) as demo:
    gr.HTML(title)
    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(type="pil", label="Input Image")
            model_selection = gr.Dropdown(
                choices=list(MODELS.keys()),
                value="4xBHI_dat2_real",
                label="Model (alphabetically sorted)",
            )
            run_button = gr.Button("Upscale", variant="primary")
            
        with gr.Column(scale=2):
            result_slider = ImageSlider(
                interactive=False,
                label="Compare Original vs. Upscaled",
                show_label=True,
                show_download_button=False 
            )
            
            # --- THIS IS THE NEW ADDITION ---
            # Add a descriptive note to guide the user about the preview vs. download quality.
            gr.Markdown(
                "<center><i>Note: The slider above shows a web-optimized preview. For the full-quality, lossless PNG, please use the download button below.</i></center>"
            )
            
            download_output = gr.File(label="Download Full-Quality Upscaled Image (Lossless PNG)")

    # --- Event Handling ---
    run_button.click(
        fn=clear_outputs,
        inputs=None,
        outputs=[result_slider, download_output],
        queue=False 
    ).then(
        fn=upscale_image,
        inputs=[input_image, model_selection],
        outputs=[result_slider, download_output],
    )

# --- Pre-load the default model for a faster first-time user experience ---
try:
    print("Pre-loading default model...")
    get_upscaler("4xNomosWebPhoto_RealPLKSR")
    print("Default model loaded successfully.")
except Exception as e:
    print(f"Could not pre-load the default model. The app will still work. Error: {e}")

# Queueing is essential for public-facing apps to handle concurrent users.
demo.queue()
demo.launch(share=False)