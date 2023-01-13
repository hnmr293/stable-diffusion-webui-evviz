import numpy as np
from torch import nn
import gradio as gr
from PIL import Image

from modules import scripts, sd_hijack
from modules.processing import StableDiffusionProcessing, Processed, process_images
from modules.textual_inversion.textual_inversion import EmbeddingDatabase
from modules.sd_hijack import StableDiffusionModelHijack
from modules.sd_hijack_clip import FrozenCLIPEmbedderWithCustomWordsBase

class Script(scripts.Script):
    
    def __init__(self):
        super().__init__()
    
    def title(self):
        return "Emb. Vec. Visualizer"
    
    def show(self, is_img2img):
        return not is_img2img
    
    def ui(self, is_img2img):
        if is_img2img:
            return
        
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    width = gr.Slider(value=1, minimum=1, maximum=8, step=1, label="Width for each dim.")
                    height = gr.Slider(value=1, minimum=1, maximum=8, step=1, label="Height for each dim.")
                with gr.Row():
                    min = gr.Slider(value=-1, minimum=-10, maximum=0, step=0.01, label="Clamp minimum")
                    max = gr.Slider(value=1, minimum=0, maximum=10, step=0.01, label="Clamp maximum")
                no_emb = gr.Checkbox(value=False, label="DO NOT use embeddings")
        
        return [
            width,
            height,
            min,
            max,
            no_emb,
        ]
    
    def run(self,
            p: StableDiffusionProcessing,
            width: float,
            height: float,
            min: float,
            max: float,
            no_emb: bool
    ) -> Processed:
        
        w = int(width)
        h = int(height)
        assert min < max, f"invalid clamp values min={min}, max={max}"
        assert 1 <= w
        assert 1 <= h
        
        saved_db = None
        reload_db = p.do_not_reload_embeddings
        
        try:
            if no_emb:
                saved_db = sd_hijack.model_hijack.embedding_db
                sd_hijack.model_hijack.embedding_db = EmbeddingDatabase("")
                p.do_not_reload_embeddings = True
            
            return process(p, min, max, w, h)
            
        finally:
            if saved_db is not None:
                sd_hijack.model_hijack.embedding_db = saved_db
                p.do_not_reload_embeddings = reload_db

def process(
    p: StableDiffusionProcessing,
    min: float,
    max: float,
    w: int,
    h: int
) -> Processed:
    
    assert p.sd_model.model.conditioning_key == "crossattn", f"conditioning_key={p.sd_model.model.conditioning_key}" # type: ignore
    
    cond_stage_model : StableDiffusionModelHijack | FrozenCLIPEmbedderWithCustomWordsBase
    cond_stage_model = p.sd_model.cond_stage_model # type: ignore
    
    in00: nn.Module = p.sd_model.model.diffusion_model.input_blocks[0] # type: ignore
    
    if isinstance(cond_stage_model, StableDiffusionModelHijack):
        assert cond_stage_model.clip is not None
        cond_stage_model = cond_stage_model.clip
    
    uc = []
    c = []
    def fn_cond(module, inputs, outputs):
        ps, = inputs
        steps, pos, emb = outputs.shape
        assert steps == len(ps), f"inputs={len(ps)}, outputs={outputs.shape}"
        assert pos % 77 == 0, f"pos={pos}"
        assert emb == 768, f"emb={emb}"
        
        if len(uc) <= len(c):
            uc.append(list(zip(ps, outputs.cpu().numpy())))
        else:
            c.append(list(zip(ps, outputs.cpu().numpy())))
    
    n = 0
    def fn_end(module, inputs, outputs):
        nonlocal n
        n += 1
        if p.n_iter <= n:
            raise Escape()
    
    hooks = []
    try:
        hooks.append(cond_stage_model.register_forward_hook(fn_cond))
        hooks.append(in00.register_forward_hook(fn_end))
        p.batch_size = 1
        p.steps = 1
        p.width = 64
        p.height = 64
        process_images(p)
    except Escape as e:
        # ok
        assert 0 < len(c)
        assert 0 < len(uc)
        assert len(c) == len(uc)
    else:
        assert False, "must not happen"
    finally:
        for hook in hooks:
            hook.remove()
    
    image_list = []
    all_prompts = []
    all_negative_prompts = []
    all_seeds = []
    all_subseeds = []
    infotexts = []
    
    for cc, ucc in zip(c, uc):
        uc_images = create_images(ucc, min, max, w, h)
        c_images = create_images(cc, min, max, w, h)
        
        for pr, img in c_images:
            image_list.append(img)
            all_prompts.append(pr)
            all_negative_prompts.append("")
            all_seeds.append(0)
            all_subseeds.append(0)
            infotexts.append(f"Prompt: {pr}")
        
        for pr, img in uc_images:
            image_list.append(img)
            all_prompts.append("")
            all_negative_prompts.append(pr)
            all_seeds.append(0)
            all_subseeds.append(0)
            infotexts.append(f"Neg. Prompt: {pr}")
    
    return Processed(
        p,
        images_list=image_list,
        all_prompts=all_prompts,
        all_negative_prompts=all_negative_prompts,
        all_seeds = all_seeds,
        all_subseeds = all_subseeds,
        infotexts=[""] * len(image_list)
    )

def create_images(
    prompts: list[tuple[str,np.ndarray]],
    min: float,
    max: float,
    w: int,
    h: int
) -> list[tuple[str,Image.Image]]:
    
    result = []
    
    for p, t in prompts:
        # (-∞, +∞) -> [-1, +1] -> [0, 1] -> [0, 255]
        t = (t.clip(min, max) - min) / (max-min) * 256
        t = t.clip(0, 255).astype(np.uint8)
        t = t.repeat(h, axis=0).repeat(w, axis=1)
        result.append((p, Image.fromarray(t, mode="L")))
    
    return result

class Escape(RuntimeError):
    pass
