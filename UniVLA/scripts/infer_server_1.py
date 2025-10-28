# app.py
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import cv2
import numpy as np
from PIL import Image
import base64
from io import BytesIO
import torch

from dataclasses import dataclass
from typing import Union
from pathlib import Path

import sys

# 添加你的项目路径
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent))

from PIL import Image
from dataclasses import dataclass
from typing import Union
from datetime import datetime
import cv2
import numpy as np
import draccus
from experiments.robot.geniesim.genie_model import WrappedGenieEvaluation, WrappedModel

# 初始化 FastAPI 应用
app = FastAPI(title="Genie Robot Policy Inference API", version="0.1")

# ==============================
# 配置类（与你原来的 GenerateConfig 一致）
# ==============================

@dataclass
class GenerateConfig:
    model_family: str = "openvla"
    pretrained_checkpoint: Union[str, Path] = "checkpoints/finetuned"
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    center_crop: bool = False
    local_log_dir: str = "./experiments/eval_logs"
    seed: int = 7
    action_decoder_path: str = "checkpoints/finetuned/action_decoder.pt"
    window_size: int = 30
    n_layers: int = 2
    hidden_dim: int = 1024
    with_proprio: bool = True
    smooth: bool = False
    balancing_factor: float = 0.1
    task_name: str = "iros_stamp_the_seal"


# ==============================
# 全局变量：模型加载
# ==============================

model_loaded = False
policy = None


def load_model():
    global policy, model_loaded
    if not model_loaded:
        print("Loading model...")
        cfg = GenerateConfig()
        print(f"[cfg] {cfg}")
        wrapped_model = WrappedModel(cfg)
        wrapped_model.cuda()
        wrapped_model.eval()
        policy = WrappedGenieEvaluation(cfg, wrapped_model)
        model_loaded = True
        print("Model loaded successfully.")
    return policy


@app.on_event("startup")
def startup_event():
    load_model()


# ==============================
# 输入数据模型定义
# ==============================

class InferenceRequest(BaseModel):
    head_img: str  # base64 encoded image
    wrist_left_img: Optional[str] = None
    wrist_right_img: Optional[str] = None
    instruction: str
    state: Optional[List[float]] = None  # joint positions


class InferenceResponse(BaseModel):
    action: List[float]
    timestamp: str
    status: str


# ==============================
# 工具函数
# ==============================

def base64_to_pil(b64_str: str) -> Image.Image:
    try:
        image_data = base64.b64decode(b64_str)
        image = Image.open(BytesIO(image_data))
        return image
    except Exception as e:
        raise ValueError(f"Invalid base64 image: {e}")


def resize_img(img, width=224, height=224):
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
    return np.array(img)


# ==============================
# 推理接口
# ==============================

@app.post("/infer", response_model=InferenceResponse)
async def infer(request: InferenceRequest):
    try:
        # 解码图像
        head_pil = base64_to_pil(request.head_img)
        head_rgb = np.array(head_pil)

        wrist_l_rgb = None
        wrist_r_rgb = None

        if request.wrist_left_img:
            wrist_l_pil = base64_to_pil(request.wrist_left_img)
            wrist_l_rgb = np.array(wrist_l_pil)

        if request.wrist_right_img:
            wrist_r_pil = base64_to_pil(request.wrist_right_img)
            wrist_r_rgb = np.array(wrist_r_pil)

        # 获取语言指令
        lang = request.instruction

        # 获取本体感觉（可选）
        state = np.array(request.state) if request.state else None

        # 获取模型
        policy = load_model()

        # 执行推理
        with torch.no_grad():
            if state is not None and policy.cfg.with_proprio:
                action = policy.step(head_rgb, wrist_l_rgb, wrist_r_rgb, lang, state)
            else:
                action = policy.step(head_rgb, wrist_l_rgb, wrist_r_rgb, lang)

        action_list = action.tolist() if isinstance(action, np.ndarray) else action

        return InferenceResponse(
            action=action_list,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            status="success"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


# ==============================
# 健康检查接口
# ==============================

@app.get("/health")
async def health_check():
    return {"status": "ok", "model_loaded": model_loaded}



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
