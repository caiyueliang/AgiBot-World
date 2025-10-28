# model_service.py
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import base64
from io import BytesIO
import json

import torch
import numpy as np
from PIL import Image
import cv2
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# 添加路径
sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent))

from dataclasses import dataclass
from typing import Union
import draccus
from experiments.robot.geniesim.genie_model import WrappedGenieEvaluation, WrappedModel

os.environ["TOKENIZERS_PARALLELISM"] = "false"


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


class ModelInferenceService:
    def __init__(self, config: GenerateConfig):
        self.config = config
        self.model = None
        self.policy = None
        self.is_loaded = False
        
    def load_model(self):
        """加载模型"""
        try:
            print("Loading model...")
            wrapped_model = WrappedModel(self.config)
            wrapped_model.cuda()
            wrapped_model.eval()
            self.policy = WrappedGenieEvaluation(self.config, wrapped_model)
            self.is_loaded = True
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def preprocess_image(self, image_data: bytes, target_width: int = 224, target_height: int = 224) -> np.ndarray:
        """预处理图像"""
        try:
            # 将bytes转换为numpy数组
            nparr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # 调整尺寸
            resized_img = cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_AREA)
            return resized_img
        except Exception as e:
            print(f"Error preprocessing image: {e}")
            raise
    
    def inference(
        self,
        head_image: np.ndarray,
        left_wrist_image: np.ndarray,
        right_wrist_image: np.ndarray,
        language_instruction: str,
        joint_state: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """执行推理"""
        if not self.is_loaded:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        try:
            if self.config.with_proprio and joint_state is not None:
                state = np.array(joint_state)
                action = self.policy.step(head_image, left_wrist_image, right_wrist_image, language_instruction, state)
            else:
                action = self.policy.step(head_image, left_wrist_image, right_wrist_image, language_instruction)
            
            # 将numpy数组转换为列表
            if isinstance(action, np.ndarray):
                action = action.tolist()
            
            return {
                "action": action,
                "status": "success",
                "message": "Inference completed successfully"
            }
        except Exception as e:
            print(f"Inference error: {e}")
            raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


# 创建FastAPI应用
app = FastAPI(title="Genie Model Inference Service", version="1.0.0")

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局模型服务实例
model_service = None


class InferenceRequest(BaseModel):
    language_instruction: str
    joint_state: Optional[List[float]] = None


class InferenceResponse(BaseModel):
    action: List[float]
    status: str
    message: str


@app.on_event("startup")
async def startup_event():
    """启动时加载模型"""
    global model_service
    try:
        config = GenerateConfig()
        print(f"[startup_event] config: {config}")
        model_service = ModelInferenceService(config)
        model_service.load_model()
        print("Model service started successfully!")
    except Exception as e:
        print(f"Failed to start model service: {e}")
        raise


@app.get("/")
async def root():
    return {"message": "Genie Model Inference Service", "status": "running"}


@app.get("/health")
async def health_check():
    """健康检查"""
    if model_service and model_service.is_loaded:
        return {"status": "healthy", "model_loaded": True}
    else:
        return {"status": "unhealthy", "model_loaded": False}


@app.post("/inference", response_model=InferenceResponse)
async def inference(
    head_image: UploadFile = File(...),
    left_wrist_image: UploadFile = File(...),
    right_wrist_image: UploadFile = File(...),
    language_instruction: str = Form(...),
    joint_state: str = Form(None)
):
    """
    执行模型推理
    
    Args:
        head_image: 头部摄像头图像
        left_wrist_image: 左手腕摄像头图像  
        right_wrist_image: 右手腕摄像头图像
        language_instruction: 语言指令
        joint_state: 关节状态（可选，JSON字符串格式）
    """
    try:
        # 解析关节状态
        joint_state_list = None
        if joint_state:
            joint_state_list = json.loads(joint_state)
        
        # 读取并预处理图像
        head_img_data = await head_image.read()
        left_wrist_img_data = await left_wrist_image.read()
        right_wrist_img_data = await right_wrist_image.read()
        
        head_img = model_service.preprocess_image(head_img_data)
        left_wrist_img = model_service.preprocess_image(left_wrist_img_data)
        right_wrist_img = model_service.preprocess_image(right_wrist_img_data)
        
        # 执行推理
        result = model_service.inference(
            head_image=head_img,
            left_wrist_image=left_wrist_img,
            right_wrist_image=right_wrist_img,
            language_instruction=language_instruction,
            joint_state=joint_state_list
        )
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Service error: {str(e)}")


@app.post("/inference_base64")
async def inference_base64(request: Dict[str, Any]):
    """
    使用base64编码的图像进行推理
    
    Args:
        request: 包含以下字段的字典
            - head_image: base64编码的头部图像
            - left_wrist_image: base64编码的左手腕图像
            - right_wrist_image: base64编码的右手腕图像  
            - language_instruction: 语言指令
            - joint_state: 关节状态列表（可选）
    """
    try:
        # 解码base64图像
        head_img_data = base64.b64decode(request["head_image"])
        left_wrist_img_data = base64.b64decode(request["left_wrist_image"])
        right_wrist_img_data = base64.b64decode(request["right_wrist_image"])
        
        head_img = model_service.preprocess_image(head_img_data)
        left_wrist_img = model_service.preprocess_image(left_wrist_img_data)
        right_wrist_img = model_service.preprocess_image(right_wrist_img_data)
        
        # 执行推理
        result = model_service.inference(
            head_image=head_img,
            left_wrist_image=left_wrist_img,
            right_wrist_image=right_wrist_img,
            language_instruction=request["language_instruction"],
            joint_state=request.get("joint_state")
        )
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Service error: {str(e)}")


def get_instruction(task_name: str) -> str:
    """获取任务指令（从原代码复制）"""
    instructions = {
        "iros_clear_the_countertop_waste": "Pick up the yellow functional beverage can on the table with the left arm.;Threw the yellow functional beverage can into the trash can with the left arm.;Pick up the green carbonated beverage can on the table with the right arm.;Threw the green carbonated beverage can into the trash can with the right arm.",
        "iros_restock_supermarket_items": "Pick up the brown plum juice from the restock box with the right arm.;Place the brown plum juice on the shelf where the brown plum juice is located with the right arm.",
        # ... 其他任务指令
    }
    return instructions.get(task_name, "")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)