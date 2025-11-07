#!/usr/bin/env python3
"""
多目+关节状态推理客户端（支持命令行传参）
"""
import argparse
import base64
import requests
import time
from pathlib import Path

# -------------------- 命令行解析 --------------------
def parse_args():
    parser = argparse.ArgumentParser(description="VLA 推理客户端")
    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:8080/act",
        help="推理服务完整 URL，默认 http://localhost:8080/act",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="可选 JWT/Bearer token，传入后会自动加到请求头",
    )
    return parser.parse_args()

# -------------------- 图像工具 --------------------
def image_to_base64(img_path: str) -> str:
    """读取图像并转为 base64 字符串"""
    with open(img_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

# -------------------- 主流程 --------------------
def main():
    args = parse_args()

    # 1. 本地图像路径（按需修改）
    base_path = "./images/iros_clear_table_in_the_restaurant_20251028_111828/"
    head_img_path = base_path + "head_00050.png"
    wrist_left_img_path = base_path + "wrist_l_00050.png"
    wrist_right_img_path = base_path + "wrist_r_00050.png"

    # 2. 构造 JSON 请求体
    request_data = {
        "image": image_to_base64(head_img_path),
        "wrist_image_l": image_to_base64(wrist_left_img_path),
        "wrist_image_r": image_to_base64(wrist_right_img_path),
        "prompt": "Pick up the bowl on the table near the right arm with the right arm.",
        "state": [
            -1.106, 0.529, 0.454, -1.241, 0.584, 1.419, -0.076, 0.000,
            1.297, -0.814, -0.504, 1.077, -1.145, -1.398, 0.328, 0.000,
        ],
    }

    # 3. 构造请求头
    headers = {"Content-Type": "application/json"}
    if args.token:
        headers["Authorization"] = f"{args.token}"

    # 4. 发送请求
    print(f"[INFO] POST -> {args.url}")
    start = time.time()
    response = requests.post(args.url, json=request_data, headers=headers)
    end = time.time()
    print(f"[INFO] Response time: {end - start:.4f} seconds")

    # 5. 处理返回
    print("state:", request_data["state"])
    if response.status_code == 200:
        result = response.json()
        print("Action:", result["action"])
        print("Timestamp:", result["timestamp"])
    else:
        print("Error:", response.json())

if __name__ == "__main__":
    main()