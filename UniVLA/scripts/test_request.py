# client.py
import requests
import base64
from PIL import Image
import numpy as np

# 示例图像路径（替换为真实图像）
base_path = "/home/caiyueliang/AgiBot-World_28ad77d/UniVLA/frames/iros_clear_table_in_the_restaurant_20251028_111828/"
head_img_path = base_path + "head_00050.png"
wrist_left_img_path = base_path + "wrist_l_00050.png"
wrist_right_img_path = base_path + "wrist_r_00050.png"

# 读取图像并转为 base64
def image_to_base64(img_path):
    with open(img_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

# 构造请求
request_data = {
    "head_img": image_to_base64(head_img_path),
    "wrist_left_img": image_to_base64(wrist_left_img_path),
    "wrist_right_img": image_to_base64(wrist_right_img_path),
    "instruction": "Pick up the bowl on the table near the right arm with the right arm.;Place the bowl on the plate on the table with the right arm.",
    "state":  [-1.106, 0.529, 0.454, -1.241, 0.584, 1.419, -0.076, 0.000, 
               1.297, -0.814, -0.504, 1.077, -1.145, -1.398, 0.328, 0.000]  # 示例关节状态，16维
}

# 发送请求
response = requests.post("http://localhost:8080/act", json=request_data)

print("state:", request_data["state"])
if response.status_code == 200:
    result = response.json()
    print("Action:", result["action"])
    print("Timestamp:", result["timestamp"])
else:
    print("Error:", response.json())