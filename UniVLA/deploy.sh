# 需要ubuntu环境22.04
cd
# conda 安装文件下载
wget -O Miniconda3-latest-Linux-x86_64.sh   https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-x86_64.sh
# conda 源配置
cat >> ~/.condarc <<'EOF'
channels:
  - defaults
show_channel_urls: true
default_channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2
custom_channels:
  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
EOF
# conda 安装
bash Miniconda3-latest-Linux-x86_64.sh
# baseline python环境配置（包括了gdk运行环境）
conda create -n vla python=3.10 pip
conda activate vla
# pip 源配置
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
pip config set install.trusted-host mirrors.aliyun.com
pip install -r ./requirements.txt
# ros2 (humble) 安装
wget http://fishros.com/install -O fishros && . fishros
# gdk 安装 (需要连接G1机器人)、启用
curl -sSL http://10.42.0.101:8849/install.sh | bash
cd a2d_sdk
source env.sh
robot-service -s -c ./conf/copilot.pbtxt