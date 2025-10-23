import av

# 创建容器和视频流
container = av.open('./test.mp4', mode='w', format='mp4')
stream = container.add_stream('libx264', rate=30)
stream.width = 640
stream.height = 480
stream.pix_fmt = 'yuv420p'

# 写入30帧
for i in range(30):
    frame = av.VideoFrame(640, 480, 'rgb24')
    packet = stream.encode(frame)  # 编码帧
    if packet:
        container.mux(packet)

# ❗关键：flush 编码器中的剩余帧
remaining_packets = stream.encode(None)
while remaining_packets is not None:
    if isinstance(remaining_packets, list):
        packets = remaining_packets
    else:
        packets = [remaining_packets]
    for packet in packets:
        if packet:
            container.mux(packet)
    remaining_packets = stream.encode(None)

# 关闭容器
container.close()

print("Test video saved.")