import av
container = av.open('./test.mp4', mode='w', format='mp4')
stream = container.add_stream('libx264', rate=30)
stream.width = 640
stream.height = 480
stream.pix_fmt = 'yuv420p'

for i in range(30):  # 写入1秒视频
    frame = av.VideoFrame(640, 480, 'rgb24')
    packet = stream.encode(frame)
    if packet:
        container.mux(packet)

container.close()
print("Test video saved.")