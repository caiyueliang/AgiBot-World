import os
import json
from multiprocessing import Pool
from tqdm import tqdm
import av
import cv2
import gc

# -----------------------------
SRC_DIR = "datasets_mp4/AgiBotWorld-Beta/observations"
TGT_DIR = "datasets_jpg/AgiBotWorld-Beta/observations"
TASK_JSON_DIR = "datasets_mp4/AgiBotWorld/task_info"
NPROC = 24
BATCH_SIZE = 16
# -----------------------------

def prepare_episode_list():
    all_episodes = []
    json_files = [f for f in os.listdir(TASK_JSON_DIR) if f.endswith(".json")]
    for json_file in json_files:

        task_id = os.path.basename(json_file).split("_")[1].split(".")[0]

        with open(os.path.join(TASK_JSON_DIR, json_file), "r") as f:
            data = json.load(f)
        for ep in data:
            episode_id = str(ep["episode_id"])
            action_config = ep["label_info"]["action_config"]
            if not action_config:
                continue
            start_frame = 0
            end_frame = action_config[-1]["end_frame"]
            src_video = os.path.join(SRC_DIR, task_id, episode_id, "videos", "head_color.mp4")
            tgt_dir = os.path.join(TGT_DIR, task_id, episode_id, "camera")

            if not os.path.isfile(src_video):
                print(f"[WARN] 视频不存在: {src_video}")
                continue

            # 创建 episode 根目录
            os.makedirs(tgt_dir, exist_ok=True)

            all_episodes.append({
                "src_video": src_video,
                "tgt_dir": tgt_dir,
                "start_frame": start_frame,
                "end_frame": end_frame
            })
    return all_episodes

def extract_frames_batch(episode):
    src_video = episode["src_video"]
    tgt_dir = episode["tgt_dir"]
    start_frame = episode["start_frame"]
    end_frame = episode["end_frame"]
    nframes = end_frame - start_frame

    try:
        container = av.open(src_video)
    except Exception as e:
        print(f"[ERROR] 打开视频失败: {src_video}, {e}")
        return

    stream = container.streams.video[0]
    container.seek(start_frame, stream=stream)
    frame_idx = start_frame

    with tqdm(total=nframes, desc=f"{os.path.basename(src_video)}", leave=False) as pbar:
        frames_batch = []
        frame_indices = []

        for frame in container.decode(video=0):
            if frame_idx >= end_frame:
                break

            # 检查目标 jpg 是否存在
            frame_dir = os.path.join(tgt_dir, str(frame_idx))
            target_jpg = os.path.join(frame_dir, "head_color.jpg")
            if os.path.exists(target_jpg):
                frame_idx += 1
                pbar.update(1)
                continue

            # 如果不存在才处理
            img = frame.to_ndarray(format="rgb24")
            frames_batch.append(img)
            frame_indices.append(frame_idx)
            frame_idx += 1
            pbar.update(1)

            # 达到批量大小或最后一帧，写入磁盘
            if len(frames_batch) >= BATCH_SIZE or frame_idx == end_frame:
                for idx, f in zip(frame_indices, frames_batch):
                    frame_dir = os.path.join(tgt_dir, str(idx))
                    os.makedirs(frame_dir, exist_ok=True)
                    img_bgr = cv2.cvtColor(f, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(os.path.join(frame_dir, "head_color.jpg"), img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                    del img_bgr
                frames_batch.clear()
                frame_indices.clear()
                gc.collect()

    container.close()
    gc.collect()

def main():
    all_episodes = prepare_episode_list()
    print(f"[INFO] 总共 {len(all_episodes)} 个 episode，将开始抽帧...")

    with tqdm(total=len(all_episodes), desc="Episodes") as outer_pbar:
        def callback(_):
            outer_pbar.update(1)

        with Pool(NPROC) as pool:
            for ep in all_episodes:
                pool.apply_async(extract_frames_batch, args=(ep,), callback=callback)
            pool.close()
            pool.join()

if __name__ == "__main__":
    main()
