import pickle
import random
import time
import pandas as pd
import numpy as np
from scipy.spatial import ConvexHull
import face_recognition
import cv2
import dlib
from skimage.metrics import structural_similarity as compare_ssim
import os


def generate_diff(img_1, ima_2):
    d, a = compare_ssim(img_1, ima_2, channel_axis=-1, full=True)
    a = 1 - a
    diff = (a * 255).astype(np.uint8)
    diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    return diff


def generate_record(label, crop, landmark, diff):
    if landmark:
        landmark = landmark[0]
        landmark["lip"] = np.concatenate((landmark["bottom_lip"], landmark["top_lip"]), axis=0).tolist()
        landmark["eyes"] = np.concatenate((landmark["left_eye"], landmark["right_eye"]), axis=0).tolist()
        landmark["nose"] = np.concatenate((landmark["nose_bridge"], landmark["nose_tip"]), axis=0).tolist()
        landmark["outline"] = np.array(np.concatenate((
            landmark["chin"],
            landmark["lef t_eyebrow"],
            landmark["right_eyebrow"]), axis=0)
        )
        landmark["outline"] = landmark["outline"][ConvexHull(landmark["outline"]).vertices].tolist()
        landmark = [landmark]
    else:
        landmark = []

    return {
        "label": label,
        "crop": crop.tobytes(),
        "crop_shape": crop.shape,
        "diff": diff.tobytes(),
        "diff_shape": diff.shape,
        "landmark": landmark
    }


def get_distributed_frames(videos, n=10, jitter=True):
    t0 = time.perf_counter()
    name = videos["real"].split("/")[-1].replace(".mp4", "")
    print(f"Processing video {name}")
    real_cap = cv2.VideoCapture(videos["real"])
    fake_cap = [cv2.VideoCapture(video) for video in videos["fake"]]
    frame_count = int(real_cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frames = np.linspace(3, frame_count - 1, n, endpoint=False, dtype=int)

    records = []

    for i, frame in enumerate(frames):
        if jitter:
            offset = 1 if random.random() < 0.5 else -1
            frame = frame + offset
        print(f"Start processing frame {i} at pos {frame}")
        real_cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        img = real_cap.read()[1]
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        [cap.set(cv2.CAP_PROP_POS_FRAMES, frame) for cap in fake_cap]
        fake_img = [cap.read()[1] for cap in fake_cap]
        [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in fake_img]
        w, h, _ = img.shape

        for face in face_recognition.face_locations(img):
            # Adding 30% Margin on each side but cap it to prevent wrap around
            d_w = ((face[1] - face[3]) // 3)
            d_h = ((face[2] - face[0]) // 3)
            x1, y1, x2, y2 = face[3] - d_w, face[0] - d_h, face[1] + d_w, face[2] + d_h
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
            face = (y1, x2, y2, x1)
            try:
                crop = np.array(img[face[0]:face[2], face[3]:face[1]])
                landmarks = face_recognition.face_landmarks(crop)
                diff = generate_diff(crop, crop)

                records.append(generate_record("real", crop, landmarks, diff))

                for fake in fake_img:
                    try:
                        f_crop = np.array(fake[face[0]:face[2], face[3]:face[1]])
                        diff = generate_diff(crop, f_crop)
                        records.append(generate_record("fake", f_crop, landmarks, diff))
                    except:
                        continue
            except:
                continue
        print(f"Finished processing frame {i} at pos {frame}")
    print(f"Finished processing images for {name} in {time.perf_counter() - t0}s")
    real_cap.release()
    [cap.release() for cap in fake_cap]

    return name, records


# 0-19153
if __name__ == "__main__":
    index = int(os.environ['SLURM_ARRAY_TASK_ID'])

    with open('pairs', 'rb') as pair_file:
        placesList = pickle.load(pair_file)
    videos = placesList[index]

    name, res = get_distributed_frames(videos=videos, n=20)

    if len(res) == 0:
        print(f"No faces processed for {name}")
        with open(f'/home/ai21m034/master_project/data/parquet/{name}.txt', 'w') as f:
            exit(0)
    print(type(res[0]["landmark"]))
    df = pd.DataFrame.from_records(res, index=range(len(res)))
    df.to_parquet(f'/home/ai21m034/master_project/data/parquet/{name}.parquet', engine='pyarrow', compression='snappy')
