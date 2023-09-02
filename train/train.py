import numpy as np
import pandas as pd
import cv2

def decode_row(row):
    label = row["label"]
    crop = np.frombuffer(row["crop"], dtype=np.uint8)
    crop = crop.reshape(row["crop_shape"])
    diff = np.frombuffer(row["diff"], dtype=np.uint8)
    diff = diff.reshape(row["diff_shape"])
    landmarks = row["landmark"]
    if landmarks.size > 0:
        landmarks = landmarks[0]
        landmarks["outline"] = np.array(landmarks["outline"])
        landmarks["lip"] = np.array(landmarks["lip"])
        landmarks["eyes"] = np.array(landmarks["eyes"])
        landmarks["nose"] = np.array(landmarks["nose"])

    return label, crop, diff, landmarks


df = pd.read_parquet(r"/data/data/vpmyeepbep.data")
for i in range(len(df)):
    label, img, diff, landmarks = decode_row(df.iloc[i])
    cv2.imwrite("test.png", img)
    break
    print(label, img, landmarks, diff)
    #if landmarks:
        #img = blackout_convex_hull(img, landmarks["outline"], mode='half')
    # img = remove_landmark(img, landmark["nose"], axis=0)
    # img = remove_landmark(img, landmark["lip"], axis=1)
    # img = remove_landmark(img, landmark["eyes"], axis=1)

    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #img = Image.fromarray(img)
    #img.show()