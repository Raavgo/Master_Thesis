import math
import random

import numpy as np
import cv2
from scipy.ndimage import binary_dilation

from albumentations import \
    Compose, RandomBrightnessContrast, HorizontalFlip, FancyPCA, HueSaturationValue, OneOf, ToGray, \
    ShiftScaleRotate, ImageCompression, PadIfNeeded, GaussNoise, GaussianBlur, DualTransform, ImageOnlyTransform, Normalize

from albumentations.pytorch.transforms import ToTensorV2
from PIL import Image

def crop(img: np.ndarray, x_min: int, y_min: int, x_max: int, y_max: int):
    height, width = img.shape[:2]
    if x_max <= x_min or y_max <= y_min:
        raise ValueError(
            "We should have x_min < x_max and y_min < y_max. But we got"
            " (x_min = {x_min}, y_min = {y_min}, x_max = {x_max}, y_max = {y_max})".format(
                x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max
            )
        )

    if x_min < 0 or x_max > width or y_min < 0 or y_max > height:
        raise ValueError(
            "Values for crop should be non negative and equal or smaller than image sizes"
            "(x_min = {x_min}, y_min = {y_min}, x_max = {x_max}, y_max = {y_max}, "
            "height = {height}, width = {width})".format(
                x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, height=height, width=width
            )
        )

    return img[y_min:y_max, x_min:x_max]


def dist(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def augment(image, payload):
    augmentation, data = payload

    if augmentation is None:
        return image
    elif augmentation == 'tiles':
        return blackout_tiles(image, data)
    elif augmentation == 'face_features':
        return blackout_facial_feature(image, data)
    elif augmentation == 'face_outline':
        return blackout_convex_hull(image, data)


def apply_augmentation(image, image_shape, diff, diff_shape, landmark):
    image = np.frombuffer(image, dtype=np.uint8)
    image = image.reshape(image_shape)
    diff = np.frombuffer(diff, dtype=np.uint8)
    diff = diff.reshape(diff_shape)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    augmentation_choice = [("tiles", diff), (None, None)]
    p = [0.6, 0.4]
    if landmark.size > 0:
        landmark = landmark[0]
        #landmark["outline"] = np.array(landmark["outline"])
        landmark["lip"] = np.array(landmark["lip"])
        landmark["eyes"] = np.array(landmark["eyes"])
        landmark["nose"] = np.array(landmark["nose"])
        augmentation_choice.extend([("face_features", landmark), ("face_outline", landmark["outline"])])
        p = [0.15, 0.05, 0.5, 0.3]

    augmentation = random.choices(population=augmentation_choice, k=1, weights=p)[0]
    image = augment(image, augmentation)
    #image = Image.fromarray(image)
    return image


def dataset_augmentation_worker(x):
    return apply_augmentation(x["crop"], x["crop_shape"], x["diff"], x["diff_shape"], x["landmark"])


def blackout_facial_feature(image, landmark):
    n_sample = random.randint(1, 3)
    facial_features = random.choices([("nose", 0), ("lip", 1), ("eyes", 1)], k=n_sample)
    for feature in facial_features:
        image = blackout_landmark(image, landmark[feature[0]], axis=feature[1])
    return image


def blackout_landmark(image, landmark, axis=0):
    image = image.copy()
    x, y = zip(*landmark)
    x = np.array(x)
    y = np.array(y)

    if axis == 0:
        x1 = x2 = int(np.average(x))
        y1, y2 = int(np.min(y)), int(np.max(y))
    else:
        x1, x2 = int(np.min(x)), int(np.max(x))
        y1 = y2 = int(np.average(y))

    mask = np.zeros_like(image[..., 0])

    line = cv2.line(mask, (x1, y1), (x2, y2), color=(1), thickness=2)

    w = dist((x1, y1), (x2, y2))
    dilation = int(w // 4)
    line = binary_dilation(line, iterations=dilation)
    image[line, :] = 0
    return image


def change_padding(image, part=5):
    h, w = image.shape[:2]
    # original padding was done with 1/3 from each side, too much
    pad_h = int(((3 / 5) * h) / part)
    pad_w = int(((3 / 5) * w) / part)
    image = image[h // 5 - pad_h:-h // 5 + pad_h, w // 5 - pad_w:-w // 5 + pad_w]
    return image


def blackout_convex_hull(img, landmark):
    img = img.copy()

    x, y = zip(*landmark)
    x, y = np.array(x), np.array(y)
    x_average = np.average(x)
    y_average = np.average(y)
    #print(landmark)
    #print(landmark.shape)
    direction = random.choice([
        ('vertical', 'left'),
        ('vertical', 'right'),
        ('horizontal', 'top'),
        ('horizontal', 'bottom')
    ])

    if direction[0] == 'vertical' and direction[1] == 'left':
        x[x < x_average] = x_average
        #landmark[landmark[:, 0] < x_average, 0] = x_average

    if direction[0] == 'vertical' and direction[1] == 'right':
        x[x > x_average] = x_average
        #landmark[landmark[:, 0] > x, 0] = x_average

    if direction[0] == 'horizontal' and direction[1] == 'top':
        y[y < y_average] = y_average
        #landmark[landmark[:, 1] < y, 1] = y

    if direction[0] == 'horizontal' and direction[1] == 'bottom':
        y[y > y_average] = y_average
        #landmark[landmark[:, 1] > y, 1] = y
    landmark = np.vstack((x, y)).T
    cv2.fillPoly(img, pts=[landmark], color=(1))
    return img


def blackout_tiles(img, mask):
    def prepare_bit_masks(mask):
        h, w = mask.shape
        mid_w = w // 2
        mid_h = w // 2
        masks = []
        ones = np.ones_like(mask)
        ones[:mid_h] = 0
        masks.append(ones)
        ones = np.ones_like(mask)
        ones[mid_h:] = 0
        masks.append(ones)
        ones = np.ones_like(mask)
        ones[:, :mid_w] = 0
        masks.append(ones)
        ones = np.ones_like(mask)
        ones[:, mid_w:] = 0
        masks.append(ones)
        ones = np.ones_like(mask)
        ones[:mid_h, :mid_w] = 0
        ones[mid_h:, mid_w:] = 0
        masks.append(ones)
        ones = np.ones_like(mask)
        ones[:mid_h, mid_w:] = 0
        ones[mid_h:, :mid_w] = 0
        masks.append(ones)
        return masks

    img = img.copy()
    binary_mask = mask > 0.4 * 255
    masks = prepare_bit_masks((binary_mask * 1).astype(np.uint8))
    bitmap_msk = random.choice(masks)
    img *= np.expand_dims(bitmap_msk, axis=-1)

    return img


def _decode_row(row):
    label = 0 if row["label"] == 'fake' else 1
    image = np.frombuffer(row["crop"], dtype=np.uint8)
    image = image.reshape(row["crop_shape"])
    diff = np.frombuffer(row["diff"], dtype=np.uint8)
    diff = diff.reshape(row["diff_shape"])
    landmark = row["landmark"]

    if landmark.size > 0:
        landmark = landmark[0]
        landmark["outline"] = np.array(landmark["outline"])
        landmark["lip"] = np.array(landmark["lip"])
        landmark["eyes"] = np.array(landmark["eyes"])
        landmark["nose"] = np.array(landmark["nose"])

    return [label, image, diff, landmark]


# return {"image": image, "labels": np.array((label,)), "img_name": os.path.join(video, img_file),
#        "valid": valid_label, "rotations": rotation}
def decode_df(df):
    results = []
    for i in range(len(df)):
        results.append(_decode_row(df.iloc(i)))
    return results



def isotropically_resize_image(img, size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC):
    h, w = img.shape[:2]

    if max(w, h) == size:
        return img
    if w > h:
        scale = size / w
        h = h * scale
        w = size
    else:
        scale = size / h
        w = w * scale
        h = size
    interpolation = interpolation_up if scale > 1 else interpolation_down

    resized = cv2.resize(img, (int(w), int(h)), interpolation=interpolation)
    return resized


class IsotropicResize(DualTransform):
    def __init__(self, max_side, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC,
                 always_apply=False, p=1):
        super(IsotropicResize, self).__init__(always_apply, p)
        self.max_side = max_side
        self.interpolation_down = interpolation_down
        self.interpolation_up = interpolation_up

    def apply(self, img, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC, **params):
        return isotropically_resize_image(img, size=self.max_side, interpolation_down=interpolation_down,
                                          interpolation_up=interpolation_up)

    def apply_to_mask(self, img, **params):
        return self.apply(img, interpolation_down=cv2.INTER_NEAREST, interpolation_up=cv2.INTER_NEAREST, **params)

    def get_transform_init_args_names(self):
        return ("max_side", "interpolation_down", "interpolation_up")


def create_train_transforms(size=300):
    return Compose([
        ImageCompression(quality_lower=60, quality_upper=100, p=0.5),
        GaussNoise(p=0.1),
        GaussianBlur(blur_limit=3, p=0.05),
        HorizontalFlip(),
        OneOf([
            IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC),
            IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_LINEAR),
            IsotropicResize(max_side=size, interpolation_down=cv2.INTER_LINEAR, interpolation_up=cv2.INTER_LINEAR),
        ], p=1),
        PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT),
        OneOf([RandomBrightnessContrast(), FancyPCA(), HueSaturationValue()], p=0.7),
        ToGray(p=0.2),
        ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=10, border_mode=cv2.BORDER_CONSTANT, p=0.5),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def create_val_transforms(size=300):
    return Compose([
        IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC),
        PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
