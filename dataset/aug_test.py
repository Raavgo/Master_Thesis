from augmentation.augmentation import *
import pandas as pd
import pyarrow.parquet as pq
import cv2
from PIL import Image
from albumentations import \
    Compose, RandomBrightnessContrast, HorizontalFlip, FancyPCA, HueSaturationValue, OneOf, ToGray, \
    ShiftScaleRotate, ImageCompression, PadIfNeeded, GaussNoise, GaussianBlur, DualTransform, ImageOnlyTransform, Normalize


def save_transformed_image(image_path, transform, save_path):
    # Read the image
    image = cv2.imread(image_path)
    # Convert the image from BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Apply the transform
    transformed = transform(image=image)['image']

    # Convert the transformed image from RGB to BGR
    transformed = cv2.cvtColor(transformed, cv2.COLOR_RGB2BGR)

    # Save the transformed image
    cv2.imwrite(save_path, transformed)

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
    ])

def blackout_tiles_no(img, mask, i):
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
    bitmap_msk = masks[i]
    img *= np.expand_dims(bitmap_msk, axis=-1)

    return img

def blackout_convex_hull_no(img, landmark, i):
    img = img.copy()

    x, y = zip(*landmark)
    x, y = np.array(x), np.array(y)
    x_average = np.average(x)
    y_average = np.average(y)

    direction = [
        ('vertical', 'left'),
        ('vertical', 'right'),
        ('horizontal', 'top'),
        ('horizontal', 'bottom')
    ][i]

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

def merge_images(images, rows, name):
    # Assume all images are of the same size
    width, height = images[0].size

    # Define the padding
    padding = 10
    # Calculate the total width and height of the merged image
    total_width = max(rows) * width + (max(rows) - 1) * padding
    total_height = len(rows) * height + (len(rows) - 1) * padding

    # Create a new blank image with the calculated size
    merged_image = Image.new('RGB', (total_width, total_height), color=(255, 255, 255))

    # Paste the images into the merged_image with padding
    current_y = 0

    for i, num_images in enumerate(rows):
        current_x = 0
        for j in range(num_images):
            img_index = sum(rows[:i]) + j
            img = images[img_index]
            if num_images == 1:
                # Center the last image
                current_x = (total_width - width) // 2
            merged_image.paste(img, (current_x, current_y))
            current_x += width + padding
        current_y += height + padding

    # Save the merged image
    merged_image.save(name)

def blackout_aug(image, landmark, facial_features=None):
    if facial_features is None:
        facial_features = [("nose", 0), ("lip", 1), ("eyes", 1)]
    for feature in facial_features:
        image = blackout_landmark(image, landmark[feature[0]], axis=feature[1])
    return image

file = "/home/ai21m034/master_project/data/data/train/ahmtarbkeg.parquet"
row = pd.read_parquet(file).iloc[0]
img = np.frombuffer(row["crop"], dtype=np.uint8)
img = img.reshape(row["crop_shape"])

landmark = row["landmark"][0]
# landmark["outline"] = np.array(landmark["outline"])
landmark["lip"] = np.array(landmark["lip"])
landmark["eyes"] = np.array(landmark["eyes"])
landmark["nose"] = np.array(landmark["nose"])
diff = np.frombuffer(row["diff"], dtype=np.uint8)
diff = diff.reshape(row["diff_shape"])

#cv2.imwrite("test.png", img)

cv2.imwrite("blackout_tiles_1.png",  blackout_tiles_no(img, diff, 0))
cv2.imwrite("blackout_tiles_2.png",  blackout_tiles_no(img, diff, 1))
cv2.imwrite("blackout_tiles_3.png",  blackout_tiles_no(img, diff, 2))
cv2.imwrite("blackout_tiles_4.png",  blackout_tiles_no(img, diff, 3))
cv2.imwrite("blackout_tiles_5.png",  blackout_tiles_no(img, diff, 4))
cv2.imwrite("blackout_tiles_6.png",  blackout_tiles_no(img, diff, 5))

images = [
    Image.open('blackout_tiles_1.png'),
    Image.open('blackout_tiles_2.png'),
    Image.open('blackout_tiles_3.png'),
    Image.open('blackout_tiles_4.png'),
    Image.open('blackout_tiles_5.png'),
    Image.open('blackout_tiles_6.png')
]

merge_images(images, [3, 3], "blackout_tiles_merged.png")
cv2.imwrite("blackout_convex_hull_1.png",  blackout_convex_hull_no(img, landmark["outline"], 0))
cv2.imwrite("blackout_convex_hull_2.png",  blackout_convex_hull_no(img, landmark["outline"], 1))
cv2.imwrite("blackout_convex_hull_3.png",  blackout_convex_hull_no(img, landmark["outline"], 2))
cv2.imwrite("blackout_convex_hull_4.png",  blackout_convex_hull_no(img, landmark["outline"], 3))

images = [
    Image.open('blackout_convex_hull_1.png'),
    Image.open('blackout_convex_hull_2.png'),
    Image.open('blackout_convex_hull_3.png'),
    Image.open('blackout_convex_hull_4.png')
]

merge_images(images, [2, 2], "blackout_convex_hull_merged.png")

cv2.imwrite("blackout_facial_feature_1.png",  blackout_aug(img, landmark, [("nose", 0)]))
cv2.imwrite("blackout_facial_feature_2.png",  blackout_aug(img, landmark, [("lip", 1)]))
cv2.imwrite("blackout_facial_feature_3.png",  blackout_aug(img, landmark, [("eyes", 1)]))
cv2.imwrite("blackout_facial_feature_4.png",  blackout_aug(img, landmark, [("eyes", 1), ("lip", 1)]))
cv2.imwrite("blackout_facial_feature_5.png",  blackout_aug(img, landmark, [("eyes", 1), ("nose", 0)]))
cv2.imwrite("blackout_facial_feature_6.png",  blackout_aug(img, landmark, [("lip", 1), ("nose", 0)]))
cv2.imwrite("blackout_facial_feature_7.png",  blackout_aug(img, landmark, [("nose", 0), ("lip", 1), ("eyes", 1)]))

# Load the images
images = [
    Image.open('blackout_facial_feature_1.png'),
    Image.open('blackout_facial_feature_2.png'),
    Image.open('blackout_facial_feature_3.png'),
    Image.open('blackout_facial_feature_4.png'),
    Image.open('blackout_facial_feature_5.png'),
    Image.open('blackout_facial_feature_6.png'),
    Image.open('blackout_facial_feature_7.png'),
]

merge_images(images, [3, 3, 1], "blackout_facial_feature_merged.png")

images = []
transforms = create_train_transforms(size=320)
image_path = '/home/ai21m034/master_project/dataset/test.png'

for i in range(6):
    save_transformed_image(image_path, transforms, f'transformed_image_{i + 1}.png')
    images.append(Image.open(f'transformed_image_{i + 1}.png'))

merge_images(images, [3, 3], "transformed_image_merged.png")