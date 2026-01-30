import os
import cv2
import numpy as np
import tensorflow as tf
from glob import glob
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# -------------------------------------
# 1️⃣ Hyperparameters
# -------------------------------------
IMG_SIZE = (256, 256)
BATCH_SIZE = 8
EPOCHS = 10
IMG_DIR = "processed"
MASK_DIR = "masks_auto"
CHECKPOINT = os.path.join(MODEL_DIR, "unet_best.h5")



# -------------------------------------
# 2️⃣ Modified U-Net Architecture
# -------------------------------------
def conv_block(x, filters):
    x = layers.Conv2D(filters, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x

def modified_unet(input_shape=(256,256,3), base_filters=32):
    inputs = layers.Input(shape=input_shape)

    # Encoder
    c1 = conv_block(inputs, base_filters)
    p1 = layers.MaxPooling2D()(c1)

    c2 = conv_block(p1, base_filters*2)
    p2 = layers.MaxPooling2D()(c2)

    c3 = conv_block(p2, base_filters*4)
    p3 = layers.MaxPooling2D()(c3)

    c4 = conv_block(p3, base_filters*8)
    p4 = layers.MaxPooling2D()(c4)

    # Bottleneck
    bn = conv_block(p4, base_filters*16)

    # Decoder
    u4 = layers.Conv2DTranspose(base_filters*8, 2, strides=2, padding="same")(bn)
    u4 = layers.Concatenate()([u4, c4])
    c5 = conv_block(u4, base_filters*8)

    u3 = layers.Conv2DTranspose(base_filters*4, 2, strides=2, padding="same")(c5)
    u3 = layers.Concatenate()([u3, c3])
    c6 = conv_block(u3, base_filters*4)

    u2 = layers.Conv2DTranspose(base_filters*2, 2, strides=2, padding="same")(c6)
    u2 = layers.Concatenate()([u2, c2])
    c7 = conv_block(u2, base_filters*2)

    u1 = layers.Conv2DTranspose(base_filters, 2, strides=2, padding="same")(c7)
    u1 = layers.Concatenate()([u1, c1])
    c8 = conv_block(u1, base_filters)

    outputs = layers.Conv2D(1, 1, activation="sigmoid")(c8)

    return models.Model(inputs, outputs, name="Modified_U-Net")

# -------------------------------------
# 3️⃣ Data Loading
# -------------------------------------
def load_data(img_dir, mask_dir):
    imgs, masks = [], []
    for cls in os.listdir(img_dir):
        ip = os.path.join(img_dir, cls)
        mp = os.path.join(mask_dir, cls)
        if not os.path.isdir(ip): continue
        for f in glob(os.path.join(ip, "*.jpg")):
            m = os.path.join(mp, os.path.basename(f))
            if os.path.exists(m):
                imgs.append(f)
                masks.append(m)
    return imgs, masks

def preprocess(img_path, mask_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, IMG_SIZE)
    img = img.astype("float32") / 255.0

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, IMG_SIZE)
    mask = (mask > 127).astype("float32")
    mask = np.expand_dims(mask, -1)

    return img, mask

def data_generator(imgs, masks, batch_size):
    while True:
        idxs = np.random.permutation(len(imgs))
        for i in range(0, len(imgs), batch_size):
            X, Y = [], []
            for j in idxs[i:i+batch_size]:
                img, mask = preprocess(imgs[j], masks[j])
                X.append(img)
                Y.append(mask)
            yield np.array(X), np.array(Y)

# -------------------------------------
# 4️⃣ Training
# -------------------------------------
if __name__ == "__main__":
    imgs, masks = load_data(IMG_DIR, MASK_DIR)
    print(f"Found {len(imgs)} image-mask pairs")

    split = int(0.8 * len(imgs))
    train_imgs, val_imgs = imgs[:split], imgs[split:]
    train_masks, val_masks = masks[:split], masks[split:]

    model = modified_unet(input_shape=(256,256,3))
    model.compile(
        optimizer=Adam(1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    callbacks = [
        ModelCheckpoint(CHECKPOINT, save_best_only=True, monitor="val_loss"),
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    ]

    model.fit(
        data_generator(train_imgs, train_masks, BATCH_SIZE),
        validation_data=data_generator(val_imgs, val_masks, BATCH_SIZE),
        steps_per_epoch=len(train_imgs)//BATCH_SIZE,
        validation_steps=len(val_imgs)//BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    model.save(os.path.join(MODEL_DIR, "final_unet.h5"))
    print("✅ Modified U-Net trained and saved in models/ folder")

