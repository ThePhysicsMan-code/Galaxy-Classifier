import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import os
from PIL import Image
import numpy as np

# ----------------------------
# 1️⃣ Resize all images in dataset
# ----------------------------
DATA_DIR = "dataset"
TARGET_SIZE = (128, 128)

for class_folder in os.listdir(DATA_DIR):
    class_path = os.path.join(DATA_DIR, class_folder)
    if os.path.isdir(class_path):
        for img_file in os.listdir(class_path):
            img_path = os.path.join(class_path, img_file)
            try:
                img = Image.open(img_path).convert('RGB')
                img = img.resize(TARGET_SIZE)
                img.save(img_path)
            except Exception as e:
                print(f"Error resizing {img_path}: {e}")

# ----------------------------
# 2️⃣ Train the CNN model
# ----------------------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=TARGET_SIZE,
    batch_size=4,
    class_mode='binary',
    subset='training',
    shuffle=True
)

val_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=TARGET_SIZE,
    batch_size=4,
    class_mode='binary',
    subset='validation',
    shuffle=False
)

model = Sequential([
    tf.keras.Input(shape=(128, 128, 3)),
    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

EPOCHS = 10
model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS
)

model.save("best_galaxy_classifier.h5")
print("✅ Model trained and saved successfully as best_galaxy_classifier.h5!")
# ----------------------------
# 3️⃣ TESTING NEW IMAGES (Visual Pop-up + CSV + Save)
# ----------------------------
import csv
from PIL import ImageDraw, ImageFont
import matplotlib.pyplot as plt

TEST_DIR = "new_galaxies"
OUTPUT_DIR = "predicted_galaxies"
os.makedirs(OUTPUT_DIR, exist_ok=True)
CLASS_NAMES = list(train_generator.class_indices.keys())

# Prepare CSV file
csv_file = "galaxy_predictions.csv"
with open(csv_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Image", "Predicted Class", "Confidence"])

    for img_file in os.listdir(TEST_DIR):
        img_path = os.path.join(TEST_DIR, img_file)
        try:
            # Load and preprocess image
            img = Image.open(img_path).convert('RGB')
            img_array = np.array(img.resize(TARGET_SIZE)) / 255.0
            img_array_exp = np.expand_dims(img_array, axis=0)

            # Prediction
            prediction = model.predict(img_array_exp)[0][0]
            label = CLASS_NAMES[0] if prediction < 0.5 else CLASS_NAMES[1]
            confidence = prediction if label == CLASS_NAMES[1] else 1 - prediction

            # Overlay text on image
            draw = ImageDraw.Draw(img)
            font = ImageFont.load_default()
            draw.text((10, 10), f"{label} ({confidence*100:.1f}%)", fill=(255, 0, 0), font=font)

            # Save labeled image
            output_path = os.path.join(OUTPUT_DIR, f"predicted_{img_file}")
            img.save(output_path)

            # Write to CSV
            writer.writerow([img_file, label, f"{confidence*100:.2f}%"])

            # Display image in pop-up
            plt.imshow(img)
            plt.title(f"{img_file} → {label} ({confidence*100:.1f}%)")
            plt.axis('off')
            plt.show()  # Shows each image one by one

            print(f"🪐 {img_file}: {label} ({confidence*100:.2f}% confidence)")

        except Exception as e:
            print(f"Error testing {img_file}: {e}")

print(f"\n✅ All galaxy images analyzed! Labeled images saved in '{OUTPUT_DIR}', CSV saved as '{csv_file}'.")

