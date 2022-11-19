from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import os
import pandas as pd

np.set_printoptions(suppress=True)

model = load_model('fire_model.h5', compile=False)
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

test_imgs = os.listdir('test_imgs')

positive = []
negative = []

total_score = 0

for testImg in test_imgs:
    image = Image.open(f'test_imgs/{testImg}').convert('RGB')
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    image_array = np.asarray(image)

    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

    data[0] = normalized_image_array

    prediction = model.predict(data)
    index = np.argmax(prediction)

    if index:
        positive.append([testImg + ',' + str(index)])
    else:
        negative.append([testImg + ',' + str(index)])
    
    total_score += prediction[0][index]


for dataX in positive:
    df = pd.DataFrame(dataX)
    df.to_csv('test_labels.csv', mode='a', index=False, header=False)
for dataX in negative:
    df = pd.DataFrame(dataX)
    df.to_csv('test_labels.csv', mode='a', index=False, header=False)

print(f'Images with Fire: {len(positive)}')
print(f'Images without Fire: {len(negative)}')
print(f'Total Score: {total_score/len(test_imgs)}')