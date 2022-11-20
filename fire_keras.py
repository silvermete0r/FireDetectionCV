from keras.models import load_model
from PIL import Image, ImageOps
import pandas as pd
import numpy as np
import os

np.set_printoptions(suppress=True)

model = load_model('fire_model.h5', compile=False)
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

test_imgs = os.listdir('test_imgs')

positive = []
negative = []

total_score = 0

for testImg in test_imgs:
    img = Image.open(f'test_imgs/{testImg}').convert('RGB')
    size = (224, 224)
    img = ImageOps.fit(img, size, Image.Resampling.LANCZOS)

    img_arr = np.asarray(img)

    norm_img_arr = (img_arr.astype(np.float32) / 127.0) - 1

    data[0] = norm_img_arr

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
