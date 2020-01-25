import io
import os
import cv2
from google.cloud import vision_v1p3beta1 as vision
from datetime import datetime
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'My Project 6358-c94b461ca5e8.json'

SOURCE_PATH = 'G:/fruits/'

FOOD_TYPE = "Fruit" 

def load_food_name(food_type):

    names = {line.rstrip('\n').lower() for line in open('fruitlist/' + food_type + '.dict')}
    return names

def recognise_food(img_path,list_foods):

    start_time = datetime.now()

    img = cv2.imread(img_path)

    height, width = img.shape[:2]

    img = cv2.resize(img, (800, int((height * 800) / width)))

    cv2.imwrite(SOURCE_PATH + "output.jpg", img)

    client = vision.ImageAnnotatorClient()

    with io.open(img_path, 'rb') as image_file :
        content = image_file.read()

    image = vision.types.Image(content=content)

    response = client.label_detection(image=image)

    labels = response.label_annotations

    for label in labels:
        desc = label.description.lower()

        score = round(label.score, 2)

        print('label : ', desc,'    score : ', score)
        if(desc in list_foods):
            cv2.putText(img, desc.upper(), (200, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
            cv2.imshow('done ', img)
            cv2.waitKey(0)

            break

    print('Total_time : {}'.format(datetime.now() - start_time))

print('start of the program')

list_foods = load_food_name(FOOD_TYPE)
print(list_foods)

path= SOURCE_PATH + 'fruit1.jpg'
recognise_food(path,list_foods)
print('-----------------------END---------------------')


