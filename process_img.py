import json
import cv2
import numpy as np
import urllib.request as url
import time

def process_images():
    file = open("data_image_urls.json", "r")
    data = json.load(file)

    color_data = []


    debug_max = 5000
    debug_count = 0
    for i in data:
        debug_count += 1
        img_colors = get_colour_array(i["image"]["url"])
        if img_colors["blue/green"] == "error!":
            continue
        color_data.append({
            "id": i["video"],
            "topic" : i["topic"],
            "colors": img_colors
        })
        if debug_count >= debug_max:
            break
        print(debug_count)
    file2 = open("data_colors_smaller.json", "w")
    file2.write(json.dumps(color_data))
    file2.close()

def get_colour_array(img):

    color_dict = {
        "blue/green": 0,
        "yellow/orange": 0,
        "red/pink": 0,
        "purple/blue": 0
    }

    try:
        resp = url.urlopen(img)
    except url.HTTPError:
        return {"blue/green": "error!"}
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)


    iin = 0
    for i in hsv:
        jin = 0
        for j in i:
            if iin in range(12, 79):
                color = match_color(hsv[iin][jin])
                color_dict[color] += 1
            jin += 1
        iin += 1
    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB) 

    return color_dict

def match_color(hsv_val):    
    hue = hsv_val[0]

    if hue in range(0,85):
        return "blue/green"
    if hue in range(85, 110):
        return "yellow/orange"
    if hue in range(110, 157):
        return "red/pink"
    if hue in range(157, 181):
        return "purple/blue"
    

if __name__ == "__main__":
    start = time.time()

    process_images()

    end = time.time()
    print(end-start) 
    ### 920 seconds