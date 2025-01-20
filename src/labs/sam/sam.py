# import torch
# import torchvision
# import sys
import numpy as np
# import torch
import matplotlib.pyplot as plt3
import cv2
from segment_anything import sam_model_registry, SamPredictor

# constants
image = None
image_path = "truck.jpg"

sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"

device = "cpu"

sam = None
predictor = None

input_point = None
input_label = None


def sam_menu():
    print("#######################################################")
    print("# you're in Segment Anything lab")
    print("# what do you want to do?")
    print("#\t- press 0 to run")
    print("#\t- press b to back to main menu")
    print("#######################################################")


def show_mask(mask, ax, random_color = False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis = 0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size = 375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color = 'green', marker = '*', s = marker_size, edgecolor = 'white',
               linewidth = 1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color = 'red', marker = '*', s = marker_size, edgecolor = 'white',
               linewidth = 1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt3.Rectangle((x0, y0), w, h, edgecolor = 'green', facecolor = (0, 0, 0, 0), lw = 2))


def load_image():
    global image, image_path

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def show_image():
    plt3.figure(figsize = (10, 10))
    plt3.imshow(image)
    plt3.axis('on')
    plt3.show()


def load_sam():
    global sam, model_type, sam_checkpoint, device

    sam = sam_model_registry[model_type](checkpoint = sam_checkpoint)
    sam.to(device = device)


def load_predictor():
    global sam, predictor

    predictor = SamPredictor(sam)


def process_image():
    global predictor, image

    predictor.set_image(image)


def set_points():
    global input_point, input_label

    input_point = np.array([[500, 375]])
    input_label = np.array([1])


def show_points():
    global image, input_point, input_label

    plt3.figure(figsize = (10, 10))
    plt3.imshow(image)
    show_points(input_point, input_label, plt3.gca())
    plt3.axis('on')
    plt3.show()


def sam_main():
    while True:
        sam_menu()
        action = input()
        if action == "0":
            load_image()
        elif action == "1":
            show_image()
        elif action == "2":
            load_sam()
        elif action == "3":
            load_predictor()
        elif action == "4":
            process_image()
        elif action == "5":
            set_points()
        elif action == "6":
            show_points()
        # elif action == "7":
        elif action == "b":
            break

sam_main()