import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from segment_anything import sam_model_registry, SamPredictor

image_path = None
image = None

sam_checkpoint = "models\\sam_vit_h_4b8939.pth"
model_type = "vit_h"

device = "cpu"

sam = None
predictor = None

input_points = None
input_labels = None
input_boxes = None

batch_operation = None


def show_mask(mask, ax, random_color = False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis = 0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size = 100):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color = 'green', marker = '*', s = marker_size,
               edgecolor = 'white',
               linewidth = 1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color = 'red', marker = '*', s = marker_size, edgecolor = 'white',
               linewidth = 1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor = 'green', facecolor = (0, 0, 0, 0), lw = 2))


def provide_image_path():
    global image_path
    image_path = input("Provide image: ")


def load_image():
    global image, image_path

    print("Loading image...")
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print("Image has been loaded.")


def load_sam():
    global sam, model_type, sam_checkpoint, device

    print("Loading segment anything model...")
    sam = sam_model_registry[model_type](checkpoint = sam_checkpoint)
    sam.to(device = device)
    print("Segment anything model has been loaded.")


def load_predictor():
    global sam, predictor

    print("Loading predictor...")
    predictor = SamPredictor(sam)
    print("Predictor has been loaded.")


def process_image():
    global predictor, image

    print("Processing image by predictor...")
    predictor.set_image(image)
    print("Image has been processed.")


def get_input_points():
    global input_points, input_labels, batch_operation, predictor

    input_points = []
    input_labels = []

    while True:
        x = int(input("x: "))
        y = int(input("y: "))
        label = input("Is it foreground point? [y/n]: ").lower()

        input_points.append([x, y])

        if label == 'y':
            input_labels.append(1)
        else:
            input_labels.append(0)

        add_next_point = input("Do you want to add next point? [y/n]: ").lower()

        if add_next_point != 'y':
            if batch_operation == 'y':
                input_points = torch.tensor(input_points, device = predictor.device)
                input_labels = torch.tensor(input_labels, device = predictor.device)
            else:
                input_points = np.array(input_points)
                input_labels = np.array(input_labels)

            print(f"Points: {input_points}")
            print(f"Points' labels: {input_labels}")
            break


def get_input_box():
    global input_boxes, batch_operation, predictor

    input_boxes = []

    if batch_operation == 'y':
        while True:
            x1 = int(input("x1: "))
            y1 = int(input("y1: "))
            x2 = int(input("x2: "))
            y2 = int(input("y2: "))

            input_boxes.append([x1, y1, x2, y2])

            add_next_box = input("Do you want to add next box? [y/n]: ").lower()

            if add_next_box != 'y':
                input_boxes = torch.tensor(input_boxes, device = predictor.device)
                break
    else:
        x1 = int(input("x1: "))
        y1 = int(input("y1: "))
        x2 = int(input("x2: "))
        y2 = int(input("y2: "))

        input_boxes.append(x1)
        input_boxes.append(y1)
        input_boxes.append(x2)
        input_boxes.append(y2)
        input_boxes = np.array(input_boxes)

    print(f"Box: {input_boxes}")


def select_objects():
    global image, input_points, input_labels, input_boxes, batch_operation, predictor

    if batch_operation == 'y':
        masks, scores, logits = predictor.predict_torch(
            point_coords = input_points,
            point_labels = input_labels,
            boxes = predictor.transform.apply_boxes_torch(input_boxes, image.shape[:2]),
            multimask_output = False
        )
    else:
        masks, scores, logits = predictor.predict(
            point_coords = input_points,
            point_labels = input_labels,
            box = input_boxes,
            multimask_output = True
        )

    print(f"masks.shape: {masks.shape}")

    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize = (10, 10))
        plt.imshow(image)
        # plt.title(f"Mask {i + 1}, Score: {score:.3f}", fontsize = 18)
        plt.axis('on')

        if batch_operation == 'y':
            for mask in masks:
                show_mask(mask.cpu().numpy(), plt.gca(), random_color = True)

            if input_points is not None:
                show_points(input_points.cpu().numpy(), input_labels.cpu().numpy(), plt.gca())

            if input_boxes is not None:
                for box in input_boxes:
                    show_box(box.cpu().numpy(), plt.gca())
        else:
            show_mask(mask, plt.gca())

            if input_points is not None:
                show_points(input_points, input_labels, plt.gca())

            if input_boxes is not None:
                show_box(input_boxes, plt.gca())

        plt.show()


def init():
    provide_image_path()
    load_image()
    load_sam()
    load_predictor()
    process_image()


def analyze_image():
    global input_points, input_labels, input_boxes, batch_operation
    init()

    input_points = None
    input_labels = None
    input_boxes = None

    batch_operation = input("Do you want to provide more than one box? [y/n] ").lower()

    while True:
        add_points = input("Do you want to add any points? [y/n] ").lower()

        if add_points == 'y':
            get_input_points()

        add_box = input("Do you want to add any box? [y/n] ").lower()

        if add_box == 'y':
            get_input_box()

        select_objects()

        another_analyze = input("Do you want to perform another analyze on loaded and processed image? [y/n] ")

        if another_analyze != 'y':
            break


def main():
    while True:
        analyze_image()
        analyze_another_image = input("Do you want to analyze another image? [y/n] ").lower()

        if analyze_another_image == 'n':
            print("Goodbye! :)")
            break


main()
