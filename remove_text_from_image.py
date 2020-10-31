# Import required modules
import math
import cv2
import numpy as np


# Utility functions
def decode(scores, geometry, score_threshold):
    detections = []
    confidences = []

    # CHECK DIMENSIONS AND SHAPES OF geometry AND scores
    assert len(scores.shape) == 4, "Incorrect dimensions of scores"
    assert len(geometry.shape) == 4, "Incorrect dimensions of geometry"
    assert scores.shape[0] == 1, "Invalid dimensions of scores"
    assert geometry.shape[0] == 1, "Invalid dimensions of geometry"
    assert scores.shape[1] == 1, "Invalid dimensions of scores"
    assert geometry.shape[1] == 5, "Invalid dimensions of geometry"
    assert scores.shape[2] == geometry.shape[2], "Invalid dimensions of scores and geometry"
    assert scores.shape[3] == geometry.shape[3], "Invalid dimensions of scores and geometry"
    height = scores.shape[2]
    width = scores.shape[3]
    for y in range(0, height):

        # Extract data from scores
        scores_data = scores[0][0][y]
        x0_data = geometry[0][0][y]
        x1_data = geometry[0][1][y]
        x2_data = geometry[0][2][y]
        x3_data = geometry[0][3][y]
        angles_data = geometry[0][4][y]
        for x in range(0, width):
            score = scores_data[x]

            # If score is lower than threshold score, move to next x
            if score < score_threshold:
                continue

            # Calculate offset
            offset_x = x * 4.0
            offset_y = y * 4.0
            angle = angles_data[x]

            # Calculate cos and sin of angle
            cos_a = math.cos(angle)
            sin_a = math.sin(angle)
            h = x0_data[x] + x2_data[x]
            w = x1_data[x] + x3_data[x]

            # Calculate offset
            offset = (
                [offset_x + cos_a * x1_data[x] + sin_a * x2_data[x],
                 offset_y - sin_a * x1_data[x] + cos_a * x2_data[x]])

            # Find points for rectangle
            p1 = (-sin_a * h + offset[0], -cos_a * h + offset[1])
            p3 = (-cos_a * w + offset[0], sin_a * w + offset[1])
            center = (0.5 * (p1[0] + p3[0]), 0.5 * (p1[1] + p3[1]))
            detections.append((center, (w, h), -1 * angle * 180.0 / math.pi))
            confidences.append(float(score))

    # Return detections and confidences
    return [detections, confidences]


def remove_text_from_image(input_image_path, output_image_path, discarded_image_path):
    np.random.seed(42)

    conf_threshold = 0.7
    nms_threshold = 0.4
    input_default_width = 128
    input_default_height = 224
    model = "frozen_east_text_detection.pb"

    # Load network
    net = cv2.dnn.readNet(model)

    # Create a new named window
    output_layers = ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]

    # Open an image file
    image = cv2.imread(input_image_path)

    frame = image

    # Get frame height and width
    height_ = frame.shape[0]
    width_ = frame.shape[1]
    r_w = width_ / float(input_default_width)
    r_h = height_ / float(input_default_height)

    # Create a 4D blob from frame.
    blob = cv2.dnn.blobFromImage(frame, 1.0, (input_default_width, input_default_height), (123.68, 116.78, 103.94),
                                 True, False)

    # Run the model
    net.setInput(blob)
    output = net.forward(output_layers)
    t, _ = net.getPerfProfile()
    # label = 'Inference time: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())

    # Get scores and geometry
    scores = output[0]
    geometry = output[1]
    [boxes, confidences] = decode(scores, geometry, conf_threshold)
    # Apply NMS
    indices = cv2.dnn.NMSBoxesRotated(boxes, confidences, conf_threshold, nms_threshold)
    print(input_image_path, "\t", len(indices), "(text boxes)")
    for i in indices:
        # get 4 corners of the rotated rect
        vertices = cv2.boxPoints(boxes[i[0]])
        # scale the bounding box coordinates based on the respective ratios
        for j in range(4):
            vertices[j][0] *= r_w
            vertices[j][1] *= r_h

        # get upper-left and lower-right corners
        p1 = ((int(vertices[1][0]) if vertices[1][0] > 0 else 0),
              (int(vertices[1][1]) if vertices[1][1] > 0 else 0))
        p2 = ((int(vertices[3][0]) - 1 if vertices[3][0] < frame.shape[1] - 1 else frame.shape[1] - 2),
              (int(vertices[3][1]) - 1 if vertices[3][1] < frame.shape[0] - 1 else frame.shape[0] - 2))
        print(frame.shape, p1, p2)

        # compute average color of the pixels on the bounding box border
        edges = [frame[p1[1], p1[0]:p2[0]], frame[p2[1], p1[0]:p2[0]], frame[p1[1]:p2[1], p1[0]],
                 frame[p1[1]:p2[1], p2[0]]]
        sum_color = np.zeros((1, 3))
        num_edges = 0
        for edge in edges:
            if edge.shape[0]:
                sum_color += np.reshape(np.mean(edge, axis=0), (1, 3))
                num_edges += 1
        avg_color = np.reshape(sum_color / num_edges, (3, 1))
        color = tuple([int(channel) for channel in avg_color])

        # cover the text bounding box with an uniform colour
        cv2.rectangle(frame, p1, p2, color, -1)

        # apply noise to the rectangle
        for x in range(p1[0], p2[0] + 1):
            for y in range(p1[1], p2[1] + 1):
                noise = np.random.randint(10, size=3) * (np.random.randint(2, size=3) - 1)
                frame[y, x] = (frame[y, x] + noise) % 256

    if len(indices):
        cv2.imwrite(output_image_path, frame)
    else:
        cv2.imwrite(discarded_image_path, frame)

