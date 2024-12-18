import time
import numpy as np
import cv2
from tensorflow.lite.python.interpreter import Interpreter
from PIL import Image, ImageDraw


def infer(model_path, input_image):
    """Inference using TFLite model"""
    # Load TFLite model and allocate tensors
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Prepare sample input (dummy image for testing)
    input_shape = input_details[0]['shape']
    if input_shape[1] < 256 or input_shape[2] < 256:
        input_shape = (1, 256, 256, 3)

    # Changes as per model input
    input_image = input_image.resize((input_shape[1], input_shape[2]))  # width, height
    input_image = np.expand_dims(input_image, axis=0).astype(np.int32)
    # Measure inference time
    start_time = time.time()
    interpreter.set_tensor(input_details[0]['index'], input_image)
    interpreter.invoke()
    inference_time = time.time() - start_time

    # Get output
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Print results
    # print(f"Inference time: {inference_time:.4f} seconds")
    # print(f"Output shape: {output_data.shape}")

    return output_data

def draw_keypoints(input_image, output_data):
    """Only for visualizing movenet (https://www.kaggle.com/models/google/movenet/)"""
    # Draw keypoints on the input image
    draw = ImageDraw.Draw(input_image)
    keypoints = output_data[0]
    for i in range(6):
        if keypoints[i][-1] < 0.5:  # Confidence score threshold for person
            continue
        # Draw keypoints
        for j in range(17):
            y = keypoints[i][j * 3] * input_image.height
            x = keypoints[i][j * 3 + 1] * input_image.width
            draw.ellipse([x - 5, y - 5, x + 5, y + 5], fill=(255, 0, 0))
        # Draw bounding box
        ymin = keypoints[i][-5] * input_image.height
        xmin = keypoints[i][-4] * input_image.width
        ymax = keypoints[i][-3] * input_image.height
        xmax = keypoints[i][-2] * input_image.width
        draw.rectangle([xmin, ymin, xmax, ymax], outline=(0, 255, 0), width=2)

    input_image.show()
    # save the image
    input_image.save("output.jpg")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--input_image", type=str, default=None)
    args = parser.parse_args()
    assert args.model_path is not None and args.input_image is not None, "PLEASE PROVIDE --model_path AND --input_image PATH"

    input_image = Image.open(args.input_image)
    output_data = infer(args.model_path, input_image)
    print(output_data)
    print("Person confidences:", output_data[:, :, -1])

    # Draw keypoints on the input image
    draw_keypoints(input_image, output_data)