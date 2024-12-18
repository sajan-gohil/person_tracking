import os
from typing import Union
import cv2
from collections import deque
from infer import infer
from PIL import Image


def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) between two bounding boxes."""
    x1, y1, x2, y2 = box1
    x1_p, y1_p, x2_p, y2_p = box2

    # Calculate intersection
    inter_x1 = max(x1, x1_p)
    inter_y1 = max(y1, y1_p)
    inter_x2 = min(x2, x2_p)
    inter_y2 = min(y2, y2_p)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

    # Calculate union
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_p - x1_p) * (y2_p - y1_p)
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0


def real_time_tracking(video_path=0):
    # Initialize video capture (webcam)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not access webcam.")
        return

    model_path = os.getenv("TRACKING_MODEL_PATH",
                           "models/optimized_movenet.tflite")

    max_persons = 6  # Maximum number of persons to track. Model only gives 6 outputs max
    max_missing_frames = 5  # Maximum missing frames before removing a person
    iou_threshold = 0.2  # Threshold for assigning bounding boxes to existing persons

    # Track persons with their bounding boxes and trajectories
    persons = {
        i: {
            'box': None,
            'trajectory': deque(maxlen=50),
            'missing_frames': 0
        }
        for i in range(max_persons)
    }
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read frame.")
            break

        # Get bounding boxes from `infer`
        output = infer(model_path,
                       Image.fromarray(frame))  # List of bounding boxes
        output = output[0]
        bounding_boxes = []
        for i in range(len(output)):
            if output[i][-1] > 0.2:  # Confidence threshold
                ymin = int(output[i][-5] * frame.shape[0])
                xmin = int(output[i][-4] * frame.shape[1])
                ymax = int(output[i][-3] * frame.shape[0])
                xmax = int(output[i][-2] * frame.shape[1])
                bounding_boxes.append((xmin, ymin, xmax, ymax))

        # Match new detections with existing persons based on iou matches
        assigned = set()
        for person_id, person in persons.items():
            if person['box'] is None:
                continue

            best_iou = 0
            best_box = None
            best_index = -1
            for idx, box in enumerate(bounding_boxes):
                iou = calculate_iou(person['box'], box)
                if iou > best_iou and iou > iou_threshold:
                    best_iou = iou
                    best_box = box
                    best_index = idx

            # Update person if a match is found
            if best_box is not None:
                person['box'] = best_box
                center_x = (best_box[0] + best_box[2]) // 2
                center_y = (best_box[1] + best_box[3]) // 2
                person['trajectory'].append((center_x, center_y))
                person['missing_frames'] = 0
                assigned.add(best_index)

        # Assign unmatched bounding boxes to free persons
        for idx, box in enumerate(bounding_boxes):
            if idx in assigned:
                continue

            for person_id, person in persons.items():
                if person['box'] is None:
                    person['box'] = box
                    center_x = (box[0] + box[2]) // 2
                    center_y = (box[1] + box[3]) // 2
                    person['trajectory'].append((center_x, center_y))
                    person['missing_frames'] = 0
                    assigned.add(idx)
                    break

        # Handle missing persons
        for person_id, person in persons.items():
            if person['box'] is not None and person_id not in assigned:
                person['missing_frames'] += 1
                if person['missing_frames'] > max_missing_frames:
                    person['box'] = None
                    person['trajectory'].clear()

        # Draw bounding boxes and trajectories
        for person_id, person in persons.items():
            if person['box'] is not None:
                x1, y1, x2, y2 = person['box']
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # Add label
                cv2.putText(frame, f'Person {person_id}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Draw trajectory
                for i in range(1, len(person['trajectory'])):
                    if person['trajectory'][
                            i - 1] is None or person['trajectory'][i] is None:
                        continue
                    cv2.line(frame, person['trajectory'][i - 1],
                             person['trajectory'][i], (0, 0, 255), 2)

        # Display the frame with overlay
        cv2.imshow('Real-Time Human Tracking', frame)
        # Write to file
        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", default=0, type=str)
    args = parser.parse_args()
    if args.video_path is not None:
        # If a digit, then choose corresponding webcam
        if args.video_path.isdigit():
            args.video_path = int(args.video_path)
        real_time_tracking(args.video_path)
    