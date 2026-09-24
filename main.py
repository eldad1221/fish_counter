import cv2
from ultralytics import YOLO
from common import get_home_video_filepath

FISH_CLASS_ID = 0


def initialize_model(model_path="yolo26l.pt"):
    """
    Loads and returns the YOLO model.
    """
    print(f"Loading model: {model_path}...")
    return YOLO(model_path)


def process_frame(frame, model, counted_ids, target_class_id=FISH_CLASS_ID):
    """
    Runs tracking on a single frame, updates the counted IDs,
    and draws the bounding boxes and text.
    """
    # Define the custom classes to search for using text prompts
    model.set_classes(["fish"])

    # Run detection and tracking (persist=True maintains IDs across frames)
    results = model.track(
        frame,
        persist=True,
        classes=[target_class_id],
        tracker="bytetrack.yaml",
        verbose=False  # Suppress per-frame terminal output for cleaner logs
    )

    # Check if any objects were detected with valid tracking IDs
    if results[0].boxes and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().tolist()

        for box, track_id in zip(boxes, track_ids):
            # Record the unique ID
            counted_ids.add(track_id)

            # Draw bounding box
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw tracking ID
            cv2.putText(
                frame,
                f"Fish #{track_id}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    # Overlay the cumulative count on the video frame
    cv2.putText(
        frame,
        f"Total Fish: {len(counted_ids)}",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 0, 255),
        3
    )

    return frame


def process_video(input_path, output_path, model_path="yolo26l.pt"):
    """
    Main pipeline to open a video, process it frame by frame, and save the result.
    """
    model = initialize_model(model_path)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file '{input_path}'")
        return

    # Retrieve video properties for the output writer
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Initialize the video writer
    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    counted_ids = set()

    print(f"Processing video: {input_path} (Press 'q' to stop early)")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Process the current frame
        annotated_frame = process_frame(frame, model, counted_ids, FISH_CLASS_ID)

        # Write to the output video
        out.write(annotated_frame)

        # Display the frame in a window (optional)
        cv2.imshow("Fish Tracking", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Processing interrupted by user.")
            break

    # Clean up resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"Done! Saved to: {output_path}")
    print(f"Total unique fish identified: {len(counted_ids)}")


def main():
    """
    Entry point of the script. Defines paths and triggers the video processing.
    """
    input_video = "fish6.mp4"
    video_filepath = get_home_video_filepath(input_video)
    output_video = get_home_video_filepath(f'out_{input_video}')
    yolo_model = "yolov8l-world.pt"

    process_video(video_filepath, output_video, yolo_model)


if __name__ == "__main__":
    main()
