import cv2

def calculate_frame_difference(video_path, threshold):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file!")
        return

    prev_frame = None
    # overall_difference = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Convert frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Skip the first frame
        if prev_frame is None:
            prev_frame = gray_frame
            continue

        # Calculate absolute difference between current and previous frame
        frame_diff = cv2.absdiff(prev_frame, gray_frame)
        frame_diff_map = frame_diff > 0
        # Sum of pixel differences in the frame
        # overall_difference += frame_diff.sum()
        diff_ratio = frame_diff_map.sum()/frame_diff.size
        print(diff_ratio)

        if diff_ratio > threshold:
            print("True")
        else:
            print("False")

        prev_frame = gray_frame
        # cv2.imshow("Frame", frame)
        # cv2.imshow("Frame Diff", frame_diff)
        # cv2.waitKey(1)

    cap.release()

    # Compare overall difference with the threshold



# Example usage
video_path = "/Users/cheungbh/Downloads/IMG_8468.MOV"
threshold = 0.9  # Adjust the threshold value as per your requirements
calculate_frame_difference(video_path, threshold)