import cv2

# Open the input video file
input_file = 'input_video.mp4'
cap = cv2.VideoCapture('../asset/video/smh_wheelchair_rgb.mp4')

# Get the video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec for output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# Create a VideoWriter object to save the grayscale video
output_file = 'output_video_gray.mp4'
out = cv2.VideoWriter(output_file, fourcc, fps, (width, height), False)

# Process each frame in the input video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Write the grayscale frame to the output video file
    out.write(gray_frame)

    # Display the grayscale frame (optional)
    cv2.imshow('Grayscale Video', gray_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the resources
cap.release()
out.release()
cv2.destroyAllWindows()