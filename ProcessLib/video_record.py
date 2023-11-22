import cv2

rtsp_url = "rtsp://example.com/stream"
output_file = "output_video.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

cap = cv2.VideoCapture(rtsp_url)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter(output_file, fourcc, 30.0, (frame_width, frame_height))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    out.write(frame)
    cv2.imshow("Frame", frame)
    # Press "q" to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
