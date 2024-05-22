import cv2

# For IP camera
# rtsp_url = "rtsp://admin:admin12345@192.168.1.2:554/live"
# For mobile phone
# rtsp_url = "rtsp://admin:admin@192.168.88.173:8080/h264_ulaw.sdp"
# For wireless camera
rtsp_url = "rtsp://admin:88888888@192.168.1.100/10554/udp/av0_0"

output_file = "save.mp4"
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
