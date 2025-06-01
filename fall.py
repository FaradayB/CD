from ultralytics import YOLO
import cv2

model = YOLO("best.pt")

stream_url = "http://192.168.137.166/stream"

cap = cv2.VideoCapture(stream_url)

if not cap.isOpened():
    raise RuntimeError(f"cannot open stream at {stream_url}")

while True:
    ret, frame = cap.read()
    if not ret: break
    
    results = model(frame)
    
    for r in results:
        for box in r.boxes:
            x1, x2, y1, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            label = f"{model.names[cls]} {conf:.2f}"
            color = (0,0,225) if model.names[cls]=="Fall" else (255, 255, 255)
            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_COMPLEX, 0.6, color, 2) 
     
    cv2.imshow("Fall Detection", frame)
    if cv2.waitKey(1) & 0xFF in (27, ord('q')):
        break
    
cap.release()
cv2.destroyAllWindows()