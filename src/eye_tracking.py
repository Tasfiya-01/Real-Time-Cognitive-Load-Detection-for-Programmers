import cv2
import time
import requests

def run():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Webcam not found!")
        return

    blink_count    = 0
    eye_closed     = False
    start_time     = time.time()
    prev_eye_count = 2

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    eye_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_eye.xml'
    )

    print("Eye Tracking Started! Press Q to quit.")
    print("="*50)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        eye_count = 0
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            eyes     = eye_cascade.detectMultiScale(roi_gray)
            eye_count = len(eyes)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 1)

        if eye_count < 2 and prev_eye_count >= 2:
            if not eye_closed:
                blink_count += 1
                eye_closed = True
        elif eye_count >= 2:
            eye_closed = False
        prev_eye_count = eye_count

        elapsed    = max(time.time() - start_time, 1)
        blink_rate = int(blink_count * (60 / elapsed))

        # Both too high (>25) AND too low (<8) = stress/fatigue
        if blink_rate > 25 or blink_rate < 8:
            eye_load = 'High'
            color    = (0, 0, 255)
        elif blink_rate > 18 or blink_rate < 12:
            eye_load = 'Medium'
            color    = (0, 165, 255)
        else:
            eye_load = 'Low'
            color    = (0, 200, 0)

        cv2.putText(frame, f"Blink Rate: {blink_rate}/min", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, f"Eye Load:   {eye_load}",       (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, f"Eyes seen:  {eye_count}",      (20, 115),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(frame, f"Blinks:     {blink_count}",    (20, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(frame, "Press Q to quit",               (20, 175),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

        # Send to backend every 5 seconds
        if int(elapsed) % 5 == 0 and int(elapsed) > 0:
            try:
                requests.post(
                    'http://localhost:5000/update_eye',
                    json={'blink_rate': blink_rate, 'eye_load': eye_load},
                    timeout=1
                )
            except:
                pass

        cv2.imshow('Eye Tracking - Cognitive Load Monitor', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nSession Summary:")
    print(f"  Total Blinks:  {blink_count}")
    print(f"  Blink Rate:    {blink_rate}/min")
    print(f"  Eye Load:      {eye_load}")
    print(f"  Duration:      {int(elapsed)}s")

if __name__ == '__main__':
    run()