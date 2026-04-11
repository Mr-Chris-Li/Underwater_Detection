from ultralytics import YOLO
import os

def choose_weights():
    # prefer local yolov11, fallback to local yolo26n, else use Ultralytics hub name
    w1 = os.path.join('copilot_plan', 'weights', 'yolov11n.pt')
    w2 = os.path.join('copilot_plan', 'weights', 'yolo26n.pt')
    if os.path.isfile(w1) and os.path.getsize(w1) > 0:
        return w1
    if os.path.isfile(w2) and os.path.getsize(w2) > 0:
        return w2
    return 'yolo26n.pt'

def main():
    weights = choose_weights()
    model = YOLO(weights)
    model.train(
        data='data/urpc.yaml',
        epochs=10,
        imgsz=384,
        batch=16,
        device=0,
        project='copilot_plan/train_outputs',
        name='short_baseline',
        exist_ok=True,
    )

if __name__ == '__main__':
    main()
