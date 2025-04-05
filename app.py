from ultralytics import YOLO

model = YOLO('yolo11x-seg-ft_ncnn_model')

def infer(image):
    results = model(image)
    return results

print(infer("sample_cyst.jpg"))