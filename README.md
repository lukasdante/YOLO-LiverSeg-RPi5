# YOLO for Liver Segmentation on RPi5

This is a finetuned YOLO 11 XL segmentation model finetuned for focal liver lesion segmentation. The model has already been exported to an NCNN format for Raspberry Pi optimization. Refer to YOLO documentation for further details.

## How to use
1. Download the model [here](https://drive.google.com/drive/folders/1mYUs9F3wnqbS67T5zsHcZqaiT550K3Ev?usp=sharing). Download all the files, unzip it, and place it in the same directory or folder as the code here. Your directory should look like this:
```
Folder
├── api.py
├── app.py
├── client.py
├── env
├── README.md
├── requirements.txt
├── sample_cyst.jpg
├── sample_tumor.jpg
├── yolo11x-seg-ft_ncnn_model
├── yolo11x-seg-ft.pt
└── yolo11x-seg-ft.torchscript
```
1. Install the necessary modules for running the model.
```
pip install -r requirements.txt
```

2. Run the model using a sample image.
```
python app.py
```

## How to customize
1. Perform How to use.
2. Suppose you have an image stored locally, just perform these lines of code.
```
from ultralytics import YOLO

model = YOLO('yolo11x-seg-ft_ncnn_model')

def infer(image):
    results = model(image)
    return results
```
3. Write the path of your image and call the `infer()` function.
```
infer("sample_tumor.jpg")
```
4. It will print out the results, or if you have a pipeline for preprocessing or postprocessing you can just copy the lines of code above and append it to your current code.

## How to use API

1. Perform How to use.
2. Open a terminal and run this line of code.
```
uvicorn api:app --reload --host 0.0.0.0 --port 8080
```
3. Once it is running, open a new terminal and use `curl` or create a Python code for getting a request. Replace `ABSOLUTE_IMAGE_PATH` to the path of the target image.
```
curl -X POST http://localhost:8080/infer -H "Content-Type: application/json" -d '{"image_path": "ABSOLUTE_IMAGE_PATH"}'
```
Running this with the `sample_cyst.jpg` image will result to the inference result shown below. This consists of the prediction, confidence of the prediction, the bounding boxes, and the segmentation mask.
```
[{"name":"Liver Cyst","class":1,"confidence":0.94259,"box":{"x1":130.35216,"y1":303.94183,"x2":238.91887,"y2":423.77313},"segments":{"x":[187.0,187.0,183.0,182.0,181.0,180.0,178.0,176.0,175.0,174.0,173.0,172.0,171.0,169.0,168.0,167.0,166.0,164.0,163.0,162.0,161.0,159.0,158.0,157.0,156.0,154.0,153.0,152.0,151.0,150.0,149.0,148.0,147.0,146.0,145.0,144.0,143.0,141.0,139.0,138.0,137.0,135.0,134.0,130.0,130.0,134.0,136.0,136.0,137.0,137.0,138.0,138.0,139.0,139.0,140.0,140.0,141.0,141.0,142.0,142.0,143.0,143.0,144.0,144.0,145.0,145.0,146.0,146.0,147.0,147.0,148.0,148.0,149.0,149.0,150.0,150.0,151.0,151.0,152.0,152.0,153.0,153.0,154.0,154.0,155.0,155.0,156.0,156.0,157.0,157.0,159.0,159.0,160.0,160.0,162.0,162.0,164.0,164.0,169.0,171.0,172.0,178.0,179.0,180.0,181.0,182.0,183.0,184.0,185.0,186.0,187.0,188.0,189.0,190.0,191.0,193.0,194.0,196.0,197.0,198.0,199.0,201.0,202.0,204.0,205.0,206.0,208.0,209.0,210.0,212.0,213.0,216.0,217.0,220.0,221.0,225.0,226.0,229.0,233.0,233.0,234.0,234.0,235.0,235.0,234.0,234.0,233.0,233.0,232.0,232.0,231.0,231.0,230.0,230.0,229.0,229.0,228.0,228.0,227.0,227.0,226.0,226.0,225.0,225.0,224.0,224.0,223.0,223.0,222.0,222.0,221.0,221.0,220.0,220.0,219.0,219.0,218.0,218.0,217.0,217.0,216.0,216.0,215.0,215.0,214.0,214.0,213.0,213.0,212.0,212.0,211.0,211.0,210.0,210.0,209.0,209.0,208.0,208.0,207.0,207.0,206.0,206.0],"y":[302.0,306.0,310.0,310.0,311.0,311.0,313.0,313.0,314.0,314.0,315.0,315.0,316.0,316.0,317.0,317.0,318.0,318.0,319.0,319.0,320.0,320.0,321.0,321.0,322.0,322.0,323.0,323.0,324.0,324.0,325.0,325.0,326.0,326.0,327.0,327.0,328.0,328.0,330.0,330.0,331.0,331.0,332.0,332.0,347.0,347.0,349.0,350.0,351.0,353.0,354.0,356.0,357.0,359.0,360.0,362.0,363.0,365.0,366.0,369.0,370.0,372.0,373.0,374.0,375.0,376.0,377.0,379.0,380.0,381.0,382.0,384.0,385.0,387.0,388.0,389.0,390.0,391.0,392.0,393.0,394.0,395.0,396.0,397.0,398.0,399.0,400.0,401.0,402.0,403.0,405.0,406.0,407.0,408.0,410.0,411.0,413.0,414.0,419.0,419.0,420.0,420.0,419.0,419.0,418.0,418.0,417.0,417.0,416.0,416.0,415.0,415.0,414.0,414.0,413.0,413.0,412.0,412.0,411.0,411.0,410.0,410.0,409.0,409.0,408.0,408.0,406.0,406.0,405.0,405.0,404.0,404.0,403.0,403.0,402.0,402.0,401.0,401.0,397.0,396.0,395.0,392.0,391.0,388.0,387.0,381.0,380.0,378.0,377.0,374.0,373.0,371.0,370.0,369.0,368.0,366.0,365.0,364.0,363.0,361.0,360.0,359.0,358.0,357.0,356.0,355.0,354.0,353.0,352.0,351.0,350.0,348.0,347.0,346.0,345.0,344.0,343.0,341.0,340.0,339.0,338.0,337.0,336.0,335.0,334.0,333.0,332.0,330.0,329.0,327.0,326.0,325.0,324.0,321.0,320.0,316.0,315.0,312.0,311.0,308.0,307.0,302.0]}}]
```
4. You can use a Python code to request to this endpoint also. Inside `client.py` change the `ABSOLUTE_IMAGE_PATH` variable to your target image then run the code below. With this, you can use an API endpoint to send your preprocessed image. You can perform postprocessing within the script or a separate script (look at the comments where to put postprocessing). Open a terminal and run the code below.

```
python client.py
```

This will output the same output above.