<<<<<<< HEAD
# ***Prediction Pipeline***

This project implements an object detection pipeline using `YOLOv8` architecture with `Effientnetb6` backbone.

## **Setup**

1. Clone the repository:
  ```
  git clone https://github.com/FTCService/takeoff.git
  ```
2. Change directory to git clone:
```
cd "path\to\takeoff"
```
3. Install the dependencies:
  ```
  pip install -r requirements.txt
  ```

## **Configuration**

4. Update the `params.yaml` file with the required following configuration parameters: <br />
a) Input image path `(JPEG format)`<br />
b) Model path in your local `(takeoff\models\trained_models_with_tiles.pt)`<br />
c) Output Image `.jpeg` saving path<br />
d) Output image details `.csv` file saving path

## **Running the Pipeline**

5. To run the prediction pipeline, execute:
```
python src/pipeline/prediction_pipeline.py
```
=======
# AI-Assisted-Bidding-TEST-APP
>>>>>>> 375082bbe51af89b258c9e438b94c0dce5eac06b
