# **AI ASSISTED BIDDING - TEST APP**
### ***Prediction Pipeline***

This project implements an object detection pipeline using `YOLOv8` architecture with `Effientnetb6` backbone.

## **Setup**

1. Clone the repository:
  ```
  git clone https://github.com/FTCService/takeoff.git
  ```
2. Change directory to git clone:
```
cd "path\to\AAB-test-app"
```
3. Install the dependencies: (version specific given)
  ```
  pip install -r requirements.txt
  ```

## **Configuration**

4. The `params.yaml` file contains S3 buscket keys for:
   a) Model key
   b) Input/uploaded image key
   c) Output/downloadable image key
   d) Output `.csv` details key

   *These can be only accessed through Access ID and secret key for the bucket

## **Running the Pipeline**

5. To run the prediction pipeline, execute:
```
python src/pipeline/prediction_pipeline.py
```
