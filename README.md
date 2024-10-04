# **AI ASSISTED BIDDING - TEST APP**

---
# **Demo**

<div align="center">
  <!-- First row -->
  <img src="assets/login.png" style="width:40%; display:inline-block; margin:15px;">
  <img src="assets/interface.png" style="width:40%; display:inline-block; margin:15px;">
</div>

<div align="center">
  <!-- First row -->
  <img src="assets/uploading.png" style="width:40%; display:inline-block; margin:15px;">
  <img src="assets/web_annotated.png" style="width:40%; display:inline-block; margin:15px;">
</div>

<div align="center">
  <!-- First row -->
  <img src="assets/full scale.png" style="width:40%; display:inline-block; margin:15px;">
  <img src="assets/bar.png" style="width:40%; display:inline-block; margin:15px;">
</div>

<div align="center">
  <!-- First row -->
  <img src="assets/estimator.png" style="width:40%; display:inline-block; margin:15px;">
  <img src="assets/estimate.png" style="width:40%; display:inline-block; margin:15px;">
</div>


### ***Prediction Pipeline***

This project implements an object detection pipeline using `YOLOv8` architecture with `Effientnetb6` backbone.

## **Setup**

1. Clone the repository:
  ```
  git clone https://github.com/RamakrishnaReddyPalle/AAB-test-app.git
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

4. The `params.yaml` file contains S3 buscket keys for:<br >
   a) Model key <br >
   b) Input/uploaded image key<br >
   c) Output/downloadable image key<br >
   d) Output `.csv` details key<br >

   * These can be only accessed through Access ID and secret key for the bucket
   * If you want to run the pipeline locally, params.yaml has #commented code which are the params for local, and the app.py abd prediction_pipeline.py has to be modified accordingly.

## **Running the Pipeline locally:**

5. To run the prediction pipeline, execute:
```
python src/pipeline/prediction_pipeline.py
```
