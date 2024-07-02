# import streamlit as st
# import os
# import yaml
# import boto3
# import pandas as pd
# from io import BytesIO
# from PIL import Image
# from src.pipeline.prediction_pipeline import main as run_pipeline

# # Function to load configuration from params.yaml
# def load_config(config_path='params.yaml'):
#     with open(config_path, 'r') as file:
#         config = yaml.safe_load(file)
#     return config

# def upload_to_s3(file_data, bucket_name, key):
#     try:
#         s3 = boto3.resource(
#             service_name='s3',
#             aws_access_key_id='AKIA4E6QHUK7RFON3G5C',
#             aws_secret_access_key='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',
#             region_name='ap-south-1'  # Mumbai region
#         )
#         bucket = s3.Bucket(bucket_name)
#         # Ensure file_data is a file-like object
#         file_data.seek(0)
#         bucket.upload_fileobj(file_data, key)
#         return True
#     except Exception as e:
#         st.sidebar.error(f"Error uploading file to S3: {e}")
#         return False

# # Function to download file from S3
# def download_from_s3(bucket_name, key):
#     try:
#         s3 = boto3.resource(
#             service_name='s3',
#             aws_access_key_id='AKIA4E6QHUK7RFON3G5C',
#             aws_secret_access_key='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',
#             region_name='ap-south-1'  # Mumbai region
#         )
#         file_stream = BytesIO()
#         bucket = s3.Bucket(bucket_name)
#         bucket.download_fileobj(key, file_stream)
#         file_stream.seek(0)
#         return file_stream
#     except Exception as e:
#         st.sidebar.error(f"Error downloading file from S3: {e}")
#         return None

# # Function to load class counts from CSV
# def load_class_counts(csv_stream):
#     df = pd.read_csv(csv_stream)
#     class_counts = df['class'].value_counts().to_dict()
#     return class_counts

# # Main Streamlit app
# def main():
#     st.title('Object Detection with YOLOv8')

#     # Load configuration from params.yaml
#     config = load_config()

#     # File upload section
#     st.sidebar.title('Upload Image')
#     uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "pdf"])

#     if uploaded_file is not None:
#         st.sidebar.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
#         st.sidebar.write("Processing...")

#         # Upload file to S3
#         bucket_name = config['s3_bucket_name']
#         input_image_key = config['s3_input_image_key']
#         if upload_to_s3(uploaded_file, bucket_name, input_image_key):
#             st.sidebar.success("File uploaded to S3 successfully!")

#             # Run prediction pipeline
#             config['s3_input_image_key'] = input_image_key
#             run_pipeline()

#             # Download and display annotated image from S3
#             output_image_key = config['s3_output_image_key']
#             image_stream = download_from_s3(bucket_name, output_image_key)
#             if image_stream:
#                 annotated_image = Image.open(image_stream)
#                 st.image(annotated_image, caption='Annotated Image', use_column_width=True)
#             else:
#                 st.error("Annotated image not found in S3!")

#             # Display download link for annotated image
#             s3_img_url = f"https://{bucket_name}.s3.amazonaws.com/{output_image_key}"
#             st.markdown(f"Download annotated image [HERE!]({s3_img_url})")

#             # Download and display class counts from S3
#             output_csv_key = config['s3_output_csv_key']
#             csv_stream = download_from_s3(bucket_name, output_csv_key)
#             if csv_stream:
#                 class_counts = load_class_counts(csv_stream)
#                 st.write("Class Counts:")
#                 st.bar_chart(class_counts)
#             else:
#                 st.error("CSV with class counts not found in S3!")

#             # Display download link for detailed results CSV
#             s3_csv_url = f"https://{bucket_name}.s3.amazonaws.com/{output_csv_key}"
#             st.markdown(f"Download detailed results CSV [HERE!]({s3_csv_url})")
#         else:
#             st.error("Failed to upload file to S3!")

# if __name__ == "__main__":
#     main()

import streamlit as st
import os
import yaml
import boto3
import pandas as pd
from io import BytesIO
from PIL import Image
from src.pipeline.prediction_pipeline import main as run_pipeline

# Function to load configuration from params.yaml
def load_config(config_path='params.yaml'):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def upload_to_s3(file_data, bucket_name, key):
    try:
        s3 = boto3.resource(
            service_name='s3',
            aws_access_key_id='AKIA4E6QHUK7RFON3G5C',
            aws_secret_access_key='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',
            region_name='ap-south-1'  # Mumbai region
        )
        bucket = s3.Bucket(bucket_name)
        # Ensure file_data is a file-like object
        file_data.seek(0)
        bucket.upload_fileobj(file_data, key)
        return True
    except Exception as e:
        st.sidebar.error(f"Error uploading file to S3: {e}")
        return False

# Function to download file from S3
def download_from_s3(bucket_name, key):
    try:
        s3 = boto3.resource(
            service_name='s3',
            aws_access_key_id='AKIA4E6QHUK7RFON3G5C',
            aws_secret_access_key='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',
            region_name='ap-south-1'  # Mumbai region
        )
        file_stream = BytesIO()
        bucket = s3.Bucket(bucket_name)
        bucket.download_fileobj(key, file_stream)
        file_stream.seek(0)
        return file_stream
    except Exception as e:
        st.sidebar.error(f"Error downloading file from S3: {e}")
        return None

# Function to load class counts from CSV
def load_class_counts(csv_stream):
    df = pd.read_csv(csv_stream)
    class_counts = df['class'].value_counts().to_dict()
    return class_counts

# Mapping class IDs to class names
class_names = {
    0: 'ceiling_mounted_luminaire',
    1: 'wall_mounted_luminaire',
    2: 'lightingstandard',
    3: 'self_powered_exit_sign',
    4: 'emergency_battery_unit',
    5: 'emergency_remote_head',
    6: 'single_pole_light_switch',
    7: 'key_operated_switch',
    8: 'three_way_light_switch',
    9: 'occupancy_sensor_manual_override_light_switch',
    10: 'dimmer_switch',
    11: 'suite_master_switch',
    12: 'fan_switch',
    13: 'ceiling_occupancy_switch',
    14: '15A_Duplex_Receptacle',
    15: '15A_Duplex_Receptacle_Mounted',
    16: 'T_Slot_Receptacle',
    17: 'Interrupter_Receptacle',
    18: 'Split_Type_Receptacle',
    19: 'Ground_Fault_Receptacle',
    20: 'Quadplex_Ground_Fault_Receptacle',
    21: 'Floor_Mounted_Box',
    22: '20A_Duplex_Ground_Fault_Receptacle',
    23: '20A_Duplex_Receptacle',
    24: '20A_Quadplex_Receptacle',
    25: '15A_125V_Receptacle',
    26: '20A_Receptacle',
    27: '30A_Receptacle',
    28: '50A_Receptacle',
    29: 'Wall_Mounted_Panel_Board',
    30: 'Unfused_Disconnect_Switch',
    31: 'Disconnect_Switch',
    32: 'Motor_Starter_SF1',
    33: 'Transformer_As_Noted',
    34: 'Electric_Heater',
    35: 'Line_Voltage_Thermostat',
    36: 'OBC_Push_Button',
    37: 'Direct_Power_Connection_To_Equipment',
    38: 'Provide_Power_Connection_To_Motor',
    39: 'Fire_Alarm_Adressable_Module',
    40: 'FIre_Alarm_Isolator',
    41: 'Fire_Alarm_Manual_Pull_Station',
    42: 'Fire_Alarm_Heat_Detector_Ceiling_Mounted',
    43: 'Fire_Alarm_Heat_Detector_Fixed',
    44: 'Fire_Alarm_Smoke_Detector_Ceiling_Mounted',
    45: 'Fire_Alarm_Smoke_Combination_Detector',
    46: 'Fire_Alarm_Co_Detector',
    47: 'Fire_Alarm_Duct_Type_Smoke_Detector',
    48: 'Fire_Alarm_Strobe',
    49: '125V_Smoke_Alarm',
    50: 'Fire_Alarm_Trouble_Light',
    51: 'Fire_Alarm_Handset',
    52: 'Fire_Alarm_Speaker_Strobe',
    53: 'Fire_Alarm_Speaker',
    54: 'Wireles_Door_Button',
    55: 'Wireless_Strobe_Chime',
    56: 'Telephone_Wall_Outlet',
    57: 'Data_Wall_Outlet',
    58: 'Wireless_Access_Points',
    59: 'TV_Wall_Outlet',
    60: 'CCTCV_Camera',
    61: 'SecurityDoor_Contact',
    62: 'Security_Electric_Strike',
    63: 'Junction_Box',
    64: 'Doorbell_System',
    65: 'Intercom',
    66: 'CardReader',
    67: 'Speakers',
    68: 'GroundBus',
    69: 'GroundRod',
    70: 'fire_alarm_horn',
    71: 'fire_alarm_strobe_horn'
}


def map_class_ids_to_names(class_counts, class_names):
    mapped_counts = {class_names.get(class_id, str(class_id)): count for class_id, count in class_counts.items()}
    return mapped_counts

def main():
    st.title('AI ASSISTED BIDDING')
    st.markdown('<h7 style="text-align: left;">By FTCS</h2>', unsafe_allow_html=True)
    config = load_config()
    
    # Initialize session state variables
    if 'pipeline_run' not in st.session_state:
        st.session_state.pipeline_run = False
    if 'annotated_image' not in st.session_state:
        st.session_state.annotated_image = None
    if 'detailed_csv' not in st.session_state:
        st.session_state.detailed_csv = None
    if 'start_prediction' not in st.session_state:
        st.session_state.start_prediction = False
    
    st.sidebar.title('Upload Image')
    uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg"])

    if uploaded_file is not None:
        st.sidebar.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
        st.sidebar.write("Ready for prediction...")

        bucket_name = config['s3_bucket_name']
        input_image_key = config['s3_input_image_key']
        if upload_to_s3(uploaded_file, bucket_name, input_image_key):
            st.sidebar.success("File uploaded to S3 successfully!")

    # Main window buttons
    if uploaded_file is not None:
        if st.button("Start Prediction"):
            st.session_state.start_prediction = True
            st.session_state.pipeline_run = False
        
        if st.session_state.start_prediction and not st.session_state.pipeline_run:
            progress_bar = st.progress(0)
            st.write("Running prediction pipeline...")

            def progress_callback(progress):
                progress_bar.progress(progress)

            run_pipeline(progress_callback)
            st.session_state.pipeline_run = True
            st.success("Prediction pipeline completed!")
            progress_bar.empty()

            if st.session_state.pipeline_run:
                output_image_key = config['s3_output_image_key']
                image_stream = download_from_s3(bucket_name, output_image_key)
                if image_stream is not None:
                    st.session_state.annotated_image = image_stream.getvalue()
                
                output_csv_key = config['s3_output_csv_key']
                csv_stream = download_from_s3(bucket_name, output_csv_key)
                if csv_stream is not None:
                    st.session_state.detailed_csv = csv_stream.getvalue()

    # Display annotated image
    if st.session_state.annotated_image:
        st.image(st.session_state.annotated_image, caption='Annotated Image', use_column_width=True)
        st.download_button(
            label="Download Annotated Image",
            data=st.session_state.annotated_image,
            file_name="annotated_image.jpg",
            mime="image/jpeg"
        )

    # Display table of detected items and counts
    if st.session_state.detailed_csv:
        class_counts = load_class_counts(BytesIO(st.session_state.detailed_csv))
        mapped_class_counts = map_class_ids_to_names(class_counts, class_names)
        st.table(mapped_class_counts)

        # Display bar chart of detected items
        st.bar_chart(pd.DataFrame(mapped_class_counts.values(), index=mapped_class_counts.keys(), columns=['Count']))

        # Provide download button for detailed CSV
        st.download_button(
            label="Download Detailed CSV",
            data=st.session_state.detailed_csv,
            file_name="detailed_results.csv",
            mime="text/csv"
        )

if __name__ == '__main__':
    main()
