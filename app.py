import streamlit as st
import torch
from PIL import Image
from io import BytesIO
import glob
from datetime import datetime
import os
import wget
import pathlib
pathlib.PosixPath=pathlib.WindowsPath
import sys
import pandas as pd
import warnings
import numpy as np
from food_recommender import FoodRecommendationSystem


recommender = FoodRecommendationSystem("recommended.csv", "ratings.csv")


from ast import literal_eval
from collections import Counter
from tqdm.auto import tqdm

import warnings

pd.set_option('display.max_colwidth', 1000)
warnings.filterwarnings("ignore")
tqdm.pandas()

# Configurations
CFG_MODEL_PATH = "models/best(1).pt"
CFG_ENABLE_URL_DOWNLOAD = True
CFG_ENABLE_VIDEO_PREDICTION = True
if CFG_ENABLE_URL_DOWNLOAD:
    # Configure this if you set cfg_enable_url_download to True
    url = "https://archive.org/download/yoloTrained/yoloTrained.pt"
# End of Configurations

data = {
    'Name': [
        'Aloo Gobi', 'Aloo Matar', 'Aloo Methi', 'Aloo Tikki', 'Apple', 'Bhindi Masala',
        'Biryani', 'Boiled Egg', 'Bread', 'Burger', 'Butter Chicken', 'Chai', 'Chicken Curry',
        'Chicken Tikka', 'Chicken Wings', 'Chole', 'Daal', 'French Fries', 'French Toast', 'Fried Egg',
        'Kadhi Pakora', 'Kheer', 'Lobia Curry', 'Omelette', 'Onion Pakora', 'Onion Rings', 'Palak Paneer',
        'Pancakes', 'Paratha', 'Rice', 'Roti', 'Samosa', 'Sandwich', 'Spring Rolls', 'Waffles', 'White Rice'
    ],

    'Calories': [
        150, 170, 160, 200, 95, 120, 320, 70, 80, 250, 350, 45, 220, 180, 280, 200, 160, 220, 200,
        90, 150, 220, 180, 150, 190, 210, 230, 220, 180, 130, 100, 180, 280, 160, 250, 150
    ]
}

food_calories = [(name, calorie) for name, calorie in zip(data['Name'], data['Calories'])]


num_items_detected_list = []  # Initialize an empty list to store the number of items detected

def imageInput(model, src):
    output_text = ""  # Initialize an empty string to store the output
    num_items_detected_list = []  # Initialize an empty list to store the number of items detected
    
    if src == 'Upload your own data.':
        image_file = st.file_uploader(
            "Upload An Image", type=['png', 'jpeg', 'jpg'])
        col1, col2 = st.columns(2)
        if image_file is not None:
            img = Image.open(image_file)
            with col1:
                st.image(img, caption='Uploaded Image',
                         use_column_width='always')
            ts = datetime.timestamp(datetime.now())
            imgpath = os.path.join('data/uploads', str(ts)+image_file.name)
            outputpath = os.path.join(
                'data/outputs', os.path.basename(imgpath))
            with open(imgpath, mode="wb") as f:
                f.write(image_file.getbuffer())

            with st.spinner(text="Predicting..."):
                # Load model
                pred = model(imgpath)
                pred.render()
                # save output to file
                for im in pred.ims:
                    im_base64 = Image.fromarray(im)
                    im_base64.save(outputpath)

            # Predictions
            img_ = Image.open(outputpath)
            with col2:
                st.image(img_, caption='Model Prediction(s)',
                         use_column_width='always')

            # Construct the output text
            output_text += "\n"
            num_items_detected = 0
            for det in pred.pred[0]:
                label = model.names[int(det[-1])]
                prob = det[4]
                output_text += f"{label}\n"
                num_items_detected += 1
            output_text += f"Number of items detected: {num_items_detected}\n"
            num_items_detected_list.append(num_items_detected)
                
        # Save the output to a text file
        with open("output.txt", "a") as text_file:
            text_file.write(output_text)
            
            
        # Write the number of items detected to a new text file
        with open("number_items_counter.txt", "a") as counter_file:
            for num_items in num_items_detected_list:
                counter_file.write(f"{num_items}\n")
    
    elif src == 'From example data.':
        # Image selector slider
        imgpaths = glob.glob('data/example_images/*')
        imgpaths = [img_path for img_path in imgpaths if img_path.lower().endswith(('.png', '.jpg', '.jpeg'))]  # Filter out non-image files
        if len(imgpaths) == 0:
            st.error('No images found, Please upload example images in data/example_images', icon="")
            return
        for image_file in imgpaths:
            col1, col2 = st.columns(2)
            with col1:
                img = Image.open(image_file)
                st.image(img, caption='Selected Image', use_column_width='always')
            with col2:
                with st.spinner(text="Predicting..."):
                    # Load model
                    pred = model(image_file)
                    pred.render()
                    # save output to file
                    outputpath = os.path.join('data/outputs', os.path.basename(image_file))
                    for im in pred.ims:
                        im_base64 = Image.fromarray(im)
                        im_base64.save(outputpath)
                    # Display prediction
                    img_ = Image.open(outputpath)
                    st.image(img_, caption='Model Prediction(s)', use_column_width='always')
                    
                    # Construct the output text
                    output_text += f"\n"
                    num_items_detected = 0
                    for det in pred.pred[0]:
                        label = model.names[int(det[-1])]
                        prob = det[4]
                        output_text += f"{label}\n"
                        num_items_detected += 1
                    output_text += f"Number of items detected: {num_items_detected}\n"
                    num_items_detected_list.append(num_items_detected)
                        
        # Save the output to a text file
        with open("output.txt", "w") as text_file:
            text_file.write(output_text)
    
        # Write the number of items detected to a new text file
        with open("number_items_counter.txt", "w") as counter_file:
            for num_items in num_items_detected_list:
                counter_file.write(f"{num_items}\n")
                
        num_items_detected_list = []
        
    
    
    # Print the output to terminal
    print(output_text)


        
def detected_food_items():
    df = pd.read_csv('indianrecipes.csv')

    columns_to_drop = ['RecipeIngredientParts', 'RecipeInstructions', 'RecipeId', 'Unnamed: 0']
    df.drop(columns=columns_to_drop, inplace=True)

    df.drop_duplicates(subset='Name', keep='first', inplace=True)

    df.to_csv('indianmodrecipes.csv', index=False)

    data = pd.read_csv('indianmodrecipes.csv')

    with open('output.txt', 'r') as file:
        items = [line.strip() for line in file]

    filtered_rows = []

    for item in items:
        filtered_rows.append(data[data['Name'] == item])

    newdata = pd.concat(filtered_rows, ignore_index=True)

    newdata_mod = newdata.assign(foodnumber=[i for i in range(len(newdata))])

    newdata_mod.to_csv('detectedfooditems.csv', index=False)
    
    df=pd.read_csv('detectedfooditems.csv')

    with open('number_items_counter.txt', 'r') as file:
        num_items_list = [int(line.strip()) for line in file]

    grouped_rows = []
    idx = 0

    for num_items in num_items_list:
        group_rows = df.iloc[idx:idx+num_items]
        combined_names = ', '.join(group_rows['Name'])
        combined_data = {
            'Name': combined_names,
            'CookTime': group_rows['CookTime'].sum(),
            'PrepTime': group_rows['PrepTime'].sum(),
            'TotalTime': group_rows['TotalTime'].sum(),
            'Calories': group_rows['Calories'].sum(),
            'FatContent': group_rows['FatContent'].sum(),
            'SaturatedFatContent': group_rows['SaturatedFatContent'].sum(),
            'CholesterolContent': group_rows['CholesterolContent'].sum(),
            'SodiumContent': group_rows['SodiumContent'].sum(),
            'CarbohydrateContent': group_rows['CarbohydrateContent'].sum(),
            'FiberContent': group_rows['FiberContent'].sum(),
            'SugarContent': group_rows['SugarContent'].sum(),
            'ProteinContent': group_rows['ProteinContent'].sum()
        }
        grouped_rows.append(combined_data)
        idx += num_items

    newdata_mod = pd.DataFrame(grouped_rows)
    newdata_mod['foodnumber']=[i for i in range(len(newdata_mod))]
    
    columns_to_drop = ['CookTime','PrepTime','TotalTime']
    newdata_mod.drop(columns=columns_to_drop, inplace=True)
    newdata_mod.to_csv('finaldetectedfooditems.csv', index=False)
    




def main():
    if CFG_ENABLE_URL_DOWNLOAD:
        downloadModel()
    else:
        if not os.path.exists(CFG_MODEL_PATH):
            st.error(
                'Model not found, please config if you wish to download model from url set `cfg_enable_url_download = True`  ', icon="⚠️")

    # Page Title
    st.title('Food Detection')
    
    # Select Input Source
    datasrc = st.radio("Select input source:", ['From example data.', 'Upload your own data.'])

    
    imageInput(loadmodel('cpu'), datasrc)
    detected_food_items()
    
    st.text('Check whether uploaded foods are recommended by doctors')
    st.title("Food Recommendation System")
    option = st.selectbox(
            'Select an option:',
            ('Vegan', 'Chicken', 'Less time to make')
    )

    if option == 'Vegan':
        st.subheader("Recommended Vegan Meals:")
        recommendations = recommender.recommend_vegan()
        if isinstance(recommendations, str):
            st.write(recommendations)
        else:
            st.write(recommendations)
    elif option == 'Chicken':
        st.subheader("Recommended Chicken Meals:")
        recommendations = recommender.recommend_chicken()
        if isinstance(recommendations, str):
            st.write(recommendations)
        else:
            st.write(recommendations)
    elif option == 'Less time to make':
        st.subheader("Recommended Meals with Less Time to Make:")
        recommendations = recommender.recommend_less_time()
        if isinstance(recommendations, str):
            st.write(recommendations)
        else:
            st.write(recommendations)
        
        
        
    


# Downlaod Model from url.
@st.cache_data
def downloadModel():
    if not os.path.exists(CFG_MODEL_PATH):
        wget.download(url, out="models/")

@st.cache_data
def loadmodel(device):
    if CFG_ENABLE_URL_DOWNLOAD:
        CFG_MODEL_PATH = f"models/{url.split('/')[-1:][0]}"
    model = torch.hub.load('ultralytics/yolov5', 'custom',
                           path='models/best(1).pt', force_reload=True, device=device)
    return model

if __name__ == '__main__':
    main()