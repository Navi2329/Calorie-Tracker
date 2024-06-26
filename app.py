import streamlit as st
import numpy as np
import cv2
from PIL import Image
import pandas as pd
import random
import os
import requests
import tempfile
from ultralytics import YOLOv10

model = YOLOv10("runs/detect/train4/weights/best.pt")

classes = ['Biriyani', 'Chole', 'Dal Makhani', 'Dosa', 'Gulab Jamun', 'Idly', 'Khichdi',
           'Mango', 'Omlette', 'Paapad', 'Plain Rice', 'Poha', 'Poori', 'Rajma', 'Rasgulla',
           'Roti', 'Sambhar', 'Uttapam', 'Vada', 'almond', 'apple', 'apricots', 'banana',
           'dragon fruit', 'grapes', 'guava', 'orange', 'peach', 'pear', 'pineapple',
           'strawberry', 'sugar apple', 'walnut']

calories = pd.read_csv("calories.csv")

api_url = 'https://api.calorieninjas.com/v1/nutrition?query='

key = st.secrets["API_KEY"]


def generate_colors(num_colors):
    return [tuple(random.randint(0, 255) for _ in range(3)) for _ in range(num_colors)]


class_colors = {cls: color for cls, color in zip(
    classes, generate_colors(len(classes)))}


def annotate_image(image, results):
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = box.conf[0]
        cls_id = int(box.cls[0])
        label = f"{results[0].names[cls_id]} {conf:.2f}"
        color = class_colors[results[0].names[cls_id]]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(image, (x1, y1 - h - 10), (x1 + w, y1), color, -1)
        cv2.putText(image, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    return image


def annotate_video(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(
        *'H264'), fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model.predict(frame, conf=0.85, save=False)
        annotated_frame = annotate_image(frame.copy(), results)
        out.write(annotated_frame)

    cap.release()
    out.release()


def get_nutrition_info(query):
    response = requests.get(
        api_url + query, headers={'X-Api-Key': key})
    if response.status_code == requests.codes.ok:
        response = response.json()
        items = response.get('items', [])
        df = pd.DataFrame(columns=['Name', 'Serving Size', 'Calories', 'Total Fat',
                                   'Saturated Fat', 'Cholesterol', 'Sodium',
                                   'Carbohydrates', 'Fiber', 'Sugar', 'Protein'])
        for item in items:
            name = item.get('name', '')
            calories = str(item.get('calories', '')) + " kcal"
            serving_size = str(item.get('serving_size_g', '')) + " g"
            total_fat = str(item.get('fat_total_g', '')) + " g"
            saturated_fat = str(item.get('fat_saturated_g', '')) + " g"
            cholesterol = str(item.get('cholesterol_mg', '')) + " mg"
            sodium = str(item.get('sodium_mg', '')) + " mg"
            carbohydrates = str(item.get('carbohydrates_total_g', '')) + " g"
            fiber = str(item.get('fiber_g', '')) + " g"
            sugar = str(item.get('sugar_g', '')) + " g"
            protein = str(item.get('protein_g', '')) + " g"
            df.loc[len(df)] = [name, serving_size, calories, total_fat, saturated_fat,
                               cholesterol, sodium, carbohydrates, fiber, sugar, protein]
        st.dataframe(df, hide_index=True)
        return df
    else:
        st.error(f"Error: {response.status_code} - {response.text}")
        return None


def main():
    st.title("Food Calorie Estimator")
    st.sidebar.title("Options")
    option = st.sidebar.selectbox(
        'Choose an option',
        ('Image/Video Upload', 'Text/API Inference', 'Live Webcam Inference')
    )
    if option == 'Image/Video Upload':
        uploaded_file = st.file_uploader(
            "Choose a file...", type=["jpg", "jpeg", "png", "mp4", "avi", "mov"]
        )
        if uploaded_file is not None:
            query = None
            if uploaded_file.type in ["video/mp4", "video/avi", "video/quicktime"]:
                with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                    temp_file.write(uploaded_file.read())
                    video_path = temp_file.name
                with st.spinner('Processing video...'):
                    output_path = 'annotated_video.mp4'
                    annotate_video(video_path, output_path)
                    st.success("Video processing complete!")
                os.remove(video_path)
                st.video(output_path)
            else:
                image = Image.open(uploaded_file)
                image_np = np.array(image)
                image_cv2 = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
                results = model.predict(image_cv2, conf=0.3, save=False)
                annotated_image = annotate_image(image_cv2.copy(), results)
                annotated_image = cv2.cvtColor(
                    annotated_image, cv2.COLOR_BGR2RGB)
                st.image(annotated_image, caption='Annotated Image.', width=640,
                         use_column_width=False)

                preds = {}
                for result in results:
                    for box in result.boxes:
                        class_id = int(box.data[0][-1])
                        if model.names[class_id] in preds:
                            preds[model.names[class_id]] += 1
                        else:
                            preds[model.names[class_id]] = 1

                labels = [keys for keys in preds.keys()]
                relevant_calories = calories[calories['Name'].isin(labels)]
                total_calories = 0
                for label, quantity in preds.items():
                    calorie_value = relevant_calories[relevant_calories['Name']
                                                      == label]['Calories'].values[0]
                    total_calories += quantity * calorie_value

                st.subheader("Calories Information:")
                st.dataframe(relevant_calories, hide_index=True)
                st.subheader(f"Total Calories: {total_calories:.2f} kcal")

    elif option == 'Text/API Inference':
        query = st.text_area(
            "Enter the items you want to eat (separated by commas)")
        if query:
            get_nutrition_info(query)

    elif option == 'Live Webcam Inference':
        st.sidebar.markdown("### Live Webcam Inference(Coming Soon)")


if __name__ == '__main__':
    main()
