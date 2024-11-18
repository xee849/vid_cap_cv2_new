from ultralytics import YOLO
import numpy as np
import tempfile
import pandas as pd
import streamlit as st
import altair as alt
import cv2
import time  # For controlling playback speed

# Load YOLOv8 model
model = YOLO('best.pt')

def detect_animals(image):
    """Detect animals in the given image and return the annotated image and counts."""
    animal_counts = {}
    results = model(image)
    annotated_image = results[0].plot()  # Plot bounding boxes
    for result in results:
        for class_index in result.boxes.cls:
            class_name = model.names[int(class_index)]  # Map class index to class name
            animal_counts[class_name] = animal_counts.get(class_name, 0) + 1
    return annotated_image, animal_counts

def process_video_frames(video_path):
    """Process video frames and return a list of annotated frames and counts."""
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))  # Frames per second
    frame_interval = int(fps // 20)  # Process 1 frame every 5th of a second

    frames = []
    all_animal_counts = {}  # Accumulate counts for the entire video

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if frame_count % frame_interval != 0:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        annotated_frame, animal_counts = detect_animals(frame)
        frames.append(annotated_frame)
        
        # Update cumulative counts
        for animal, count in animal_counts.items():
            all_animal_counts[animal] = all_animal_counts.get(animal, 0) + count

    cap.release()
    return frames, all_animal_counts, fps

def plot_animal_counts(animal_counts):
    """Generate a bar chart for animal counts."""
    if animal_counts:
        data = pd.DataFrame(list(animal_counts.items()), columns=["Animal", "Count"])
        chart = (
            alt.Chart(data)
            .mark_bar()
            .encode(
                x="Animal:O",
                y="Count:Q",
                color="Animal:O",
                tooltip=["Animal", "Count"]
            )
            .properties(width=400, height=300)
        )
        return chart
    else:
        return None

def main():
    st.sidebar.image("animalsimage.jpg", use_column_width=True)
    st.title("Species Identification and Monitoring Terrestrial")

    # Sidebar for user input
    st.sidebar.header("Upload Options")
    upload_type = st.sidebar.radio("Select Input Type", ["Video", "Image"])

    if upload_type == "Video":
        video_file = st.sidebar.file_uploader("Upload a video file", type=["mp4", "mov", "avi"])
        if video_file:
            # Save video file to a temporary location
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(video_file.read())
            video_path = tfile.name

            st.sidebar.header("Detection Progress")
            progress_bar = st.sidebar.progress(0)

            # Process video frames and store in a list
            frames, animal_counts, fps = process_video_frames(video_path)
            
            # Update progress bar
            progress_bar.progress(100)

            # Display frames like a video
            video_placeholder = st.empty()
            frame_delay = 1 / fps  # Calculate delay between frames

            for frame in frames:
                video_placeholder.image(frame, channels="RGB", use_column_width=False, width=640)
                time.sleep(frame_delay)  # Pause to simulate video playback speed

            # Display the chart after the video
            chart = plot_animal_counts(animal_counts)
            if chart:
                st.altair_chart(chart)

    elif upload_type == "Image":
        image_file = st.sidebar.file_uploader("Upload an image file", type=["jpg", "jpeg", "png"])
        if image_file:
            file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            annotated_image, animal_counts = detect_animals(image)

            st.image(annotated_image, channels="RGB", use_column_width=True, caption="Detected Animals")

            chart = plot_animal_counts(animal_counts)
            if chart:
                st.altair_chart(chart)

if __name__ == "__main__":
    main()






















# from ultralytics import YOLO
# import numpy as np
# import tempfile
# import pandas as pd
# import streamlit as st
# import altair as alt
# import cv2

# # Load YOLOv8 model
# model = YOLO('best.pt')

# def detect_animals(image):
#     """Detect animals in the given image and return the annotated image and counts."""
#     animal_counts = {}
#     results = model(image)
#     annotated_image = results[0].plot()  # Plot bounding boxes
#     for result in results:
#         for class_index in result.boxes.cls:
#             class_name = model.names[int(class_index)]  # Map class index to class name
#             animal_counts[class_name] = animal_counts.get(class_name, 0) + 1
#     return annotated_image, animal_counts

# def process_video_frame(video_path):
#     """Process video frame by frame and yield annotated frames and counts."""
#     cap = cv2.VideoCapture(video_path)
#     fps = int(cap.get(cv2.CAP_PROP_FPS))
#     frame_interval = int(fps // 20)  # Process 1 frame every 5th of a second

#     frame_count = 0
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
#         frame_count += 1
#         if frame_count % frame_interval != 0:
#             continue
#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         annotated_frame, animal_counts = detect_animals(frame)
#         yield annotated_frame, animal_counts
#     cap.release()

# def plot_animal_counts(animal_counts):
#     """Generate a bar chart for animal counts."""
#     if animal_counts:
#         data = pd.DataFrame(list(animal_counts.items()), columns=["Animal", "Count"])
#         chart = (
#             alt.Chart(data)
#             .mark_bar()
#             .encode(
#                 x="Animal:O",
#                 y="Count:Q",
#                 color="Animal:O",
#                 tooltip=["Animal", "Count"]
#             )
#             .properties(width=400, height=300)
#         )
#         return chart
#     else:
#         return None

# def main():
#     st.sidebar.image("animalsimage.jpg", use_column_width=True)
#     st.title("Species Identification and Monitoring Terrestrial")

#     # Sidebar for user input
    
#     st.sidebar.header("Upload Options")
#     upload_type = st.sidebar.radio("Select Input Type", ["Video", "Image"])

#     if upload_type == "Video":
#         video_file = st.sidebar.file_uploader("Upload a video file", type=["mp4", "mov", "avi"])
#         if video_file:
#             # Save video file to a temporary location
#             tfile = tempfile.NamedTemporaryFile(delete=False)
#             tfile.write(video_file.read())
#             video_path = tfile.name

#             st.sidebar.header("Detection Progress")
#             progress_bar = st.sidebar.progress(0)

#             # Main layout: video display and chart
#             video_placeholder = st.empty()  # Placeholder for the video
#             chart_placeholder = st.empty()  # Placeholder for the chart below video

#             fps = cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FPS)  # Get frame rate
#             total_frames = int(cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FRAME_COUNT))

#             # Process and display video frame by frame
#             for idx, (annotated_frame, animal_counts) in enumerate(process_video_frame(video_path)):
#                 # Update progress bar
#                 progress_bar.progress((idx + 1) / total_frames)

#                 # Display the video frame in a specific window size
#                 video_placeholder.image(
#                     annotated_frame, 
#                     channels="RGB", 
#                     use_column_width=False, 
#                     width=640
#                 )

#             # Display the chart after the video
#             chart = plot_animal_counts(animal_counts)
#             if chart:
#                 chart_placeholder.altair_chart(chart)

#     elif upload_type == "Image":
#         image_file = st.sidebar.file_uploader("Upload an image file", type=["jpg", "jpeg", "png"])
#         if image_file:
#             file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
#             image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
#             image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#             annotated_image, animal_counts = detect_animals(image)

#             st.image(annotated_image, channels="RGB", use_column_width=True, caption="Detected Animals")

#             chart = plot_animal_counts(animal_counts)
#             if chart:
#                 st.altair_chart(chart)

# if __name__ == "__main__":
#     main()









# from ultralytics import YOLO
# import numpy as np
# import tempfile
# import streamlit as st
# import cv2

# # Load YOLOv8 model
# model = YOLO('best.pt')

# def process_frame(frame):
#     # Run object detection
#     results = model(frame)
#     annotated_frame = results[0].plot()  # Plot the bounding boxes
#     return annotated_frame

# def main():
#     st.title("Species Identification and Monitoring Terrestrial")
    
#     # File uploader for video
#     video_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi"])
    
#     if video_file is not None:
#         # Save the uploaded file temporarily
#         tfile = tempfile.NamedTemporaryFile(delete=False)
#         tfile.write(video_file.read())
#         video_path = tfile.name
        
#         # OpenCV video capture
#         cap = cv2.VideoCapture(video_path)
        
#         stframe = st.empty()  # Placeholder for the video stream
#         fps = cap.get(cv2.CAP_PROP_FPS)  # Get the original frame rate of the video
#         frame_interval = int(fps // 5) 
        
#         frame_count = 0
#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 break
#             frame_count +=1
            
#             if frame_count % frame_interval != 0:
#                 continue
            
#             # Process frame
#             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB for Streamlit
#             annotated_frame = process_frame(frame)
            
#             # Display frame
#             stframe.image(annotated_frame, channels="RGB", use_column_width=True)
        
#         cap.release()
#     else:
#         st.info("Please upload a video file to start detection.")

# if __name__ == "__main__":
#     main()