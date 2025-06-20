import streamlit as st
import tensorflow as tf
import numpy as np

def model_prediction(test_image):
    model = tf.keras.models.load_model("custom_CNN_Fruit_Disease_Recognition_model.keras")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions)  # Return index of max element

# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Fruit Disease Recognition"])

# Main Page
if app_mode == "Home":
    st.header("Fruit Disease Recognition System")
    image_path = "1_S1cPcXhPe3ZFtCT_4_JXaA.png"
    st.image(image_path, use_container_width=True)
    st.markdown("""
    Welcome to the Plant Disease Recognition System! 🌿🔍
    
    Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

    ### How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
    3. **Results:** View the results and recommendations for further action.

    ### Why Choose Us?
    - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

    ### Get Started
    Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!

    ### About Us
    Learn more about the project, our team, and our goals on the **About** page.
    """)

# About Project
elif app_mode == "About":
    st.header("About")
    st.markdown("""
                #### About Dataset
                This dataset is recreated using offline augmentation from the original dataset. The original dataset can be found on this github repo.
                This dataset consists of about 87K rgb images of healthy and diseased crop leaves which is categorized into 38 different classes. The total dataset is divided into 80/20 ratio of training and validation set preserving the directory structure.
                A new directory containing 33 test images is created later for prediction purpose.
                #### Content
                1. train (21,743 images)
                2. test (5,437 images)
                3. validation (2,718 images)
                """)

# Prediction Page
elif app_mode == "Fruit Disease Recognition":
    st.header("Disease Recognition")
    test_image = st.file_uploader("Choose an Image:")
    if st.button("Show Image"):
        st.image(test_image, use_container_width=True)
    if st.button("Predict"):
        result_index = model_prediction(test_image)
        class_name = ['Citrus__Black Spot', 'Citrus__Canker', 'Citrus__Healthy', 'Guava__Healthy', 'Guava__Phytopthora',
                      'Guava__Scab', 'Guava__Styler_and_Root', 'Mango__Alternaria', 'Mango__Anthracnose',
                      'Mango__Black Mould Rot', 'Mango__Healthy', 'Mango__Stem end Rot', 'Pomegranate__Alternaria',
                      'Pomegranate__Anthracnose', 'Pomegranate__Bacterial_Blight', 'Pomegranate__Cercospora',
                      'Pomegranate__Healthy']
        disease_solutions = {
            'Citrus__Black Spot': ("**Reason:** This fungal disease is caused by *Phyllosticta citricarpa* and manifests as dark, greasy-looking spots on citrus leaves, fruits, and twigs. It proliferates in wet, warm climates and can significantly reduce the marketability of the fruit.", 
                                   
                                   "**Solution:** To manage Citrus Black Spot, use a combination of strategies: regularly remove debris, prune for better airflow, maintain optimal tree health, apply and rotate copper-based and systemic fungicides, develop and use resistant varieties, and implement strict quarantine measures to prevent disease spread."),

            'Citrus__Canker': ("**Reason:** Caused by the bacterium *Xanthomonas citri subsp. citri*. This contagious disease leads to yellow haloed lesions on leaves, fruits, and stems and spreads rapidly in rainy, windy conditions, severely degrading fruit quality.", 
                               
                               "**Solution:** To manage citrus canker effectively, focus on rigorous sanitation practices, use copper-based fungicides regularly, plant resistant varieties, and adhere to strict quarantine measures. Additionally, maintain tree health with proper nutrition and irrigation, and consider planting windbreaks to reduce the spread of the disease. Regular monitoring for early detection is also crucial to prevent outbreaks."),
                    
        'Guava__Phytopthora': ("**Reason:** The *Phytophthora* species are responsible, attacking the roots and fruits leading to rot. These pathogens favor waterlogged soils and warm temperatures, rapidly destroying guava plants if unchecked.", 
                               
                               "**Solution:** To manage Guava Phytophthora, focus on improving drainage to prevent waterlogging, apply fungicides such as metalaxyl or mefenoxam, and ensure proper soil health to reduce stress on the plants. Additionally, avoid overhead irrigation to keep foliage dry and reduce the likelihood of infection."),

        'Guava__Scab': ("**Reason:** Caused by the fungus *Elsinoë fawcettii*, it creates scabby, corky spots on the fruits and leaves. High humidity and moist conditions facilitate its spread, impacting fruit appearance and salability.",
                         
                               "**Solution:** To manage Guava Scab, apply protective fungicides like copper-based products regularly, especially before rain. Ensure good sanitation by removing and destroying infected plant parts, and maintain proper air circulation through pruning to prevent the high humidity that favors this disease."),
        
        'Guava__Styler_and_Root': ("**Reason:** Likely caused by soil-borne fungal pathogens, this disease impacts the roots and floral parts of guava plants, leading to weakened trees and reduced fruit production. Poor drainage and over-irrigation exacerbate its development.",
                         
                               "**Solution:** To manage Guava Styler and Root Disease, ensure proper soil drainage and avoid waterlogging. Use fungicides approved for root diseases, and consider soil amendments to promote healthy root development. Regularly inspect the plant base and roots for early signs of infection and apply appropriate biological control agents if available."),

        'Mango__Alternaria': ("**Reason:** *Alternaria* fungi produce dark spots on leaves, stems, and fruits, especially under prolonged leaf wetness and high humidity, which are common in tropical climates.",
                         
                               "**Solution:** To manage Mango Alternaria, also known as Alternaria leaf spot, apply copper-based fungicides and ensure good air circulation within the canopy by pruning. Remove and destroy infected plant debris to reduce spore spread and apply fungicides during wet conditions when infection risks are high."),
                        
        'Mango__Anthracnose': ("**Reason:** This is one of the most devastating fungal diseases for mango, caused by *Colletotrichum gloeosporioides*. It thrives in humid conditions, causing black lesions that can lead to significant crop losses during flowering and fruit development.",
                         
                               "**Solution:** To manage Mango Anthracnose, use systemic fungicides such as azoxystrobin or chlorothalonil, especially during the flowering stage and pre-harvest. Regularly prune trees to improve air circulation and reduce humidity around the foliage. Remove fallen leaves and infected fruit from around the tree to minimize fungal spore reservoirs. During periods of high humidity or rain, increase the frequency of fungicide applications to effectively control the disease."),                        

        'Mango__Black Mould Rot': ("**Reason:** The fungus *Aspergillus niger* colonizes sugary deposits on mango fruits, typically left by pest insects like aphids, leading to sooty black mold that ruins fruit aesthetic and texture.",
                         
                               "**Solution:** To manage Mango Black Mould Rot, ensure good air circulation within the orchard by pruning densely foliated branches. Reduce humidity around the fruit by optimizing irrigation practices. Apply protective fungicides that target black mould specifically during periods of high humidity or after rainfall. Additionally, maintain cleanliness by removing any infected fruits and debris from the orchard to reduce the spread of fungal spores."), 

        'Mango__Stem end Rot': ("**Reason:** Caused by a variety of fungi, notably *Lasiodiplodia theobromae*, this disease enters through the stem end of harvested fruits, proliferating in warm, humid storage conditions, leading to decay at the stem and potentially entire fruit spoilage.",
                         
                               "**Solution:** To manage Mango Stem End Rot, focus on proper post-harvest handling to minimize fruit injury, as wounds can be entry points for the fungus. Apply appropriate fungicides, such as prochloraz or thiabendazole, especially during the post-harvest washing or dipping processes. Store mangoes at appropriate temperatures to slow the progression of the disease and ensure the fruit is dry before storage to reduce humidity-related issues."),  

        'Pomegranate__Alternaria': ("**Reason:** *Alternaria* fungi attack under warm, humid conditions, causing leaf spots and fruit rot which reduces the crop yield and can cause premature leaf fall.",
                         
                               "**Solution:** To manage Pomegranate Alternaria, which often manifests as leaf spots and fruit rot, apply foliar fungicides that target Alternaria species, such as azoxystrobin or mancozeb. Regularly remove and destroy infected plant debris to minimize the spread of spores. Ensure proper tree spacing and prune to improve air circulation, reducing the moisture that facilitates fungal growth. During wet weather, increase surveillance and fungicide applications to protect new growth and developing fruit."),  


  'Pomegranate__Anthracnose': ("**Reason:** Triggered by *Colletotrichum* fungi, this disease produces dark, sunken lesions on the fruit, which are particularly problematic during rainy seasons and can drastically affect the external and internal quality of the fruit.",
                         
                               "**Solution:** To manage Pomegranate Anthracnose, it's crucial to apply systemic fungicides like azoxystrobin or mancozeb during the flowering period and as fruits begin to develop. Ensure proper air circulation by pruning the trees to open up the canopy, which helps to lower humidity levels around the fruit and leaves. Remove and destroy any infected plant parts to reduce fungal spore loads in the orchard. During wet and humid conditions, increase the frequency of fungicide applications to protect new growth and fruit."),   


 'Pomegranate__Bacterial_Blight': ("**Reason:** This bacterial disease, caused by *Xanthomonas axonopodis*, manifests as leaf spots and fruit lesions, often leading to severe defoliation, fruit drop, and reduced overall tree vigor.",
                         
                               "**Solution:** To manage Pomegranate Bacterial Blight, caused by the bacterium Xanthomonas axonopodis, it's important to use copper-based bactericides to reduce the spread of the bacteria. Regularly prune the pomegranate trees to remove infected branches and improve air circulation, which helps to keep the canopy dry and less hospitable to bacterial growth. Avoid overhead irrigation, as water can facilitate the spread of bacteria. It's also essential to sanitize tools and equipment used in the orchard to prevent cross-contamination between trees."),   


 'Pomegranate__Cercospora': ("**Reason:** The fungus *Cercospora punicae* leads to leaf spotting and defoliation. It's favored by high humidity and can severely affect the photosynthetic ability of trees if left unmanaged.",
                         
                               "**Solution:** To manage Pomegranate Cercospora, which typically causes leaf spots and defoliation, apply foliar fungicides such as chlorothalonil or mancozeb, especially during conditions that favor disease development like high humidity and moisture. Prune the trees to improve air circulation and light penetration within the canopy, which helps reduce the leaf wetness that supports fungal growth. Regularly remove and destroy fallen leaves and other infected plant debris to decrease the fungal spore load in the area. Maintain a regular fungicide application schedule during the rainy season to protect new growth and prevent outbreaks."),                                  

            # Add similar entries for each disease
        }
        predicted_class = class_name[result_index]
        st.success("Model is Predicting it's a {}".format(predicted_class))
        if predicted_class in disease_solutions:
            reason, solution = disease_solutions[predicted_class]
            st.markdown(f"#### Reason for {predicted_class}:")
            st.write(reason)
            st.markdown(f"#### Solution for {predicted_class}:")
            st.write(solution)
