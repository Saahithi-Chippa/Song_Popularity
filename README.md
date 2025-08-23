# Song Popularity Prediction  

This project predicts the popularity of songs by analyzing audio features and metadata from **Spotify** and **YouTube**. It combines data preprocessing, feature normalization, and a deployed machine learning model, all wrapped in an interactive **Streamlit app**.


## Project Structure  
Song_Popularity/  
│── Normalize.ipynb # Jupyter Notebook for preprocessing & normalization  
│── song_app.py # Streamlit app for predictions  
│── input_options.json # Defines input options/ranges for the app  
│── requirements.txt # Python dependencies  
│── README.md # Project documentation  


## Dataset  
The dataset used in this project is from Kaggle: [Spotify and YouTube Dataset](https://www.kaggle.com/datasets/salvatorerastelli/spotify-and-youtube)  

It contains metadata and audio features of songs gathered from both **Spotify** and **YouTube**, including:
- Song and artist information  
- Audio features (danceability, acousticness, energy, loudness, etc.)  
- Metadata such as licensed status, official video, and views/likes/comments on YouTube  

This dataset provides a rich foundation for analyzing and predicting song popularity across platforms.  


## Features  
- **Data Preprocessing**: Cleaning and normalizing raw features in `Normalize.ipynb`.  
- **Interactive App**: Users can select song features and predict popularity through `song_app.py`.    
- **Model Serving**: The app communicates with a deployed ML model via REST API.  
- **Prediction Output**: Returns estimated popularity (by likes) for a song.   

## Technology Stack
Frontend: Streamlit  
Backend: Python, MLflow  
API: REST API for model predictions  
Data Processing: JSON-based input handling  

## Input Features
The model accepts the following input parameters:
- Continuous Features (Slider Inputs)  
    Audio characteristics (danceability, energy, valence, etc.)  
    Temporal features  
    Acoustic properties  

- Categorical Features (Dropdown Inputs)  
    licensed: Whether the song is licensed (Yes/No)  
    official_video: Whether the song has an official video (Yes/No)  

## Business Impact  
This project demonstrates how machine learning can drive insights in the music and entertainment industry:
- Content Strategy: Platforms like Spotify or YouTube can predict which songs will trend, optimizing promotion strategies.
- Artist Support: Emerging artists can use feature-based predictions to refine their music for better audience engagement.
- Recommendation Systems: Enhances playlist curation and music recommendations by including popularity predictions as a factor.
- Marketing & Investment: Record labels and advertisers can use these predictions to allocate budgets toward songs with high success potential.
- User Experience: Listeners benefit from improved discovery of trending and relevant music.
By bridging data science and business strategy, this project highlights the real-world potential of predictive modeling in shaping media consumption.

## What I Learned  
Through this project, I strengthened my skills in:  
- Data Engineering & Preprocessing: Cleaning and normalizing raw real-world data.
- Feature Engineering: Selecting meaningful attributes that influence song popularity.
- Model Deployment & Integration: Connecting a Streamlit app with a REST API for live predictions.
- Collaborative Development: Working in a team environment, managing tasks, and integrating code contributions.
- Practical ML Application: Applying machine learning to a domain (music industry) with real business value.

## Contact
Email: sachippus@gmail.com  
[Linkedin](https://www.linkedin.com/in/saahithi-ch-492545183/)