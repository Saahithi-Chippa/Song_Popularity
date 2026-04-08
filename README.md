# Song Popularity Prediction  

A full end-to-end machine learning pipeline that predicts the popularity (measured by YouTube likes) of a song based on its Spotify audio features and YouTube engagement metrics. The project covers everything from database normalization through experiment tracking, containerized deployment, and a live Streamlit web app.

---

## Business Impact  
This project demonstrates how machine learning can drive insights in the music and entertainment industry:
- Content Strategy: Platforms like Spotify or YouTube can predict which songs will trend, optimizing promotion strategies.
- Artist Support: Emerging artists can use feature-based predictions to refine their music for better audience engagement.
- Recommendation Systems: Enhances playlist curation and music recommendations by including popularity predictions as a factor.
- Marketing & Investment: Record labels and advertisers can use these predictions to allocate budgets toward songs with high success potential.
- User Experience: Listeners benefit from improved discovery of trending and relevant music.
By bridging data science and business strategy, this project highlights the real-world potential of predictive modeling in shaping media consumption.

---

## What I Learned  
Through this project, I strengthened my skills in:  
- Data Engineering & Preprocessing: Cleaning and normalizing raw real-world data.
- Feature Engineering: Selecting meaningful attributes that influence song popularity.
- Model Deployment & Integration: Connecting a Streamlit app with a REST API for live predictions.
- Collaborative Development: Working in a team environment, managing tasks, and integrating code contributions.
- Practical ML Application: Applying machine learning to a domain (music industry) with real business value.

---

## Key Findings

- YouTube engagement features (`views`, `comments`) dominate feature importance across all models — they are by far the strongest predictors of `likes`
- Log transformation of skewed YouTube metrics significantly improved model fit
- ExtraTreesRegressor with hyperparameter tuning achieved the best balance of RMSE and generalization
- Reducing to the top 4–5 features via feature importance produced comparable results to using all features, confirming the dominance of the engagement cluster

---

## Dataset

**Source:** [Spotify and YouTube — Kaggle](https://www.kaggle.com/datasets/salvatorerastelli/spotify-and-youtube)

The dataset combines Spotify audio features (danceability, energy, loudness, tempo, etc.) with YouTube engagement metrics (views, likes, comments) for thousands of tracks across multiple artists and albums.

**Target variable:** `likes` (continuous regression target; quartile-binned into `Popularity` labels for stratified splitting)

---

## Project Pipeline

```
Raw CSV (Kaggle)
      │
      ▼
Database Normalization (SQLite)
  ├── Artists table
  ├── Albums table
  └── TrackingDetails table
      │
      ▼
Data Loading via SQL JOIN → Pandas DataFrame
      │
      ▼
Train / Test Split (80/20, stratified on Popularity quartile)
      │
      ▼
Exploratory Data Analysis (train set only)
  ├── ydata-profiling automated report
  ├── Correlation heatmap
  ├── Violin plots (licensed, official_video vs. likes)
  ├── Distribution plots per feature
  └── Pairplot
      │
      ▼
Preprocessing Pipeline (class-based, sklearn)
  ├── Median imputation (numerical)
  ├── Mode imputation (categorical)
  ├── Log transform (duration_ms, views, likes, comments)
  ├── StandardScaler (numerical)
  └── OneHotEncoder (licensed, official_video)
      │
      ▼
Model Selection via LazyPredict → top candidates identified
      │
      ▼
Baseline Model Training
  └── RandomForestRegressor (Experiment 1)
      │
      ▼
Top 3 Model Training (Experiments 2–4)
  ├── DecisionTreeRegressor
  ├── ExtraTreesRegressor
  └── GradientBoostingRegressor
      │
      ▼
Hyperparameter Tuning — GridSearchCV, cv=10, scoring=R² (Experiments 5–7)
      │
      ▼
Feature Importance Analysis → reduced feature sets per model (Experiments 8–10)
      │
      ▼
Combined Feature Engineering
  ├── ft1 = danceability × energy
  └── ft2 = duration_ms × loudness (Experiment 11)
      │
      ▼
PCA (n_components=10) + ExtraTreesRegressor (Experiment 12)
      │
      ▼
All experiments tracked in MLflow → DagsHub
      │
      ▼
Best model → Docker image (MLflow model serve)
      │
      ▼
Docker container deployed on DigitalOcean Droplet (port 5001)
      │
      ▼
Streamlit app → deployed on Streamlit Community Cloud
```

---

## Tech Stack 🛠️

| Layer | Technology |
|---|---|
| Language | Python 3.x |
| Data storage | SQLite (`music_project.db`) |
| Data loading | pandas, sqlite3 |
| EDA | ydata-profiling, seaborn, matplotlib |
| Preprocessing | scikit-learn (`Pipeline`, `ColumnTransformer`, `StandardScaler`, `OneHotEncoder`, `SimpleImputer`) |
| Model selection | LazyPredict (nightly) |
| Models | RandomForest, DecisionTree, ExtraTreesRegressor, GradientBoosting Regressor Models |
| Hyperparameter tuning | scikit-learn `GridSearchCV` |
| Dimensionality reduction | scikit-learn `PCA` |
| Experiment tracking | MLflow + DagsHub |
| Containerization | Docker |
| Deployment | DigitalOcean Droplet (model server) |
| Frontend | Streamlit (Streamlit Community Cloud) |
| HTTP requests | requests |

---

## Repository Structure

```
song-popularity-prediction/
│
├── Normalize.ipynb          # Main notebook: normalization → EDA → training → experiments
├── song_app.py              # Streamlit frontend app
├── requirements.txt         # Python dependencies
├── input_options.json       # Sidebar slider/selectbox config for the Streamlit app
├── music_project.db         # SQLite database (generated by notebook)
└── README.md
```

---

## Input Features
The model accepts the following input parameters:
- Continuous Features (Slider Inputs)  
    Audio characteristics (danceability, energy, valence, etc.)  
    Temporal features  
    Acoustic properties  

- Categorical Features (Dropdown Inputs)  
    licensed: Whether the song is licensed (Yes/No)  
    official_video: Whether the song has an official video (Yes/No)  

---

## EDA Highlights

- `views`, `comments`, and `likes` are strongly positively correlated — the YouTube engagement cluster is the most predictive of popularity
- `licensed` and `official_video` show meaningful correlation with `likes` (violin plots)
- `duration_ms`, `views`, `likes`, and `comments` were right-skewed → log1p transformation applied
- Median income used for stratified train/test split to preserve popularity quartile distribution

---

## Deployment

### Model server (DigitalOcean Droplet)

The best-performing model is exported from MLflow and built into a Docker image using `mlflow models build-docker`. The container exposes a REST API at `/invocations` (port 5001) that accepts POST requests with JSON payloads and returns predicted like counts.

```bash
# Pull and run the Docker image on a DigitalOcean Droplet
docker pull <your-dockerhub-username>/song-popularity-model:latest
docker run -p 5001:8080 <your-dockerhub-username>/song-popularity-model:latest
```

The Droplet provides a public IP so the Streamlit app can reach the model server from the cloud.

### Streamlit app

The app reads slider/selectbox ranges from `input_options.json` and sends user inputs as a POST request to the deployed Docker container.

```bash
# Run locally (point to local or remote Docker container)
streamlit run song_app.py
```

Or deploy for free via [Streamlit Community Cloud](https://streamlit.io/cloud) — connect your GitHub repo and set the app entrypoint to `song_app.py`.

---

## Local Setup 🚀

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/song-popularity-prediction.git
cd song-popularity-prediction
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate      # macOS / Linux
venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Additional packages used in the notebook (install as needed):

```bash
pip install ydata-profiling lazypredict-nightly mlflow dagshub scikit-learn
```

### 4. Download the dataset

Download from [Kaggle](https://www.kaggle.com/datasets/salvatorerastelli/spotify-and-youtube) and place the CSV in the project root.

### 5. Run the notebook

Open `Normalize.ipynb` in Jupyter and run all cells in order. The notebook will:
- Create the SQLite database and load the CSV into normalized tables
- Perform train/test split, EDA, and preprocessing
- Train and evaluate all models
- Log all 12 experiments to MLflow / DagsHub

### 6. Run the Streamlit app

```bash
streamlit run song_app.py
```

Update the model server URL in `song_app.py` to point to your deployed Docker container IP.

---

## Acknowledgments 👏

- [Salvatore Rastelli](https://www.kaggle.com/datasets/salvatorerastelli/spotify-and-youtube) for the Spotify + YouTube dataset
- [MLflow](https://mlflow.org/) and [DagsHub](https://dagshub.com/) for experiment tracking
- [Streamlit](https://streamlit.io/) for the frontend framework
- [DigitalOcean](https://www.digitalocean.com/) for the model server hosting ($6/month droplet)
