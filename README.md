# Personalized Recipe Recommendation System

A personalized recipe recommendation system developed using Python and machine learning. The project recommends recipes based on recipe similarity, user preferences, and ratings.

## Project Overview

The system uses three recommendation approaches:

1. **Case-Based Filtering:** Finds recipes similar to a selected recipe using ingredients, tags, descriptions, and other recipe information.
2. **Collaborative Filtering:** Recommends recipes based on similarities in user rating patterns.
3. **Hybrid Recommendation:** Combines Case-Based and Collaborative Filtering to generate recommendations.

The application provides an interactive interface built with Streamlit.

## Features

- Search for similar recipes.
- Get recommendations using a Food.com user ID.
- Generate hybrid recommendations.
- View cooking times and ingredients.
- Compare recommendation models using evaluation metrics.

## Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- SciPy
- TF-IDF Vectorization
- Cosine Similarity
- Nearest Neighbors
- Streamlit
- Jupyter Notebook

## Dataset

The project uses the Food.com Recipes and Interactions dataset from Kaggle.

Dataset: https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions

Download the following files:

- RAW_recipes.csv
- RAW_interactions.csv

Place them inside `data/raw/`.

The raw and processed datasets are not included in this GitHub repository because of their large file sizes.

## Project Structure

```text
Recipe-Recommendation-System/
│
├── app/
│   ├── app.py
│   └── test_app.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_collaborative_filtering.ipynb
│   └── 03_model_evaluation.ipynb
│
├── models/
├── results/
│   └── model_comparison.csv
│
├── requirements.txt
└── README.md
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/Hasan-Mustafa04/Recipe-Recommendation-System.git
cd Recipe-Recommendation-System
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Prepare the dataset

Download the Food.com dataset and place the two CSV files in `data/raw/`.

### 5. Generate the processed data and model

Run the notebooks in this order:

1. `01_data_exploration.ipynb`
2. `02_collaborative_filtering.ipynb`
3. `03_model_evaluation.ipynb`

The model artifact should be generated from the model-saving section of Notebook 2 and saved in `models/recommender_artifacts.joblib`.

### 6. Run the application

```bash
python -m streamlit run app/app.py
```

Open the local URL displayed in your terminal.

## Model Evaluation

The recommendation approaches were evaluated using Precision@10 and Recall@10.

| Model | Precision@10 | Recall@10 |
|---|---:|---:|
| Hybrid Recommendation | 0.0012 | 0.0035 |
| Case-Based Filtering | 0.0012 | 0.0029 |
| Collaborative Filtering | 0.0003 | 0.0004 |

The evaluation was conducted using a sample of 1,000 users. The results are preliminary and indicate opportunities for improvement, particularly in handling sparse user interactions and recipe catalogue coverage.

Detailed results are available in `results/model_comparison.csv`.

## Future Improvements

- Add recipe images and links.
- Improve recommendation accuracy.
- Introduce additional user preferences and dietary filters.
- Improve the application's user interface.

## Author

Developed as a Recommender Systems academic project.

**GitHub:** https://github.com/Hasan-Mustafa04