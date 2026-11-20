
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------
# Load saved recommendation models
# -------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "recommender_artifacts.joblib"

@st.cache_resource
def load_models():
    return joblib.load(MODEL_PATH)

artifacts = load_models()

recipes = artifacts["recipes"]
user_item_matrix = artifacts["user_item_matrix"]
item_similarity = artifacts["item_similarity"]
tfidf = artifacts["tfidf"]
recipe_vectors = artifacts["recipe_vectors"]
case_model = artifacts["case_model"]

recipe_lookup = recipes.set_index("id")
recipe_id_to_row = {
    recipe_id: index
    for index, recipe_id in enumerate(recipes["id"])
}

# -------------------------------
# Helper functions
# -------------------------------

def normalize_scores(scores):
    scores = np.array(scores, dtype=float)

    if len(scores) == 0:
        return scores

    minimum = scores.min()
    maximum = scores.max()

    if maximum == minimum:
        return np.zeros_like(scores)

    return (scores - minimum) / (maximum - minimum)


def display_recipes(recipe_ids):
    for recipe_id in recipe_ids:
        if recipe_id not in recipe_lookup.index:
            continue

        recipe = recipe_lookup.loc[recipe_id]

        st.subheader(str(recipe["name"]))

        st.write("Cooking time:", recipe["minutes"], "minutes")

        ingredients = recipe.get("ingredients_list", [])

        if isinstance(ingredients, list):
            st.write("Ingredients:", ", ".join(ingredients))

        st.divider()


# -------------------------------
# Case-Based Filtering
# -------------------------------

def recommend_case(recipe_name, n=10):

    normalized_query = " ".join(recipe_name.lower().split())

    names = (
        recipes["name"].fillna("").str.lower()
        .str.replace(r"\s+", " ", regex=True).str.strip()
    )

    matches = recipes[
        names.str.contains(normalized_query, regex=False)
    ]

    if matches.empty:
        return None, []

    recipe_index = recipes.index.get_loc(matches.index[0])

    distances, indices = case_model.kneighbors(
        recipe_vectors[recipe_index],
        n_neighbors=n + 1
    )

    results = []

    for distance, index in zip(distances[0], indices[0]):

        if index == recipe_index:
            continue

        results.append(recipes.iloc[index]["id"])

        if len(results) == n:
            break

    return matches.iloc[0]["name"], results


# -------------------------------
# Collaborative Filtering
# -------------------------------

def recommend_cf(user_id, n=10):

    if user_id not in user_item_matrix.index:
        return None

    user_ratings = user_item_matrix.loc[user_id].values

    scores = np.asarray(
        user_ratings @ item_similarity
    ).flatten()

    rated_indices = np.where(user_ratings > 0)[0]
    scores[rated_indices] = -1

    top_indices = np.argsort(scores)[::-1][:n]

    return [
        user_item_matrix.columns[index]
        for index in top_indices
        if scores[index] > 0
    ]


# -------------------------------
# Hybrid Recommendation
# -------------------------------

def recommend_hybrid(user_id, n=10):

    if user_id not in user_item_matrix.index:
        return None

    user_ratings = user_item_matrix.loc[user_id].values

    liked_indices = np.where(user_ratings >= 4)[0]

    if len(liked_indices) == 0:
        liked_indices = np.where(user_ratings > 0)[0]

    if len(liked_indices) == 0:
        return []

    recipe_ids = user_item_matrix.columns.to_numpy()

    liked_ids = recipe_ids[liked_indices]

    liked_rows = [
        recipe_id_to_row[rid]
        for rid in liked_ids
        if rid in recipe_id_to_row
    ]

    if not liked_rows:
        return []

    user_profile = np.asarray(
        recipe_vectors[liked_rows].mean(axis=0)
    )

    case_scores = cosine_similarity(
        user_profile, recipe_vectors
    ).flatten()

    cf_scores = np.asarray(
        user_ratings @ item_similarity
    ).flatten()

    candidate_indices = np.where(user_ratings == 0)[0]
    candidate_ids = recipe_ids[candidate_indices]

    valid = [
        i for i, rid in enumerate(candidate_ids)
        if rid in recipe_id_to_row
    ]

    candidate_indices = candidate_indices[valid]
    candidate_ids = candidate_ids[valid]

    if len(candidate_ids) == 0:
        return []

    cf_normalized = normalize_scores(
        cf_scores[candidate_indices]
    )

    case_normalized = normalize_scores([
        case_scores[recipe_id_to_row[rid]]
        for rid in candidate_ids
    ])

    hybrid_scores = (
        0.5 * cf_normalized +
        0.5 * case_normalized
    )

    top_indices = np.argsort(hybrid_scores)[::-1][:n]

    return candidate_ids[top_indices].tolist()


# -------------------------------
# Streamlit User Interface
# -------------------------------

st.set_page_config(
    page_title="Recipe Recommendation System",
    page_icon="🍽️",
    layout="wide"
)

st.title("🍽️ Personalized Recipe Recommendation System")

st.write(
    "Discover recipes based on your preferences, "
    "cooking history and similar recipes."
)

st.sidebar.header("Recommendation Methods")

method = st.sidebar.selectbox(
    "Choose a recommendation method",
    [
        "Case-Based Filtering",
        "Collaborative Filtering",
        "Hybrid Recommendation"
    ]
)

if method == "Case-Based Filtering":

    st.header("Find Similar Recipes")

    recipe_name = st.text_input(
        "Enter a recipe name",
        placeholder="e.g. chocolate cake"
    )

    if st.button("Find Similar Recipes"):

        if recipe_name.strip():

            matched_name, recommendations = recommend_case(
                recipe_name
            )

            if matched_name is None:
                st.error("Recipe not found. Try another name.")

            else:
                st.success(f"Similar recipes for: {matched_name}")

                display_recipes(recommendations)

        else:
            st.warning("Please enter a recipe name.")


elif method in [
    "Collaborative Filtering",
    "Hybrid Recommendation"
]:

    st.header(method)

    user_id = st.selectbox(
    "Select a User ID",
    user_item_matrix.index.tolist()
    )

    if st.button("Get Recommendations"):

        if method == "Collaborative Filtering":
            recommendations = recommend_cf(int(user_id))

        else:
            recommendations = recommend_hybrid(int(user_id))

        if recommendations is None:
            st.error("User not found in the dataset.")

        elif len(recommendations) == 0:
            st.warning("No recommendations available.")

        else:
            st.success("Here are your recommended recipes!")

            display_recipes(recommendations)

st.sidebar.divider()
st.sidebar.caption("Personalized Recipe Recommendation System")
