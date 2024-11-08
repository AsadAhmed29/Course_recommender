# Course Recommender System

This project is a content-based recommendation system designed to recommend Coursera courses based on user search queries and search history. The system provides two types of recommendations:
1. **Similar Courses**: Shows courses similar to the currently searched course based on course content and features.
2. **Personalized Recommendations**: Offers personalized course recommendations based on the user's search history.

## Project Workflow

The project workflow is as follows:

1. **Data Preprocessing**:
   - Clean and preprocess the Coursera dataset, one-hot encoding categorical variables and translating course text data to English.
   - Vectorize course summaries using the CountVectorizer for similarity calculations.

2. **Exploratory Data Analysis (EDA)**:
   - Visualize data distributions and proportions of different attributes like course duration, difficulty, and certificate types.

3. **Course Search and Recommendation**:
   - Uses cosine similarity to find and recommend courses based on a user's current search or past searches.
   - Updates a user vector to track and personalize recommendations based on search history.


##Running the Project
Clone this repository and navigate to the project directory.

Run the Streamlit application with the following command:


streamlit run app.py
Once the app is running, it will open in a web browser. Use the sidebar to explore recommendations or view visualizations.
