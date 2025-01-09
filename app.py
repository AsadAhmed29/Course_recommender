import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import streamlit as st
from googletrans import Translator
import nltk
nltk.download('punkt' , quiet = True)
#st.set_option('deprecation.showPyplotGlobalUse', False)
####################
#DATA_PREPROCESSING#
####################
df = pd.read_csv('coursera_courses.csv')
df = df.drop(['course_students_enrolled', 'course_url'], axis=1)
df = df.dropna()
df = df.reset_index(drop = True)
organization_one_hot_encoded = pd.get_dummies(df['course_organization'])
time_one_hot_encoded = pd.get_dummies(df['course_time'])
diff_one_hot_encoded = pd.get_dummies(df['course_difficulty'])
df = pd.concat([df, organization_one_hot_encoded , time_one_hot_encoded ,diff_one_hot_encoded ] , axis =1)
df_for_training = df.drop(['course_organization' , 'course_certificate_type', 'course_time', 'course_difficulty', 'course_rating','course_reviews_num'] , axis =1 )
df_for_training = df_for_training.reset_index(drop=True)

def convert_to_eng(dataframe , column_name):  #To Translate the column into english
    translated = []
    for value in dataframe[column_name]:
      try:
        translator = Translator()
        translation = translator.translate(value , dest='en')
        translated.append(translation.text)
      except Exception as e:
        print(f"Error translating: {e}")
    return translated


def convert_to_eng_missing(dataframe , column_name): ##To Translate the column iwith missing values in to english
    translated = []
    for value in dataframe[column_name]:
        try:
            if isinstance(value, str) and value.strip():  # Check if value is a non-empty string
                translator = Translator()
                translation = translator.translate(value, dest='en')
                translated.append(translation.text)
            else:
                translated.append(value)  # Append None for missing or empty values
        except Exception as e:
            print(f"Error while translating: {e}")
            translated.append(None)  # Append None for errors
    return translated


# df_for_training['course_title'] = convert_to_eng(df_for_training, 'course_title')


def clean_skills(column):
    return str(column).replace('[', '').replace(']', '').replace("'", '')

# Apply function to the 'course_skills' column
# df_for_training['course_skills'] = df_for_training['course_skills'].apply(clean_skills)
# df_for_training['course_skills'] = convert_to_eng_missing(df_for_training, 'course_skills')
# df_for_training['course_summary'] = df_for_training['course_summary'].apply(clean_skills)
# df_for_training['course_summary'] = convert_to_eng_missing(df_for_training,'course_summary')
# df_for_training['course_description'] = convert_to_eng_missing(df_for_training,'course_description')

df_for_training = pd.read_csv('df_for_training_preprocessed.csv')
df_for_training['course_skills'] = df_for_training['course_skills'].fillna('')
df_for_training['course_summary'] = df_for_training['course_summary'].fillna('')
df_for_training['course_description'] = df_for_training['course_description'].fillna('')

df_for_training['course_skills'] = df_for_training['course_skills'].str.replace(" ", "")
df_for_training['course_skills'] = df_for_training['course_skills'].apply(lambda x:x.split())
df_for_training['course_summary'] = df_for_training['course_summary'].apply(lambda x:x.split())
df_for_training['course_description'] = df_for_training['course_description'].apply(lambda x:x.split())

df_for_training['course_summary'] = df_for_training['course_skills'] + df_for_training['course_summary'] + df_for_training['course_description']
df_for_training = df_for_training.drop(['course_skills' , 'course_description'] , axis =1)
df_for_training['course_summary']  = df_for_training['course_summary'].apply(lambda x:[i.lower() for i in x])
df_for_training['course_summary']  = df_for_training['course_summary'].apply(lambda x:" ".join(x))


################################
##VECTORIZATION & USER VECTOR###
################################

from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

def stemming(text):
    words = nltk.word_tokenize(text)  # Tokenize the text into words
    stemmed_words = [ps.stem(word) for word in words]
    return " ".join(stemmed_words)

df_for_training['course_summary']  = df_for_training['course_summary'].apply(stemming)

from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import hstack

cv = CountVectorizer(max_features = 8000 , stop_words='english')
summary_vector = cv.fit_transform(df_for_training['course_summary'])

categorical_vector = df_for_training.iloc[:,2:].to_numpy()

final_vector = hstack((summary_vector, categorical_vector)).toarray()
similarity_matrix = cosine_similarity(final_vector)


st.session_state.user_vector = np.zeros((8167,))
searched_courses_count = 0

# Function to update user vector
def update_user_vector(user_vector, course_vector):
    if np.all(user_vector == 0):
        return course_vector
    else:
        return (user_vector * (searched_courses_count-1) + course_vector) / (searched_courses_count)






########################
##EXPLORATORY ANALYSIS##
########################

def plot_distribution(column , title , xlabel, cm):
    plt.figure(figsize=(10, 5))
    df[column].value_counts().plot(kind='bar', cmap= cm)
    plt.title(title,  fontsize=15)
    plt.xlabel(xlabel,  fontsize=13)
    plt.ylabel('Count' , fontsize=13)
    plt.xticks(rotation=0, fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    return plt

def plot_pie(column , title):
    value_counts = df[column].value_counts()
    categories = value_counts.index
    num_categories = len(categories)
    colors = cm.viridis(np.linspace(0, 1, num_categories))
    plt.pie(value_counts, labels = None , autopct='%2.1f%%', colors = colors , startangle = 140 , textprops={'fontsize':6})
    plt.legend(categories, loc='upper left', fontsize=6)
    plt.title(title, fontsize=10, fontweight='bold')

    plt.tight_layout()
    return plt



#####################
####STREAMLIT########
#####################
overview = st.sidebar.button('OVERVIEW')
if overview:
  st.title("Content-Based Recommender System Workflow")

  st.markdown("## Overview")
  st.write("This document outlines the workflow of the content-based recommender system implemented in this project. The system provides two types of recommendations based on the user's search query and search history: 'Similar Courses' and 'Personalized Recommendations'.")

  st.markdown("### 1. Similar Courses")
  st.write("Recommendations are made by finding courses that are most similar to the currently searched course using a similarity index.")

  st.markdown("### 2. Personalized Recommendations")
  st.write("Recommendations are personalized based on the user's search history. The system tracks the user's search history and updates a user vector. Recommendations are made by computing the dot product of the user vector with the course vectors in the retrieval list. The retrieval list consists of the top 10 most similar courses to every searched course up to the current search.")

  st.markdown("## Workflow")

  st.markdown("### 1. User Searches for a Course")
  st.write("- User inputs a search query for a course.")
  st.write("- The system processes the search query and retrieves a list of courses based on similarity index.")

  st.markdown("### 2. Display Similar Courses")
  st.write("- The system presents a list of courses that are most similar to the searched course.")
  st.write("- This list is termed 'Similar Courses'.")

  st.markdown("### 3. Update User Vector")
  st.write("- If it's the user's first search, the system creates a user vector.")
  st.write("- The system updates the user vector based on the searched course.")

  st.markdown("### 4. Form Retrieval List")
  st.write("- The system forms a retrieval list consisting of the top 10 most similar courses to every course searched by the user up to the current search.")
  st.write("- Similarity index is used to determine the similarity between courses.")

  st.markdown("### 5. Personalized Recommendations")
  st.write("- The system computes personalized recommendations by performing a dot product of the user vector with the course vectors in the retrieval list.")
  st.write("- Recommendations are based on the user's search history and preferences.")

  st.markdown("### 6. Display Personalized Recommendations")
  st.write("- The system presents personalized recommendations to the user based on their search history.")
  st.write("- These recommendations are tailored to the user's preferences and past searches.")

  st.markdown("## Conclusion")
  st.write("This workflow outlines the functionality and steps involved in the content-based recommender system implemented in this project. By providing both similar courses and personalized recommendations, the system aims to enhance the user experience and assist users in discovering relevant courses efficiently.")


st.title("Course Recommender System")

st.sidebar.header('EDA')
distributions = st.sidebar.button('Distributions')

if distributions:
  st.subheader('DISTRIBUTIONS')
  plot_distribution('course_certificate_type' , 'Distribution of Certificate Types' , 'Certificate Type', 'viridis')
  st.pyplot()

  plot_distribution('course_time' , 'Distribution of Course Durations' , 'Course Duration', 'plasma')
  st.pyplot()

  plot_distribution('course_difficulty' , 'Distribution of Course Difficulties' , 'Difficulty Level', 'inferno')
  st.pyplot()

pie_charts = st.sidebar.button('Proportions')
if pie_charts:
  st.subheader('PROPORTIONS')
  plot_pie('course_certificate_type' , 'Certificate Types by portion')
  st.pyplot()

  plot_pie('course_time' , 'Course Duration by portion')
  st.pyplot()

  plot_pie('course_difficulty' , 'Course Difficulty by portion')
  st.pyplot()

#########################
#SEARCH & RECOMMENDATION#
#########################

if 'searched_courses' not in st.session_state:
    st.session_state.searched_courses = []

search_course = st.selectbox('Search a course : ', df['course_title'].values )

search_button = st.button('Search')

if search_button:
  st.header(search_course)
  st.subheader('Course Description')
  description = df[df['course_title']== search_course].course_description.values
  if len(description) > 0:
        st.write(description[0])
  else:
        st.write("No description available for this course.")

  st.session_state.searched_courses.append(search_course)
  st.write(st.session_state.searched_courses)
  def recommend(course):
    course_index = df_for_training[df_for_training['course_title']== course].index[0]
    distance_list = list(enumerate(similarity_matrix[course_index]))
    sorted_distance = sorted(distance_list , reverse = True, key = lambda x:x[1])[1:6]
    recommendation_index = sorted_distance


    recommended_courses = []
    for i in recommendation_index:
          recommended_courses.append(df_for_training.iloc[i[0]].course_title)


    return recommended_courses

  similar_courses = recommend(search_course)


  st.subheader('SIMILAR COURSES')
  st.write(f"{idx}. {course} \n" for idx, course in enumerate(similar_courses, start=1))


  #########################
  ##UPDATING USER VECTOR###
  #########################
  def retrieve_course_vector(course):
    search_course_index = df_for_training[df_for_training['course_title']== course].index[0]
    search_course_vector = final_vector[search_course_index]
    return search_course_vector

  course_vector = retrieve_course_vector(search_course)
  searched_courses_count = searched_courses_count+1
  st.session_state.user_vector = update_user_vector(st.session_state.user_vector ,course_vector)


  ########################
  ###RETRIEVAL STAGE######
  ########################

  def retrieval(any_list):
    searched_courses_indexes = []
    recommendation_indexes = []
    for i in range(len(any_list)):
      searched_course_index = df_for_training[df_for_training['course_title'] == any_list[i]].index[0]
      searched_courses_indexes.append(searched_course_index)

    for x in searched_courses_indexes:
      distance_list = list(enumerate(similarity_matrix[x]))
      sorted_distance = sorted(distance_list , reverse = True, key = lambda x:x[1])[1:11]
      recommendation_index = [i[0] for i in sorted_distance]
      for r in recommendation_index:
        recommendation_indexes.append(r)

    recommended_courses = []
    for i in recommendation_indexes:
      # for j in range(len(i)):
          recommended_courses.append(df_for_training.iloc[i].course_title)



    return recommendation_indexes , recommended_courses

  st.session_state.retrieved_indexes, st.session_state.retrieved_titles =  retrieval(st.session_state.searched_courses)

  def retrieve_vectors(indexes):
    retrieved_vectors = np.array([ final_vector[i] for i in indexes])
    return retrieved_vectors

  st.session_state.retrived_vectors = retrieve_vectors(st.session_state.retrieved_indexes)


  #########################
  ###DOTPRODUCT & RANKING##
  #########################
  def dot_product(user, retrived_courses):
    dot_products = np.dot(user.reshape(1,8167),retrived_courses.T)

    dot_product_list = [(i+1,dot_products[0][i]) for i in range(retrived_courses.shape[0])]
    return dot_product_list

  st.session_state.dot_product_list = dot_product(st.session_state.user_vector, st.session_state.retrived_vectors)

  def rank_dot_products_and_recommend(dot_prodlist, retrieval_list):

      sorted_dot_prod = sorted(dot_prodlist , reverse = True, key = lambda x:x[1])[1:6]
      best_recommendations = [i[0] for i in sorted_dot_prod]
      #for i in best_recommendations:
      best_recommendations_index = [retrieval_list[i-1] for i in best_recommendations]
      return best_recommendations_index

  index_after_ranking = rank_dot_products_and_recommend(st.session_state.dot_product_list , st.session_state.retrieved_indexes)

  st.subheader('OTHER RECOMMENDED COURSES FOR YOU:')
  for idx, index in enumerate(index_after_ranking , start =1):
    recommended_title = df_for_training.iloc[index].course_title
    st.write(f"{idx}. {recommended_title}")

