# 📘 Course Recommendation System (Content-Based Filtering using NLP + Machine Learning)

A machine-learning powered **Course Recommendation System** that helps learners discover relevant courses based on content similarity.  
This project analyzes course metadata (title, category, skills, learning outcomes, description, etc.) using **NLP + Scikit-Learn** to generate intelligent recommendations.

---

## 🚀 Project Overview

Online learning platforms contain thousands of courses, making it challenging for users to find the right one.  
This project solves that by building a **content-based recommender system** that:

- Suggests **similar courses** based on course content  
- Finds courses based on **user search queries** (e.g., “beginner python data analysis”)  
- Uses **TF–IDF vectorization** to convert text into numerical features  
- Calculates similarity using **cosine similarity**  
- Works effectively even without user ratings or interactions  

---

## 📂 Dataset Description

The dataset includes **5,411+ Coursera-style courses** with the following columns:

| Column           | Description                                        |
|------------------|----------------------------------------------------|
| `name`           | Course title                                       |
| `category`       | Course domain (Data Science, Business, etc.)       |
| `what_you_learn` | Key learning outcomes                              |
| `skills`         | Skills taught in the course                        |
| `language`       | Medium of instruction                              |
| `instructors`    | Instructor names                                   |
| `content`        | Full description / syllabus                        |
| `url`            | Course link                                        |

These text-rich attributes make the dataset ideal for text-based recommendation.

---

## 🧠 How the Recommendation System Works

### ✔ Step 1 — Data Cleaning
- Filled missing text values (`NaN`) with empty strings  
- Removed rows without course titles  
- Normalized text to ensure high-quality preprocessing  

### ✔ Step 2 — Text Feature Engineering
A combined text field was created:

name + category + skills + what_you_learn + content + language + instructors

sql
Copy code

This field represents the entire semantic content of each course.

### ✔ Step 3 — TF–IDF Vectorization
Converted all course text into numeric vectors using:

```python
TfidfVectorizer(stop_words='english', max_features=20000)
TF–IDF helps identify important words for each course.

✔ Step 4 — Cosine Similarity
Calculated similarity between every pair of courses:

python
Copy code
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
Higher score ⇒ more similar courses.

✔ Step 5 — Recommendation Functions
🔹 A. Course → Similar Courses
Given a course name, return the top-N most similar courses.

🔹 B. User Query → Recommended Courses
Search by text phrase like:

“python beginner data analytics”

The system returns the closest matching courses.

🧪 Sample Usage
🔍 Recommend Courses Similar to a Given Course
python
Copy code
recommend_similar_courses("Machine Learning", n=5)
🔍 Recommend Based on User Preferences
python
Copy code
recommend_by_user_query("beginner python data analysis", n=10)
Output includes:

Course name

Category

Skills

URL

Similarity score

🛠️ Technologies Used
Python

Pandas, NumPy

Scikit-Learn

TF–IDF Vectorizer

Cosine Similarity

Natural Language Processing (NLP)

Jupyter Notebook

Matplotlib / Seaborn (optional for EDA)

Streamlit (optional for UI)

📦 Project Structure
css
Copy code
course-recommendation-system/
│
├── data/
│   └── coursera_courses.csv
│
├── notebooks/
│   └── course_recommender.ipynb
│
├── src/
│   ├── recommender.py
│   ├── preprocess.py
│   └── utils.py
│
└── README.md
📊 Exploratory Data Analysis (Optional)
You can add visualizations such as:

Most common skills

Distribution of course categories

Word clouds from descriptions

Course languages

Skill frequency counts

🌟 Results
Built a fully functional content-based course recommendation system

Supports both course similarity lookup and user query search

Works efficiently on a dataset of 5,000+ courses

Generates meaningful semantic recommendations using NLP
