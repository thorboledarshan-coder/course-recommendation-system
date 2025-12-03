import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

@st.cache_data
def load_data(csv_path: str):
    df = pd.read_csv(csv_path)

    # Fill missing values
    for col in ['name', 'category', 'what_you_learn', 'skills', 'language', 'instructors', 'content']:
        if col in df.columns:
            df[col] = df[col].fillna('')

    # Drop rows without name
    df = df.dropna(subset=['name'])

    # Combined text field
    df['text'] = (
        df['name'] + ' ' +
        df['category'] + ' ' +
        df['skills'] + ' ' +
        df['what_you_learn'] + ' ' +
        df['content'] + ' ' +
        df['language'] + ' ' +
        df['instructors']
    )

    # Lowercase name for matching
    df['name_lower'] = df['name'].str.lower()

    return df

@st.cache_resource
def build_tfidf_and_similarity(df: pd.DataFrame):
    # TF-IDF
    tfidf = TfidfVectorizer(stop_words='english', max_features=20000)
    tfidf_matrix = tfidf.fit_transform(df['text'])

    # Cosine similarity
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

    # Mapping name -> index
    indices = pd.Series(df.index, index=df['name_lower']).drop_duplicates()

    return tfidf, tfidf_matrix, cosine_sim, indices

def recommend_similar_courses(df, cosine_sim, indices, course_name, n=10):
    course_name_lower = course_name.lower()

    if course_name_lower not in indices:
        st.warning("Course not found. Please check the name or choose from the dropdown.")
        return pd.DataFrame()

    idx = indices[course_name_lower]

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Skip the same course
    sim_scores = sim_scores[1:n+1]

    course_indices = [i[0] for i in sim_scores]

    result = df.iloc[course_indices].copy()
    result = result[['name', 'category', 'skills', 'language', 'url']]

    return result

#this below code is only for recommending on the basis of query

def recommend_by_user_query(df, tfidf, tfidf_matrix, query, n=10):
    query_vec = tfidf.transform([query])
    sim_scores = linear_kernel(query_vec, tfidf_matrix).flatten()

    top_indices = sim_scores.argsort()[::-1][:n]

    result = df.iloc[top_indices].copy()
    result = result[['name', 'category', 'skills', 'language', 'url']]
    result['similarity'] = sim_scores[top_indices]

    return result

#this below code is for streamlit app
def main():
    st.set_page_config(
        page_title="Course Recommendation System",
        page_icon="🎓",
        layout="wide"
    )

    st.title("🎓 Course Recommendation System")
    st.write("Get course recommendations based on existing courses or your learning interests.")

    # Sidebar for file path (kept simple as fixed CSV)
    #st.sidebar.header("Settings")
    csv_path = "coursera_courses.csv"
    #st.sidebar.write(f"Using dataset: `{csv_path}`")
    # Load data
    df = load_data(csv_path)
    tfidf, tfidf_matrix, cosine_sim, indices = build_tfidf_and_similarity(df)

    tab1, tab2 = st.tabs(["🔁 Similar Courses", "🧠 Query-based Recommendations"])

    with tab1:
        st.subheader("🔁 Find Courses Similar to a Selected Course")

        # Dropdown of course names
        course_names = df['name'].sort_values().unique().tolist()
        selected_course = st.selectbox("Select a course", course_names)

        num_recs = st.slider("Number of recommendations", 3, 20, 10)

        if st.button("Recommend Similar Courses"):
            with st.spinner("Finding similar courses..."):
                recs = recommend_similar_courses(df, cosine_sim, indices, selected_course, n=num_recs)
            if not recs.empty:
                st.success(f"Courses similar to: **{selected_course}**")
                for i, row in recs.iterrows():
                    st.markdown(f"### {row['name']}")
                    st.write(f"**Category:** {row['category']}")
                    st.write(f"**Skills:** {row['skills']}")
                    st.write(f"**Language:** {row['language']}")
                    st.write(f"[Course Link]({row['url']})")
                    st.markdown("---")

    with tab2:
        st.subheader("🧠 Get Recommendations from Your Learning Interests")

        query = st.text_area(
            "Describe what you want to learn (e.g., 'beginner python for data analysis and SQL')",
            height=80
        )
        num_recs_query = st.slider("Number of recommendations", 3, 20, 10, key="query_slider")

        if st.button("Recommend Based on My Interests"):
            if not query.strip():
                st.warning("Please enter something in the text box.")
            else:
                with st.spinner("Searching matching courses..."):
                    recs_query = recommend_by_user_query(df, tfidf, tfidf_matrix, query, n=num_recs_query)
                if not recs_query.empty:
                    st.success("Here are some courses that match your interests:")
                    for i, row in recs_query.iterrows():
                        st.markdown(f"### {row['name']}")
                        st.write(f"**Category:** {row['category']}")
                        st.write(f"**Skills:** {row['skills']}")
                        st.write(f"**Language:** {row['language']}")
                        st.write(f"**Similarity score (for debug):** {row['similarity']:.3f}")
                        st.write(f"[Course Link]({row['url']})")
                        st.markdown("---")

if __name__ == "__main__":
    main()
