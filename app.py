import streamlit as st
import pandas as pd

# global variable
feedback_dataframe = pd.DataFrame(columns=['Name', 'Mail', 'course_suggested', 'Feedback'])

def intro():
    import streamlit as st

    st.write("# Welcome to UMBC Course Recommender! 👋")
    st.sidebar.success("Select a tool above.")

    st.markdown(
        """
        UMBC Course recommender helps you find the right course aligned with your interests and goals
        & helps you learn if we are a good fit for you!
        **👈 Select a demo from the sidebar** to explore for yourself!
    """
    )

def admission_selection():
    import streamlit as st
    import pandas as pd
    from joblib import load
    import numpy as np
    from scipy.stats import skew
    import time

    st.markdown(f"# {list(page_names_to_funcs.keys())[2]}")
    st.write(
        """
        This tool shows your chances of getting accepted into UMBC provided your standard test scores 🎓😊.
        """
    )

    # Utils functions
    def make_prediction(model, scaler, program_name, department, gpa, gre_score, toefl_score):
        # Create a DataFrame from the inputs
        data = pd.DataFrame({
            'Program Name': [program_name],
            'Department': [department],
            'GPA': [gpa],
            'GRE Score': [gre_score],
            'TOEFL Score': [toefl_score]
        })
        
        # Apply the same preprocessing as during training
        categorical_features = ['Program Name', 'Department']
        numerical_features = ['GPA', 'GRE Score', 'TOEFL Score']

        for col in numerical_features:
            if skew(data[col]) > 0 or skew(data[col]) < 0:
                data[col] = np.log1p(data[col])
        
        # Transform categorical data with OneHotEncoder
        data_encoded = pd.get_dummies(data, columns=categorical_features)
        training_columns = ['GPA', 'GRE Score', 'TOEFL Score',
        'Program Name_Chemical & Biochemical Engineering',
        'Program Name_Computer Engineering', 'Program Name_Computer Science',
        'Program Name_Cybersecurity', 'Program Name_Data Science',
        'Program Name_Electrical Engineering',
        'Program Name_Engineering Management',
        'Program Name_Environmental Engineering',
        'Program Name_Health Information Technology',
        'Program Name_Human-Centered Computing',
        'Department_Chemical & Biochemical Engineering',
        'Department_Civil & Environmental Engineering',
        'Department_Computer Science',
        'Department_Computer Science & Electrical Engineering',
        'Department_Electrical Engineering',
        'Department_Engineering Management', 'Department_Information Systems']
        
        data_encoded = data_encoded.reindex(columns=training_columns, fill_value=0)
        X_test_scaled = scaler.transform(data_encoded)

        prediction = model.predict(X_test_scaled)
        
        return prediction

    def load_artifacts(model_path, scaler_path):
        model = load(model_path)
        scaler = load(scaler_path)
        return model, scaler
    
    @st.cache_data
    def load_data(grades_path):
        # admission
        grades_dataset = pd.read_csv(grades_path)
        program_names = grades_dataset['Program Name'].unique().tolist()
        department_names = grades_dataset['Department'].unique().tolist()

        return program_names, department_names
    
    grades_path = r"/Users/vinayvarma/Desktop/streamlit_admissions_app/data/admissions_acceptance_dataset.csv"
    model_path = r"/Users/vinayvarma/Desktop/streamlit_admissions_app/models/admission_ensemble.joblib"  
    scaler_path = r"/Users/vinayvarma/Desktop/streamlit_admissions_app/models/scaler.joblib"

    model, scaler = load_artifacts(model_path, scaler_path)

    program_names, department_names = load_data(grades_path)
    
    # Example inputs
    program_name = st.selectbox("Select your Program", options=program_names)
    department = st.selectbox("Select your Department", options=department_names)
    gpa = st.number_input("Enter your GPA",min_value=0.0, max_value=4.0, value=3.5)
    gre_score = st.number_input("Enter your GRE score", max_value=340, value=315)
    toefl_score = st.number_input("Enter your TOEFL score", max_value=120, value=107)

    # Predict using the function
    if st.button('Admission Prediction'):
        with st.spinner('Prediction result'):
            time.sleep(3)
            pred = make_prediction(model, scaler, program_name, department, gpa, gre_score, toefl_score)
            if gre_score < 300:
                st.markdown(f"<div style='background-color:#F44336; color:white; padding:10px; border-radius:8px; font-weight:bold;'>May Not be Admitted</div>", unsafe_allow_html=True)
            else:
                st.write("Prediction:", 'Can be Admitted \n Fill up the details in feedback section' if pred[0] == 1 else 'May Not be Admitted')
                if pred[0] == 1:
                    st.markdown(f"<div style='background-color:#4CAF50; color:white; padding:10px; border-radius:8px; font-weight:bold;'>Can be Admitted</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div style='background-color:#F44336; color:white; padding:10px; border-radius:8px; font-weight:bold;'>May Not be Admitted</div>", unsafe_allow_html=True)


def course_recommendation():
    # Imports
    import streamlit as st
    import pandas as pd
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    from sentence_transformers import SentenceTransformer
    import os

    st.markdown(f'# {list(page_names_to_funcs.keys())[1]}')
    st.write(
        """
        This tool will help you select and compare courses based on your interests and career goals. Enjoy!
        """
    )

    # Utils functions
    def recommend_courses(interests, career_goals, course_df):
        # Combine interests and career goals into a single profile text
        profile_text = interests + " " + career_goals
        
        # Load a pre-trained sentence transformer model (BERT-based)
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Embed the profile text and all course descriptions
        profile_vector = model.encode([profile_text])
        course_vectors = model.encode(course_df['Matched Courses'].tolist())
        
        # Calculate cosine similarity between profile vector and course vectors
        similarity_scores = cosine_similarity(profile_vector, course_vectors)
        
        # Get indices of courses sorted by similarity (highest first)
        top_indices = similarity_scores.argsort()[0][::-1]

        # Collect the top two unique recommendations based on a similarity threshold
        unique_courses = []
        seen_descriptions = []
        for index in top_indices:
            course_description = course_df['Matched Courses'].iloc[index]
            if not any(cosine_similarity([model.encode(course_description)], [model.encode(seen)]) > 0.9 for seen in seen_descriptions):
                unique_courses.append(course_description)
                seen_descriptions.append(course_description)
                if len(unique_courses) == 2:
                    break
        
        return unique_courses
    
    @st.cache_data
    def load_data(courses_path):
        # Courses
        courses_dataset = pd.read_csv(courses_path)
        unique_interests = courses_dataset['Interests'].unique().tolist()
        unique_career_goals = courses_dataset['Career Goals'].unique().tolist()

        return courses_dataset, unique_interests, unique_career_goals
    
    courses_path = r"/Users/vinayvarma/Desktop/streamlit_admissions_app/data/courses_dataset.csv"  # Update with your actual path
    comparison_path = r"/Users/vinayvarma/Desktop/streamlit_admissions_app/data/course_comparison.csv"

    courses_dataset, unique_interests, \
        unique_career_goals = load_data(courses_path)
    
    comparison_dataset = pd.read_csv(comparison_path)

    # Dropdown for selecting interests and career goals
    interests = st.selectbox("Select your interests", options=unique_interests)
    career_goals = st.selectbox("Select your career goals", options=unique_career_goals)

    # Button to make prediction
    if st.button('Recommend Courses'):
        with st.spinner('Recommendation system churning'):
            recommendations = recommend_courses(interests, career_goals, courses_dataset)
            if len(recommendations[0].split(",")) > 1:
                recommendations = recommendations[0].split(",")
                recommendations[1] = recommendations[1].strip()
            st.write("Recommended Courses:")
            for course in recommendations:
                st.markdown(f"<div style='background-color:#ccffcc; font-size:20px; font-weight:bold; border-radius:5px; padding:10px;'>{course}</div><br>", unsafe_allow_html=True)
            st.dataframe(comparison_dataset.loc[(comparison_dataset['Program'] == recommendations[0]) | (comparison_dataset['Program'] == recommendations[1])])

    with st.form('feedback_form'):
        name = st.text_input('Your Name')
        mail = st.text_input('Your Mail')
        course_suggested = st.text_input('Course that you were suggested')
        feedback = st.text_area('Please provide your feedback so we can improve!')
        submitted_feedback = st.form_submit_button('Submit feedback')
        if submitted_feedback:
            feedback_row = pd.DataFrame([[name, mail, course_suggested, feedback]], columns=['Name', 'Mail', 'course_suggested', 'Feedback'])
            header = not os.path.exists('feedback.csv')
            feedback_row.to_csv('feedback.csv', mode='a', header=header, index=False)
            st.success("Thank you for your feedback!")

def admin():
    import streamlit as st
    import pandas as pd
    st.markdown("# Admin Page")
    st.write("This page lets the admin assess the feedback provided by users for iterative improvement of the product.")

    try:
        feedback_df = pd.read_csv('feedback.csv')
        if not feedback_df.empty:
            # Check and update session state for checkboxes
            if 'checkbox_state' not in st.session_state or len(st.session_state.checkbox_state) != len(feedback_df):
                st.session_state.checkbox_state = [False] * len(feedback_df)

            # Create columns for checkboxes and data
            checkbox_col, data_col = st.columns([1, 3])
            
            with checkbox_col:
                st.write("Select")  # Column header for checkboxes
                # Update each checkbox state with a loop
                updated_checkbox_state = []
                for i in range(len(feedback_df)):
                    checkbox = st.checkbox("", key=f"check{i}", value=st.session_state.checkbox_state[i])
                    updated_checkbox_state.append(checkbox)
                st.session_state.checkbox_state = updated_checkbox_state
            
            with data_col:
                st.dataframe(feedback_df)  # Display the DataFrame

            # Optionally, add a button to process checked items
            if st.button('Process Checked Items'):
                # Processing checked items
                checked_feedback = feedback_df[st.session_state.checkbox_state]
                st.write("Checked items processed:", checked_feedback)
        else:
            st.write("No feedback yet.")
    except FileNotFoundError:
        st.write("No feedback data found.")
        st.session_state.checkbox_state = []  # Reset checkbox state if no data is available

page_names_to_funcs = {
    "Menu": intro,
    "Course Recommender": course_recommendation,
    "Admission Checker": admission_selection,
    "Admin Page": admin
}

demo_name = st.sidebar.selectbox("Choose a demo", page_names_to_funcs.keys())
page_names_to_funcs[demo_name]()
