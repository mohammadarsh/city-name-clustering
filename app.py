# import streamlit as st
# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.cluster import DBSCAN
# from sklearn.metrics import silhouette_score

# # Function to preprocess city names
# def preprocess_city_names(city_name):
#     return city_name.strip().lower().replace(" ", "")

# # Function to perform clustering
# def perform_clustering(df, eps):
#     df['clean_name'] = df['name'].apply(preprocess_city_names)
#     df = df[df['clean_name'].str.len() > 0]
#     vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4))
#     tfidf_matrix = vectorizer.fit_transform(df['clean_name'])
#     dbscan = DBSCAN(eps=eps, min_samples=2, metric='cosine').fit(tfidf_matrix)
#     df['cluster'] = dbscan.labels_
#     df['group_label'] = df['cluster'].apply(lambda x: 'Unique' if x == -1 else f'Group {x}')
#     num_clusters = len(set(dbscan.labels_)) - (1 if -1 in dbscan.labels_ else 0)
#     silhouette_avg = silhouette_score(tfidf_matrix, dbscan.labels_, metric='cosine') if num_clusters > 1 else 'Not applicable'
#     return df, num_clusters, silhouette_avg

# # Streamlit app layout
# st.title("City Names Clustering")

# uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

# if uploaded_file is not None:
#     df = pd.read_csv(uploaded_file)
    
#     if 'name' not in df.columns:
#         st.error("The CSV file must contain a 'name' column.")
#     else:
#         eps = st.slider("Select eps value for DBSCAN", 0.1, 0.5, 0.3, 0.1)
        
#         # Perform clustering
#         df, num_clusters, silhouette_avg = perform_clustering(df, eps)
        
#         # Display results
#         st.write(f"Number of clusters: {num_clusters}")
#         st.write(f"Silhouette Score: {silhouette_avg}")
        
#         st.subheader("Cluster Data")
#         st.write(df[['name', 'clean_name', 'group_label']])
        
#         # Optional: Download updated CSV file
#         # csv = df.to_csv(index=False)
#         # st.download_button(label="Download Updated Data", data=csv, file_name='City_Names_Updated.csv', mime='text/csv')
# import streamlit as st
# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.cluster import DBSCAN
# from sklearn.metrics import silhouette_score

# # Function to preprocess city names
# def preprocess_city_names(city_name):
#     if isinstance(city_name, str):
#         return city_name.strip().lower().replace(" ", "")
#     else:
#         return ""

# # Function to perform clustering
# def perform_clustering(df, eps):
#     df['name'] = df['name'].astype(str)  # Ensure 'name' column is string
#     df['clean_name'] = df['name'].apply(preprocess_city_names)
#     df = df[df['clean_name'].str.len() > 0]  # Remove empty names
#     vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4))
#     tfidf_matrix = vectorizer.fit_transform(df['clean_name'])
#     dbscan = DBSCAN(eps=eps, min_samples=2, metric='cosine').fit(tfidf_matrix)
#     df['cluster'] = dbscan.labels_
#     df['group_label'] = df['cluster'].apply(lambda x: 'Unique' if x == -1 else f'Group {x}')
#     num_clusters = len(set(dbscan.labels_)) - (1 if -1 in dbscan.labels_ else 0)
#     silhouette_avg = silhouette_score(tfidf_matrix, dbscan.labels_, metric='cosine') if num_clusters > 1 else 'Not applicable'
#     return df, num_clusters, silhouette_avg

# # Streamlit app layout
# st.title("City Names Clustering")

# uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

# if uploaded_file is not None:
#     df = pd.read_csv(uploaded_file)

#     if 'name' not in df.columns:
#         st.error("The CSV file must contain a 'name' column.")
#     else:
#         eps = st.slider("Select eps value for DBSCAN", 0.1, 0.5, 0.3, 0.1)

#         # Perform clustering
#         df, num_clusters, silhouette_avg = perform_clustering(df, eps)

#         # Display results
#         st.write(f"Number of clusters: {num_clusters}")
#         st.write(f"Silhouette Score: {silhouette_avg}")

#         st.subheader("Cluster Data")
#         st.write(df[['name', 'clean_name', 'group_label']])

#         # # Optional: Download updated CSV file
#         # csv = df.to_csv(index=False)
#         # st.download_button(label="Download Updated Data", data=csv, file_name='City_Names_Updated.csv', mime='text/csv')




# import streamlit as st
# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.cluster import DBSCAN
# from sklearn.metrics import silhouette_score

# # Function to preprocess city names
# def preprocess_city_names(city_name):
#     if isinstance(city_name, str):
#         return city_name.strip().lower().replace(" ", "")
#     else:
#         return ""

# # Function to perform clustering and calculate silhouette score
# def perform_clustering(df, eps):
#     df['name'] = df['name'].astype(str)  # Ensure 'name' column is string
#     df['clean_name'] = df['name'].apply(preprocess_city_names)
#     df = df[df['clean_name'].str.len() > 0]  # Remove empty names
#     vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4))
#     tfidf_matrix = vectorizer.fit_transform(df['clean_name'])
#     dbscan = DBSCAN(eps=eps, min_samples=2, metric='cosine').fit(tfidf_matrix)
#     df['cluster'] = dbscan.labels_
#     df['group_label'] = df['cluster'].apply(lambda x: 'Unique' if x == -1 else f'Group {x}')
#     num_clusters = len(set(dbscan.labels_)) - (1 if -1 in dbscan.labels_ else 0)
#     silhouette_avg = silhouette_score(tfidf_matrix, dbscan.labels_, metric='cosine') if num_clusters > 1 else 'Not applicable'
#     return df, num_clusters, silhouette_avg

# # Streamlit app layout
# st.title("City Names Clustering")

# uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

# if uploaded_file is not None:
#     df = pd.read_csv(uploaded_file)

#     if 'name' not in df.columns:
#         st.error("The CSV file must contain a 'name' column.")
#     else:
#         st.write("Evaluating different eps values...")

#         # Define a range of eps values to test
#         eps_values = [0.1, 0.2, 0.3, 0.4, 0.5]
#         best_eps = None
#         best_score = -1
#         best_df = None
#         best_num_clusters = None

#         # Iterate through each eps value
#         for eps in eps_values:
#             df_temp, num_clusters, silhouette_avg = perform_clustering(df, eps)
#             if silhouette_avg != 'Not applicable' and silhouette_avg > best_score:
#                 best_score = silhouette_avg
#                 best_eps = eps
#                 best_df = df_temp
#                 best_num_clusters = num_clusters

#         if best_eps is not None:
#             st.write(f"Best eps value: {best_eps}")
#             st.write(f"Number of clusters: {best_num_clusters}")
#             st.write(f"Silhouette Score: {best_score}")

#             st.subheader("Cluster Data")
#             st.write(best_df[['name', 'clean_name', 'group_label']])

#             # Optional: Download updated CSV file
#             csv = best_df.to_csv(index=False)
#             st.download_button(label="Download Updated Data", data=csv, file_name='City_Names_Updated.csv', mime='text/csv')
#         else:
#             st.error("No suitable clusters found.")







# import streamlit as st
# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.cluster import DBSCAN
# from sklearn.metrics import silhouette_score

# # Function to preprocess city names
# def preprocess_city_names(city_name):
#     if isinstance(city_name, str):
#         return city_name.strip().lower().replace(" ", "")
#     else:
#         return ""

# # Function to perform clustering and calculate silhouette score
# def perform_clustering(df, column_name, eps, min_samples=2):
#     df[column_name] = df[column_name].astype(str)  # Ensure the column is string
#     df['clean_name'] = df[column_name].apply(preprocess_city_names)
#     df = df[df['clean_name'].str.len() > 0]  # Remove empty names
#     vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4))
#     tfidf_matrix = vectorizer.fit_transform(df['clean_name'])
#     dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine').fit(tfidf_matrix)
#     df['cluster'] = dbscan.labels_
#     df['group_label'] = df['cluster'].apply(lambda x: 'Unique' if x == -1 else f'Group {x}')
#     num_clusters = len(set(dbscan.labels_)) - (1 if -1 in dbscan.labels_ else 0)
#     silhouette_avg = silhouette_score(tfidf_matrix, dbscan.labels_, metric='cosine') if num_clusters > 1 else 'Not applicable'
#     return df, num_clusters, silhouette_avg

# # Introduction Page
# def introduction_page():
#     st.title("City Names Clustering App")
#     st.write("""
#     ## Introduction

#     Welcome to the City Names Clustering App!

#     This application helps you analyze and cluster city names based on their textual similarity using the DBSCAN algorithm. 

#     ### How It Works:
#     1. **Upload Your Data:** Start by uploading a CSV file that contains the city names you wish to analyze.
#     2. **Select Column:** Choose which column in your CSV file contains the city names.
#     3. **Automatic Analysis:** The app will automatically test various `eps` values to find the one that produces the best clustering result based on the Silhouette Score.
#     4. **Analyze and View Results:** The app will display the clustering results, including the number of clusters and the Silhouette Score, and provide a visual representation of the clusters.
#     5. **Download Results:** You can download the updated CSV file with cluster information.

#     ### Features:
#     - **Dynamic Column Selection:** Choose any column from your uploaded file for clustering.
#     - **Automatic Best `eps` Selection:** The app evaluates different `eps` values to find the best clustering performance.
#     - **Clustering Visualization:** Visualize the clusters in a 2D plot.
#     - **Download Data:** Download the CSV file with the clustering results.

#     Feel free to explore and experiment with different datasets and settings. If you have any questions or need assistance, please contact support.
#     """)

# # Clustering Page
# def clustering_page():
#     st.title("City Names Clustering")

#     uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

#     if uploaded_file is not None:
#         df = pd.read_csv(uploaded_file)

#         # Normalize column names
#         df.columns = [col.strip().lower() for col in df.columns]

#         # Show available columns for selection
#         st.write("Available columns:", df.columns)

#         column_name = st.selectbox("Select the column containing city names", options=df.columns)

#         if column_name:
#             st.write("Evaluating different eps values...")

#             # Define a range of eps values to test
#             eps_values = [0.1, 0.2, 0.3, 0.4, 0.5]
#             best_eps = None
#             best_score = -1
#             best_df = None
#             best_num_clusters = None

#             # Iterate through each eps value
#             for eps in eps_values:
#                 df_temp, num_clusters, silhouette_avg = perform_clustering(df, column_name, eps)
#                 if silhouette_avg != 'Not applicable' and silhouette_avg > best_score:
#                     best_score = silhouette_avg
#                     best_eps = eps
#                     best_df = df_temp
#                     best_num_clusters = num_clusters

#             if best_eps is not None:
#                 st.write(f"Best eps value: {best_eps}")
#                 st.write(f"Number of clusters: {best_num_clusters}")
#                 st.write(f"Silhouette Score: {best_score}")

#                 st.subheader("Cluster Data")
#                 st.write(best_df[[column_name, 'clean_name', 'group_label']])

#                 # Optional: Download updated CSV file
#                 csv = best_df.to_csv(index=False)
#                 st.download_button(label="Download Updated Data", data=csv, file_name='City_Names_Updated.csv', mime='text/csv')
#             else:
#                 st.error("No suitable clusters found.")
#         else:
#             st.error("No column selected. Please select a column containing city names.")

# # Main Page
# def main():
#     st.sidebar.title("Navigation")
#     selection = st.sidebar.radio("Go to", ["Introduction", "Clustering"])

#     if selection == "Introduction":
#         introduction_page()
#     elif selection == "Clustering":
#         clustering_page()

# if __name__ == "__main__":
#     main()


# import streamlit as st
# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.cluster import DBSCAN
# from sklearn.metrics import silhouette_score

# # Function to preprocess city names
# def preprocess_city_names(city_name):
#     if isinstance(city_name, str):
#         return city_name.strip().lower().replace(" ", "")
#     else:
#         return ""

# # Function to perform clustering and calculate silhouette score
# def perform_clustering(df, column_name, eps, min_samples=2):
#     df[column_name] = df[column_name].astype(str)  # Ensure the column is string
#     df['clean_name'] = df[column_name].apply(preprocess_city_names)
#     df = df[df['clean_name'].str.len() > 0]  # Remove empty names
#     vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4))
#     tfidf_matrix = vectorizer.fit_transform(df['clean_name'])
#     dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine').fit(tfidf_matrix)
#     df['cluster'] = dbscan.labels_
#     df['group_label'] = df['cluster'].apply(lambda x: 'Unique' if x == -1 else f'Group {x}')
#     num_clusters = len(set(dbscan.labels_)) - (1 if -1 in dbscan.labels_ else 0)
#     silhouette_avg = silhouette_score(tfidf_matrix, dbscan.labels_, metric='cosine') if num_clusters > 1 else 'Not applicable'
#     return df, num_clusters, silhouette_avg

# # Introduction Page
# def introduction_page():
#     st.title("🌟 City Names Clustering App 🌟")

#     st.write("""
#     ## Welcome to the City Names Clustering App!

#     This application provides a powerful tool to analyze and cluster city names based on their textual similarity. Using the DBSCAN algorithm, it helps you discover patterns and group similar city names effectively.

#     ### 🚀 How It Works:
#     1. **Upload Your Data:** 
#         - Upload a CSV file containing city names you wish to analyze.
#     2. **Select Column:** 
#         - Choose the column in your CSV file that contains the city names for clustering.
#     3. **Automatic Analysis:** 
#         - The app automatically evaluates various `eps` values to determine the best clustering result based on the Silhouette Score.
#     4. **View Results:** 
#         - The results include the best `eps` value, number of clusters, and Silhouette Score. A visual representation of the clusters will also be provided.
#     5. **Download Results:** 
#         - Download the updated CSV file with the clustering information for further analysis.

#     ### ✨ Features:
#     - **Dynamic Column Selection:** 
#         - Select any column from your file for clustering.
#     - **Automatic `eps` Selection:** 
#         - The app finds the optimal `eps` value for the best clustering performance.

#     - **Data Download:** 
#         - Easily download the CSV file with clustering results.

   

#     """)

   

# # Clustering Page
# def clustering_page():
#     st.title("City Names Clustering")

#     uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

#     if uploaded_file is not None:
#         df = pd.read_csv(uploaded_file)

#         # Normalize column names
#         df.columns = [col.strip().lower() for col in df.columns]

#         # Show available columns for selection
#         st.write("Available columns:", df.columns)

#         column_name = st.selectbox("Select the column containing city names", options=df.columns)

#         if column_name:
#             st.write("Evaluating different eps values...")

#             # Define a range of eps values to test
#             eps_values = [0.1, 0.2, 0.3, 0.4, 0.5]
#             best_eps = None
#             best_score = -1
#             best_df = None
#             best_num_clusters = None

#             # Iterate through each eps value
#             for eps in eps_values:
#                 df_temp, num_clusters, silhouette_avg = perform_clustering(df, column_name, eps)
#                 if silhouette_avg != 'Not applicable' and silhouette_avg > best_score:
#                     best_score = silhouette_avg
#                     best_eps = eps
#                     best_df = df_temp
#                     best_num_clusters = num_clusters

#             if best_eps is not None:
#                 st.write(f"Best eps value: {best_eps}")
#                 st.write(f"Number of clusters: {best_num_clusters}")
#                 st.write(f"Silhouette Score: {best_score}")

#                 st.subheader("Cluster Data")
#                 st.write(best_df[[column_name, 'clean_name', 'group_label']])

#                 # Optional: Download updated CSV file
#                 csv = best_df.to_csv(index=False)
#                 st.download_button(label="Download Updated Data", data=csv, file_name='City_Names_Updated.csv', mime='text/csv')
#             else:
#                 st.error("No suitable clusters found.")
#         else:
#             st.error("No column selected. Please select a column containing city names.")

# # Main Page
# def main():
#     st.sidebar.title("Clustering App")
#     selection = st.sidebar.selectbox("Go to", ["Introduction", "Clustering"])

#     if selection == "Introduction":
#         introduction_page()
#     elif selection == "Clustering":
#         clustering_page()

# if __name__ == "__main__":
#     main()




# import streamlit as st
# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.cluster import DBSCAN
# from sklearn.metrics import silhouette_score
# import chardet



# # Function to preprocess city names
# def preprocess_city_names(city_name):
#     if isinstance(city_name, str):
#         return city_name.strip().lower().replace(" ", "")
#     else:
#         return ""

# # Function to perform clustering and calculate silhouette score
# def perform_clustering(df, column_name, eps, min_samples=2):
#     df[column_name] = df[column_name].astype(str)  # Ensure the column is string
#     df['clean_name'] = df[column_name].apply(preprocess_city_names)
#     df = df[df['clean_name'].str.len() > 0]  # Remove empty names
#     vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4))
#     tfidf_matrix = vectorizer.fit_transform(df['clean_name'])
#     dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine').fit(tfidf_matrix)
#     df['cluster'] = dbscan.labels_
#     df['group_label'] = df['cluster'].apply(lambda x: 'Unique' if x == -1 else f'Group {x}')
#     num_clusters = len(set(dbscan.labels_)) - (1 if -1 in dbscan.labels_ else 0)
#     silhouette_avg = silhouette_score(tfidf_matrix, dbscan.labels_, metric='cosine') if num_clusters > 1 else 'Not applicable'
#     return df, num_clusters, silhouette_avg

# # Function to detect file encoding
# def detect_encoding(file):
#     raw_data = file.read()
#     result = chardet.detect(raw_data)
#     file.seek(0)  # Reset file pointer to the beginning
#     return result['encoding']

# # Introduction Page
# def introduction_page():
#     st.title("🌟 City Names Clustering App 🌟")

#     st.write("""
#     ## Welcome to the City Names Clustering App!

#     This application provides a powerful tool to analyze and cluster city names based on their textual similarity. Using the DBSCAN algorithm, it helps you discover patterns and group similar city names effectively.

#     ### 🚀 How It Works:
#     1. **Upload Your Data:** 
#         - Upload a CSV file containing city names you wish to analyze.
#     2. **Select Column:** 
#         - Choose the column in your CSV file that contains the city names for clustering.
#     3. **Automatic Analysis:** 
#         - The app automatically evaluates various `eps` values to determine the best clustering result based on the Silhouette Score.
#     4. **View Results:** 
#         - The results include the best `eps` value, number of clusters, and Silhouette Score. A visual representation of the clusters will also be provided.
#     5. **Download Results:** 
#         - Download the updated CSV file with the clustering information for further analysis.

#     ### ✨ Features:
#     - **Dynamic Column Selection:** 
#         - Select any column from your file for clustering.
#     - **Automatic `eps` Selection:** 
#         - The app finds the optimal `eps` value for the best clustering performance.

#     - **Data Download:** 
#         - Easily download the CSV file with clustering results.

   

#     """)


# # Clustering Page
# def clustering_page():
#     st.title("City Names Clustering")

#     uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

#     if uploaded_file is not None:
#         # Detect file encoding
#         encoding = detect_encoding(uploaded_file)
#         # st.write(f"Detected file encoding: {encoding}")

#         # Read CSV file with detected encoding
#         try:
#             df = pd.read_csv(uploaded_file, encoding=encoding)
#             st.write("File loaded successfully.")
#         except Exception as e:
#             st.error(f"Error reading file: {e}")
#             return

#         # Normalize column names
#         df.columns = [col.strip().lower() for col in df.columns]

#         # Show available columns for selection
#         st.write("Available columns:", df.columns)

#         column_name = st.selectbox("Select the column containing city names", options=df.columns)

#         if column_name:
#             st.write("Evaluating different eps values...")

#             # Define a range of eps values to test
#             eps_values = [0.1, 0.2, 0.3, 0.4, 0.5]
#             best_eps = None
#             best_score = -1
#             best_df = None
#             best_num_clusters = None

#             # Iterate through each eps value
#             for eps in eps_values:
#                 df_temp, num_clusters, silhouette_avg = perform_clustering(df, column_name, eps)
#                 if silhouette_avg != 'Not applicable' and silhouette_avg > best_score:
#                     best_score = silhouette_avg
#                     best_eps = eps
#                     best_df = df_temp
#                     best_num_clusters = num_clusters

#             if best_eps is not None:
#                 st.write(f"Best eps value: {best_eps}")
#                 st.write(f"Number of clusters: {best_num_clusters}")
#                 st.write(f"Silhouette Score: {best_score}")

#                 st.subheader("Cluster Data")
#                 st.write(best_df[[column_name, 'clean_name', 'group_label']])

#                 # Optional: Download updated CSV file
#                 csv = best_df.to_csv(index=False)
#                 st.download_button(label="Download Updated Data", data=csv, file_name='City_Names_Updated.csv', mime='text/csv')
#             else:
#                 st.error("No suitable clusters found.")
#         else:
#             st.error("No column selected. Please select a column containing city names.")

# # Main Page
# def main():
#     st.sidebar.title("Clustering App")
#     page = st.sidebar.selectbox("Choose a page", ["Introduction", "Clustering"])

#     if page == "Introduction":
#         introduction_page()
#     elif page == "Clustering":
#         clustering_page()

# if __name__ == "__main__":
#     main()

import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
import chardet

# Function to preprocess city names
def preprocess_city_names(city_name):
    if isinstance(city_name, str):
        return city_name.strip().lower().replace(" ", "")
    else:
        return ""

# Function to perform clustering and calculate silhouette score
def perform_clustering(df, city_column, eps, min_samples=2):
    df[city_column] = df[city_column].astype(str)  # Ensure the column is string
    df['clean_name'] = df[city_column].apply(preprocess_city_names)
    df = df[df['clean_name'].str.len() > 0]  # Remove empty names
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4))
    tfidf_matrix = vectorizer.fit_transform(df['clean_name'])
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine').fit(tfidf_matrix)
    df['cluster'] = dbscan.labels_
    df['group_label'] = df['cluster'].apply(lambda x: 'Unique' if x == -1 else f'Group {x}')
    num_clusters = len(set(dbscan.labels_)) - (1 if -1 in dbscan.labels_ else 0)
    silhouette_avg = silhouette_score(tfidf_matrix, dbscan.labels_, metric='cosine') if num_clusters > 1 else 'Not applicable'
    return df, num_clusters, silhouette_avg

# Function to get the representative value for a cluster
def get_representative_value(cluster_df, city_column):
    return cluster_df[city_column].mode().iloc[0]

# Function to detect file encoding
def detect_encoding(file):
    raw_data = file.read()
    result = chardet.detect(raw_data)
    file.seek(0)  # Reset file pointer to the beginning
    return result['encoding']

# Introduction Page
def introduction_page():
    st.title("🌟 City Names Clustering App 🌟")

    st.write("""
    ## Welcome to the City Names Clustering App!

    This application provides a powerful tool to analyze and cluster city names based on their textual similarity. Using the DBSCAN algorithm, it helps you discover patterns and group similar city names effectively.

    ### 🚀 How It Works:
    1. **Upload Your Data:** 
        - Upload a CSV file containing city names you wish to analyze.
    2. **Select Column:** 
        - Choose the column in your CSV file that contains the city names for clustering.
    3. **Automatic Analysis:** 
        - The app automatically evaluates various `eps` values to determine the best clustering result based on the Silhouette Score.
    4. **Select Accurate Values:** 
        - The app will automatically select the most frequent value within each cluster as the representative value. You can also manually adjust the representative values.
    5. **View Results:** 
        - The results include the best `eps` value, number of clusters, and Silhouette Score. A visual representation of the clusters will also be provided.
    6. **Download Results:** 
        - Download the updated CSV file with the clustering information for further analysis.

    ### ✨ Features:
    - **Dynamic Column Selection:** 
        - Select any column from your file for clustering.
    - **Automatic `eps` Selection:** 
        - The app finds the optimal `eps` value for the best clustering performance.
    - **Manual Adjustment:** 
        - Manually adjust the representative values for each cluster.
    - **Data Download:** 
        - Easily download the CSV file with clustering results.
    """)

# Clustering Page
def clustering_page():
    st.title("City Names Clustering")

    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
        # Detect file encoding
        encoding = detect_encoding(uploaded_file)

        # Read CSV file with detected encoding
        try:
            df = pd.read_csv(uploaded_file, encoding=encoding)
            st.write("File loaded successfully.")
        except Exception as e:
            st.error(f"Error reading file: {e}")
            return

        # Normalize column names
        df.columns = [col.strip().lower() for col in df.columns]

        # Show available columns for selection
        st.write("Available columns:", df.columns)

        city_column = st.selectbox("Select the column containing city names", options=df.columns)

        if city_column:
            st.write("Evaluating different eps values...")

            # Define a range of eps values to test
            eps_values = [0.1, 0.2, 0.3, 0.4, 0.5]
            best_eps = None
            best_score = -1
            best_df = None
            best_num_clusters = None

            # Iterate through each eps value
            for eps in eps_values:
                df_temp, num_clusters, silhouette_avg = perform_clustering(df, city_column, eps)
                if silhouette_avg != 'Not applicable' and silhouette_avg > best_score:
                    best_score = silhouette_avg
                    best_eps = eps
                    best_df = df_temp
                    best_num_clusters = num_clusters

            if best_eps is not None:
                st.write(f"Best eps value: {best_eps}")
                st.write(f"Number of clusters: {best_num_clusters}")
                st.write(f"Silhouette Score: {best_score}")

                # Add accurate values to dataframe
                best_df['accurate_value'] = best_df.groupby('cluster')[city_column].transform(
                    lambda x: get_representative_value(best_df[best_df['cluster'] == x.name], city_column))

                st.subheader("Cluster Data with Accurate Values")
                # Editable table to manually adjust accurate values
                edited_rows = st.data_editor(best_df[[city_column, 'clean_name', 'group_label', 'accurate_value']])
                
                # Apply the manual adjustments
                best_df.update(edited_rows)

                st.subheader("Updated Cluster Data with Manual Adjustments")
                st.write(best_df[[city_column, 'clean_name', 'group_label', 'accurate_value']])

                # Download the updated clustered data
                csv = best_df.to_csv(index=False)
                st.download_button(label="Download Updated Data", data=csv, file_name='City_Names_Updated.csv', mime='text/csv')
            else:
                st.error("No suitable clusters found.")
        else:
            st.error("No column selected. Please select a column containing city names.")

# Main Page
def main():
    st.sidebar.title("Clustering App")
    page = st.sidebar.selectbox("Choose a page", ["Introduction", "Clustering"])

    if page == "Introduction":
        introduction_page()
    elif page == "Clustering":
        clustering_page()

if __name__ == "__main__":
    main()





