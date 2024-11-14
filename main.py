import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
import seaborn as sns
import kagglehub
import os
import sklearn
import kagglehub
import statsmodels
import streamlit as st
import altair as alt 
import io

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.cluster import KMeans
from sklearn.semi_supervised import LabelPropagation
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from scipy.stats import zscore
from sklearn.preprocessing import MinMaxScaler
from prophet import Prophet

#######################
# Page configuration
st.set_page_config(
    page_title="Film Industry Insights and Predictions", # Replace this with your Project's Title
    page_icon="Data/movie_icon.png", # You may replace this with a custom icon or emoji related to your project
    layout="wide",
    initial_sidebar_state="expanded")

alt.themes.enable("dark")

#######################

# Initialize page_selection in session state if not already set
if 'page_selection' not in st.session_state:
    st.session_state.page_selection = 'about'  # Default page

# Function to update page_selection
def set_page_selection(page):
    st.session_state.page_selection = page

# Sidebar
with st.sidebar:
    st.image("Data/movie_icon.png",use_column_width="auto")
    # Sidebar Title (Change this with your project's title)
    st.title('Film Insights and Predictions')
    
    # Page Button Navigation
    st.subheader("Pages")

    if st.button("About", use_container_width=True, on_click=set_page_selection, args=('about',)):
        st.session_state.page_selection = 'about'
    
    if st.button("Dataset", use_container_width=True, on_click=set_page_selection, args=('dataset',)):
        st.session_state.page_selection = 'dataset'

    if st.button("EDA", use_container_width=True, on_click=set_page_selection, args=('eda',)):
        st.session_state.page_selection = "eda"

    if st.button("Machine Learning", use_container_width=True, on_click=set_page_selection, args=('machine_learning',)): 
        st.session_state.page_selection = "machine_learning"

    if st.button("Conclusion", use_container_width=True, on_click=set_page_selection, args=('conclusion',)):
        st.session_state.page_selection = "conclusion"

    # Project Members
    st.subheader("Group 7 Members")
    st.markdown("1. Carrillo, James\n2. Chan, Jeska\n3. Drilon, Rafael Francisco\n4. Herrera, Kael\n3. Magat, Rolando")

#######################


#Initializing Data
df = pd.read_csv('Data/movies_data.csv', encoding='ISO-8859-1')

import os
print(os.getcwd())


try:
    movie_df = pd.read_csv('Data/movies_data.csv', encoding='ISO-8859-1')
except UnicodeDecodeError:
    movie_df = pd.read_csv('Data/movies_data.csv', encoding='ISO-8859-1')

print(movie_df)


############################
#About Page
if st.session_state.page_selection == "about":
    
    st.header("📽️ About")

    st.markdown("""
        Data analysis and visualization can be done interactively with a Streamlit web application. The application allows users to enter parameters or upload files, and it uses this information to produce insights in the form of graphs and charts. Additionally, it could execute machine learning models, making predictions or assessing performance instantly. Users can easily explore and access complex data tasks with Streamlit.\n
                """)

    st.write("")  
    st.divider() 
    
    col8 = st.columns((1, 6, 1))  
    with col8[1]:
        st.image("https://i.imgur.com/Y9RMzIi.jpeg", width=800, caption='Movies')


    st.divider() 
    st.write("")
    st.markdown("""
     ### Pages
    `Dataset`: A synopsis of the Movies Dataset that is utilized in this dashboard.  
    `EDA`: Highlights the connection between the IMDb score, Box Office, Earnings, Running time, actors and other variables in the data set.  
    `Machine Learning`: The model evaluation, feature importance, and classification report are also included on this page.  
    `Conclusion`: An overview of the findings and observations from the model training and EDA.    
        """)

#####################################
# Dataset Page
elif st.session_state.page_selection == "dataset":
    st.header("💾 Dataset")

    st.markdown("**Movies (IMDb, Earnings, and More)** by Delfina Oliva is a dataset published on Kaggle, featuring sixteen (16) columns across various fields and three thousand nine hundred seventy-four (3,974) rows of data. It offers an approach for analyzing films released from the 1930s to 2016, with categories such as the movie director, box office earnings, and IMDb scores.")
    st.markdown(""" 
                The chosen dataset includes a variety of columns that provide the group with a detailed view of the information about the featured movies. The ‘Movie’ column enters the film's title, while the ‘Director’ column names the person who oversaw the movie’s production and the creative process. The ‘Running Time’ column, on the other hand, indicates the duration of the movie in minutes. Additionally, there are three columns—’Actor 1’, ‘Actor 2’, and ‘Actor 3’—which highlight the unique actors who starred in the corresponding film. The ‘Genre’ column categorizes each movie into specific genres such as Horror, Comedy, or Action.   
                """)
   
    st.subheader("Dataset Information")
    buffer = io.StringIO()
    movie_df.info(buf=buffer)
    info_str = buffer.getvalue()
    st.text(info_str)

    st.subheader("Dataset Preview")
    st.write(movie_df.head())
    
    st.subheader("Summary Statistics")
    st.write(movie_df.describe())


    
#############################################

#Explatory Data Analysis
elif st.session_state.page_selection == "eda":
    st.header("🗃️ Exploratory Data Analysis (EDA)")
  
    st.write("Explore various insights from the movie dataset. Use the menu below to navigate through the different visualizations.")

    # Menu for Navigation
    menu = ["Dataset Overview", "Distribution and Summary", "Time-Based Analysis", "Genre Analysis", 
            "Top Performers", "Box Office and Budget Relationship", "IMDb Score Analysis"]

    choice = st.selectbox("Select a visualization", menu)

    # 1. Dataset Overview
    if choice == 'Dataset Overview':
        st.header("Dataset Overview")
        
        st.markdown("""Summary Statistics: Provides key statistical metrics such as mean, median, and standard deviation for numeric columns.""")

        st.subheader("Summary Statistics")
        st.write(df.describe())  # Summary statistics for numeric columns

        st.subheader("Missing Values")
        missing_values = df.isnull().sum()
        st.bar_chart(missing_values)  # Bar chart for missing values

    # 2. Distribution and Summary of Key Metrics
    elif choice == 'Distribution and Summary':
        st.header("Distribution and Summary of Key Metrics")

        st.markdown("""**Histograms:** Shows the distribution of key metrics like IMDb score, Running time, Budget, Box Office, and Earnings.""")

        st.markdown("""**Box Plots:** Identifies potential outliers in Budget and Box Office earnings through box plot visualizations.""")
        
        # Histograms for numeric columns
        st.subheader("Histograms")
        numeric_columns = ['IMDb score', 'Running time', 'Budget', 'Box Office', 'Earnings']

        # Arrange histograms in 2 columns
        col1, col2 = st.columns(2)
        for i, col in enumerate(numeric_columns):
            if col in df.columns:
                fig, ax = plt.subplots()
                sns.histplot(df[col].dropna(), kde=True, ax=ax)
                if i % 2 == 0:
                    col1.pyplot(fig)
                else:
                    col2.pyplot(fig)

        # Box plots for outliers (arranged in 2 columns)
        st.subheader("Box Plots")
        col1, col2 = st.columns(2)
        for i, col in enumerate(['Budget', 'Box Office']):
            if col in df.columns:
                fig, ax = plt.subplots()
                sns.boxplot(x=df[col], ax=ax)
                if i % 2 == 0:
                    col1.pyplot(fig)
                else:
                    col2.pyplot(fig)

    # 3. Time-Based Analysis
    elif choice == 'Time-Based Analysis':
        st.header("Time-Based Analysis")

        st.markdown("""**Average IMDb Score Over Time:** Line chart showing the trend of average IMDb scores over the years.
                \n**Average Budget and Box Office Per Year:** Displays the average annual budget and box office earnings over time.
                \n**IMDb Score Over Time:** Scatter plot showing the evolution of IMDb scores by release year.""")

        if 'Release year' in df.columns:
            # Arrange charts in 3 columns
            col1, col2, col3 = st.columns(3)
            
            # IMDb score over time
            df_yearly = df.groupby('Release year')['IMDb score'].mean().reset_index()
            col1.subheader("Avg IMDb Score Over Time")
            col1.line_chart(df_yearly.set_index('Release year'))

            # Box Office earnings and Budget per year
            df_financials = df.groupby('Release year')[['Budget', 'Box Office']].mean().reset_index()
            col2.subheader("Avg Budget/Box Office Per Year")
            col2.line_chart(df_financials.set_index('Release year'))

            # Scatter plot for Release year vs. IMDb score
            col3.subheader("IMDb Score Over Time")
            fig, ax = plt.subplots()
            sns.scatterplot(data=df, x='Release year', y='IMDb score', ax=ax)
            col3.pyplot(fig)

    # 4. Genre Analysis
    elif choice == 'Genre Analysis':
        st.header("Genre Analysis")

        st.markdown("""**Movie Count by Genre:** Bar chart showing the number of movies produced in each genre.
                    \n**IMDb Scores by Genre:** Box plot comparing the IMDb scores across different genres to spot variations.
                    \n**Proportion of Movies by Genre:** Pie chart showing the percentage distribution of movies across genres..""")
        
        if 'Genre' in df.columns:
            # Bar chart: Count of movies by Genre (in 2 columns)
            col1, col2 = st.columns(2)
            
            genre_counts = df['Genre'].value_counts()
            col1.subheader("Movie Count by Genre")
            col1.bar_chart(genre_counts)

            # Box plot: IMDb scores by Genre
            col2.subheader("IMDb Scores by Genre")
            fig, ax = plt.subplots()
            sns.boxplot(x='Genre', y='IMDb score', data=df, ax=ax)
            plt.xticks(rotation=90)
            col2.pyplot(fig)

            # Pie chart: Proportion of Genres (full width)
            st.subheader("Proportion of Movies by Genre")
            fig, ax = plt.subplots()
            df['Genre'].value_counts().plot.pie(autopct='%1.1f%%', ax=ax)
            ax.set_ylabel('')
            st.pyplot(fig)

    # 5. Top Performers
    elif choice == 'Top Performers':
        st.header("Top Performers")

        st.markdown("""**Top 10 Movies by IMDb Score:** Displays the top 10 highest-rated movies based on IMDb scores.
                    \n**Top 10 Grossing Movies:** Displays the top 10 movies with the highest box office earnings.
                    \n**Top Directors by Box Office Earnings:** Highlights directors with the highest total box office earnings.""")

        # Top Movies by IMDb score (2 columns)
        col1, col2 = st.columns(2)
        
        col1.subheader("Top 10 Movies by IMDb Score")
        top_imdb = df[['Movie', 'IMDb score']].sort_values(by='IMDb score', ascending=False).head(10)
        col1.write(top_imdb)

        col2.subheader("Top 10 Grossing Movies by Box Office Earnings")
        top_box_office = df[['Movie', 'Box Office']].sort_values(by='Box Office', ascending=False).head(10)
        col2.write(top_box_office)

        # Top Directors by Box Office Earnings (full width)
        st.subheader("Top Directors by Box Office Earnings")
        if 'Director' in df.columns and 'Box Office' in df.columns:
            top_directors = df.groupby('Director')['Box Office'].sum().sort_values(ascending=False).head(10)
            st.bar_chart(top_directors)


    # 6. Box Office and Budget Relationship
    elif choice == 'Box Office and Budget Relationship':
        st.header("Box Office and Budget Relationship")

        st.markdown("Budget vs Box Office Earnings: Scatter plot exploring the correlation between movie budget and box office revenue.")
        
        if 'Budget' in df.columns and 'Box Office' in df.columns:
            st.subheader("Budget vs Box Office Earnings")
            fig, ax = plt.subplots()
            sns.scatterplot(x='Budget', y='Box Office', data=df, ax=ax)
            st.pyplot(fig)

    # 7. IMDb Score Analysis
    elif choice == 'IMDb Score Analysis':
        st.header("IMDb Score Analysis")
        
        st.markdown("""**Distribution of IMDb Scores:** Histogram showing the frequency distribution of IMDb scores across all movies.
                    \n**IMDb Scores by Genre:** Box plot showing the spread of IMDb scores for each genre.""")

        # Density plot or histogram for IMDb scores (2 columns)
        col1, col2 = st.columns(2)
        
        col1.subheader("Distribution of IMDb Scores")
        fig, ax = plt.subplots()
        sns.histplot(df['IMDb score'].dropna(), kde=True, ax=ax)
        col1.pyplot(fig)

        # Box plot for IMDb scores across Genres (2 columns)
        if 'Genre' in df.columns:
            col2.subheader("IMDb Scores by Genre")
            fig, ax = plt.subplots()
            sns.boxplot(x='Genre', y='IMDb score', data=df, ax=ax)
            plt.xticks(rotation=90)
            col2.pyplot(fig)


#############################################################
if st.session_state.page_selection == "machine_learning":
    st.header("🎞️ Machine Learning")
    st.title("Box Office Earnings Forecast Analysis")

    # Load and preprocess data
    data = movie_df[['Release year', 'Box Office']].dropna()
    data = data.rename(columns={'Release year': 'Year', 'Box Office': 'Earnings'})
    data['Year'] = pd.to_datetime(data['Year'], format='%Y')
    data.set_index('Year', inplace=True)

   # Scatter Plot 1: Relationship Between Budget and IMDb Score
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Relationship Between Budget and IMDb Score")
        fig1, ax1 = plt.subplots()
        plt.scatter(movie_df['Budget'], movie_df['IMDb score'], color='mediumvioletred', alpha=0.6)
        ax1.set_title('Relationship Between Budget and IMDb Score')
        ax1.set_xlabel('Budget (in millions)')
        ax1.set_ylabel('IMDb Score')
        st.pyplot(fig1)

        st.markdown("The scatterplot suggests that there is no strong correlation between Budget and IMDb scores for lower-budget movies. However, as the budget increases, the less likely it is to receive low IMDb scores.")

        st.write("")  
        st.divider()  

    # KMeans Clustering
    kmeans = KMeans(n_clusters=2, random_state=0)
    movie_df['Cluster'] = kmeans.fit_predict(movie_df[['Budget', 'IMDb score']])

    with col2:
        st.subheader("KMeans Clustering of Budget and IMDb Scores")
        fig2, ax2 = plt.subplots()
        plt.scatter(movie_df['Budget'], movie_df['IMDb score'], c=movie_df['Cluster'], cmap='PiYG', s=100)
        ax2.set_title('KMeans Clustering of Budget and IMDb Scores')
        ax2.set_xlabel('Budget')
        ax2.set_ylabel('IMDb Scores')
        plt.colorbar(label='Cluster')
        ax2.grid(True)
        st.pyplot(fig2)

        st.markdown("K-Means clustering split the diagram into lower-budget [MAGENTA] and higher-budget [GREEN] groups. While some high-budget movies still have low IMDb scores, there is a clear trend that higher-budget movies tend to perform better.")

        st.write("")  
        st.divider()  

    # Scatter Plot 2: Budget vs. Box Office
    data_cleaned = movie_df.dropna(subset=['Budget', 'Running time', 'Box Office'])
    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Budget vs. Box Office (Raw Data)")
        fig3, ax3 = plt.subplots()
        sns.scatterplot(x='Budget', y='Box Office', data=data_cleaned, color='blue', alpha=0.6)
        ax3.set_title('Budget vs. Box Office (Raw Data)')
        ax3.set_xlabel('Budget ($)')
        ax3.set_ylabel('Box Office ($)')
        ax3.grid(True)
        st.pyplot(fig3)

        st.markdown("Though many low-budget films also do well at the box office, the plots show that a larger budget typically translates into higher box office earnings.")

        st.write("")  
        st.divider()  

    # Scatter Plot 3: Running Time vs. Box Office
    with col4:
        st.subheader("Running Time vs. Box Office (Raw Data)")
        fig4, ax4 = plt.subplots()
        sns.scatterplot(x='Running time', y='Box Office', data=data_cleaned, color='green', alpha=0.6)
        ax4.set_title('Running Time vs. Box Office (Raw Data)')
        ax4.set_xlabel('Running Time (Minutes)')
        ax4.set_ylabel('Box Office ($)')
        ax4.grid(True)
        st.pyplot(fig4)

        st.markdown("Most of the movies are between 90 and 150 minutes long, and their earnings are usually less than $500 million, indicating that running time has little bearing on box office performance.")

        st.write("")  
        st.divider() 

    # Scatter Plot: Actual vs Predicted Box Office Earnings
    col5, col6 = st.columns(2)
    X, y = data_cleaned[['Budget', 'Running time']], data_cleaned['Box Office']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    with col5:
        st.subheader("Actual vs Predicted Box Office")
        fig5, ax5 = plt.subplots()
        ax5.scatter(y_test, y_pred, color='blue')
        ax5.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linewidth=2)
        ax5.set_xlabel("Actual Box Office")
        ax5.set_ylabel("Predicted Box Office")
        ax5.set_title("Actual vs Predicted Box Office")
        ax5.grid(True)
        st.pyplot(fig5)

        st.markdown("The Actual vs. Predicted Box Office plot illustrates how well the supervised linear regression model predicts earnings for lower-grossing films while significantly underestimating earnings for higher-grossing ones.")

        st.write("")  
        st.divider() 

    # Time Series Plot: Box Office Earnings Over Time
    data = movie_df.dropna()[['Release year', 'Box Office']]
    data = data.rename(columns={'Release year': 'Year', 'Box Office': 'Earnings'})
    data['Year'] = pd.to_datetime(data['Year'], format='%Y').dt.year
    data = data.sort_values('Year')

    with col6:
        st.subheader("Box Office Earnings Over Time")
        fig6, ax6 = plt.subplots()
        ax6.plot(data['Year'], data['Earnings'], color='blue', marker='o', linestyle='-')
        ax6.set_title("Box Office Earnings Over Time")
        ax6.set_xlabel("Year")
        ax6.set_ylabel("Earnings")
        ax6.grid(True)
        st.pyplot(fig6)

        st.markdown("This line graph shows Box Office Earnings by release year. Peaks correspond to years with high-grossing box office sales from successful blockbuster releases.")

        st.write("")  
        st.divider() 

    # Time Series Plot with Rolling Average
    col7, col8 = st.columns(2)
    data['Rolling_Avg'] = data['Earnings'].rolling(window=5).mean()

    with col7:
        st.subheader("Box Office Earnings with 5-Year Rolling Average")
        fig7, ax7 = plt.subplots()
        ax7.plot(data['Year'], data['Earnings'], label="Earnings", color='blue', marker='o', linestyle='-')
        ax7.plot(data['Year'], data['Rolling_Avg'], label="5-Year Rolling Average", color='orange', linestyle='--')
        ax7.set_xlabel("Year")
        ax7.set_ylabel("Earnings")
        ax7.legend()
        ax7.grid(True)
        st.pyplot(fig7)

        st.markdown("The orange line represents a 5-year rolling average, smoothing the data to show a clearer long-term trend. This helps us reduce the impact of yearly fluctuations, giving us a clearer view of the trend—whether it’s increasing, decreasing, or staying relatively stable.")

        st.write("")  
        st.divider() 

    # Time Series Forecasting with Linear Regression
    future_years = pd.DataFrame({'Year': np.arange(data['Year'].max() + 1, data['Year'].max() + 6)})
    X, y = data[['Year']], data['Earnings']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    future_predictions = model.predict(future_years)

    with col8:
        st.subheader("Box Office Earnings Forecast with Linear Regression")
        fig8, ax8 = plt.subplots()
        ax8.plot(data['Year'], data['Earnings'], label='Historical Earnings', color='blue')
        ax8.plot(future_years['Year'], future_predictions, label='Future Forecast', color='red', linestyle='--')
        ax8.set_title("Box Office Earnings Forecast")
        ax8.set_xlabel("Year")
        ax8.set_ylabel("Earnings")
        ax8.legend()
        st.pyplot(fig8)

        st.markdown("The red dashed line shows the forecast for the next five years based on the historical trend identified by linear regression. This projection assumes the same general pattern continues, with future values reflecting an estimated growth or decline based on the observed trend. ")

        st.write("")  
        st.divider() 

    # Prophet Forecasting
    movie_df.columns = movie_df.columns.str.strip()
    movie_df['Release year'] = pd.to_datetime(movie_df['Release year'], errors='coerce', format='%Y')
    movie_df.dropna(subset=['Release year'], inplace=True)
    movie_df = movie_df.groupby('Release year').agg({'Box Office': 'sum'}).reset_index()
    movie_df.columns = ['ds', 'y']

    # Initialize and fit Prophet model
    model_prophet = Prophet(yearly_seasonality=True)
    model_prophet.fit(movie_df)

    # Create future dataframe and forecast
    future = model_prophet.make_future_dataframe(periods=12, freq='Y')
    forecast = model_prophet.predict(future)

    # Prophet Forecast in col7
    with col7:
        st.subheader("Forecasted Box Office Earnings with Prophet")
        fig_prophet1 = model_prophet.plot(forecast)
        plt.title("Box Office Earnings Forecast using Prophet")
        plt.xlabel("Year")
        plt.ylabel("Box Office Earnings")
        st.pyplot(fig_prophet1)

        st.markdown("""
            The Prophet model applied to box office earnings in these graphs shows a strong upward trend in revenue over time, especially noticeable after the 1990s. The trend component reveals a steady increase, which accelerates sharply in recent years, suggesting that box office earnings have grown significantly, potentially due to factors like higher ticket prices, a greater number of releases, or increased global demand. The yearly seasonality plot highlights cyclic fluctuations in earnings, with peaks and troughs within each year, possibly aligned with major film release periods like summer and the holiday season. Prophet’s forecast with confidence intervals (in blue) indicates anticipated continued growth with periodic variations, showing Prophet's ability to capture both long-term trends and seasonal effects in box office revenue.
            """)

    # Prophet Component Plot in col8
    with col8:
        st.subheader("Forecast Components with Prophet")
        fig_prophet2 = model_prophet.plot_components(forecast)
        st.pyplot(fig_prophet2)

        st.write("")
        st.markdown("""
            The LSTM model's charts for box office earnings show a strong upward trend captured in both historical data predictions and future forecasts. The first chart illustrates a steep increase in projected earnings, suggesting the model has identified an accelerating growth trend, though it may overstate future earnings if unchecked by real-world constraints. The second chart, comparing actual and predicted values on test data, shows that the model successfully captures the overall upward trajectory but overestimates earnings in some areas, indicating it could benefit from further tuning to handle short-term fluctuations. Overall, the model is effective but may need adjustments for greater accuracy in near-term predictions.
            """)

# Conclusions Page
elif st.session_state.page_selection == "conclusion":
    st.header("📒 Conclusion")
    # Your content for the CONCLUSION page goes here
    st.markdown("""

# Time Series System
The ARIMA and Prophet models offer valuable insights into the forecast of box office earnings, each with unique strengths. ARIMA’s linear approach effectively captures steady trends, providing a straightforward outlook based on historical data. However, it may miss out on more complex, non-linear patterns often present in film industry revenues. Prophet, in contrast, incorporates seasonality and trend changes, capturing cyclical revenue fluctuations like those seen during holidays or blockbuster seasons. This makes Prophet particularly useful for predicting periods of increased or decreased revenue. Together, these models offer complementary perspectives, with ARIMA highlighting overall trends and Prophet providing nuanced, seasonally-adjusted forecasts. Leveraging both models can enhance strategic decision-making for release timing and revenue optimization.

The time series model we created, using linear regression, is designed to predict future box office earnings based on historical data. By treating Release Year as the time-dependent feature and Box Office Earnings as the target variable, we can establish a trend over time and generate projections for future years.

 This approach is particularly useful for capturing long-term trends in movie earnings, reflecting the impact of factors like industry growth, inflation, and audience demand over time. Although linear regression provides a straightforward estimation, it’s especially valuable here due to the consistent time-based structure of our data. However, if additional features like Genre, Budget, or Director Box Office % were incorporated in a multivariate time series model, we could gain insights into how specific characteristics influence earnings trends, enhancing the model's ability to predict outcomes more accurately by accounting for these other key factors.


## Clustering Systems
The analysis of the relationship between movie budgets and IMDb scores, following clustering, reveals no significant correlation between the two variables. The initial scatterplot shows that a large financial investment does not guarantee a high IMDb score. This observation suggests that audience impact and ratings are not directly influenced by budget alone. Furthermore, K-Means clustering showed no consistent pattern in IMDb scores, as both low and high-budget films exhibited a similar range of ratings. Therefore, while budget alone is not a reliable predictor of IMDb scores, incorporating other qualitative factors into the model may improve its ability to explain audience ratings more effectively.
    """)

    st.subheader("Recommender Systems")

    st.markdown("""
    The machine learning algorithm initially uses filters to only display movies of the same genre and prevent it from recommending the same movie. It uses a scoring system for select attributes where the score increases when a match is made with the original movie. After several models, the attributes and their scoring were decided after we were satisfied with the recommendations being given out.

    We gave more points if the director was a match because they often create similar movies or direct the movie's sequel if there are any. This is not a limitation however as you can see in the 4th example with "Harry Potter" as the directors change frequently.

    A known limitation with the recommender system however, is that because we filtered out other genres, it is impossible for other genres to be recommended even if they are part of the same franchise/series,are directed by the same person, and they feature the same actors.

    One example is Big Momma's House and Big Momma's House 2, an action comedy featuring Martin Lawrence. Because the two movies are labeled action and comedy, respectively, in the dataset, the algorithm will not recommend the sequel when prompted because of the different genre. We believe the fix to this is by allowing multiple genres in the same manner as how actors are listed (actor 1, actor 2, actor 3).""")
