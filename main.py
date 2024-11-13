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
from statsmodels.tsa.arima.model import ARIMA



#Initializing Data
df = pd.read_csv('Data/movies_data.csv')

import os
print(os.getcwd())


try:
    movie_df = pd.read_csv('Data/movies_data.csv')
except UnicodeDecodeError:
    movie_df = pd.read_csv('Data/movies_data.csv')

print(movie_df)


#Explatory Data Analysis
movie_df.head()
movie_df.duplicated()
movie_df.info()
movie_df.isnull().sum()
movie_df = movie_df.dropna()
movie_df.head()
movie_df.info()
movie_df['Budget'].unique()
movie_df['Budget'] = pd.to_numeric(movie_df['Budget'], errors='coerce')

budget_counts = movie_df['Budget'].value_counts()
print(budget_counts)

highest_budget_row = movie_df.loc[movie_df['Budget'].idxmax()]
print(highest_budget_row)

plt.figure(figsize=(10, 6))
plt.bar(budget_counts.index, budget_counts.values, color='black', alpha=0.7, edgecolor='darkseagreen')
plt.title('Frequency Distribution of Budget')
plt.xlabel('Budget Bins (in millions)')
plt.ylabel('Frequency')
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.show()

movie_df['IMDb score'].unique()

imdb_counts = movie_df['IMDb score'].value_counts()
print(imdb_counts)

movie_df['IMDb_Rounded'] = movie_df['IMDb score'].round()
print(movie_df.IMDb_Rounded)

movie_df['IMDb_Rounded'].unique()

for column in ['Movie', 'Director', 'Actor 1', 'Actor 2', 'Actor 3', 'Genre']:
    print(f"{column} unique values:\n", movie_df[column].unique(), "\n")

plt.figure(figsize=(12, 6))
top_directors = movie_df['Director'].value_counts().head(10).index
sns.countplot(y='Director', data=movie_df[movie_df['Director'].isin(top_directors)], order=top_directors)
plt.title('Directors with the Most Films in the Dataset')
plt.show()

full_actor_names = (
    movie_df['Actor 1'].astype(str).tolist() +
    movie_df['Actor 2'].astype(str).tolist() +
    movie_df['Actor 3'].astype(str).tolist()
)

actor_counts = pd.Series(full_actor_names).value_counts()
top_actors = actor_counts.head(10)
print("Top 10 Actors by Appearances:\n", top_actors)

top_actors.plot(kind='barh', color='skyblue')
plt.xlabel('Number of Appearances')
plt.ylabel('Actors')
plt.title('Top 10 Actors by Appearances')
plt.gca().invert_yaxis()
plt.show()

print("Column Names in Dataset:")
print(movie_df.columns)

# Uncover underlying structure
sns.pairplot(movie_df[['IMDb score', 'Earnings', 'Budget', 'Running time']], diag_kind='kde')
plt.show()


#Extract important variables from the dataset
features = movie_df[['IMDb score', 'Budget', 'Running time', 'Release year']]
target = movie_df['Earnings']
features = features.dropna()
target = target.loc[features.index]
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=0)
model = DecisionTreeRegressor(random_state=0)
model.fit(X_train, y_train)
importance = model.feature_importances_
feature_importance = pd.Series(importance, index=features.columns)
print("Feature Importances:")
print(feature_importance.sort_values(ascending=False))


# Detect outliers and anomalies (if any)
numeric_cols = movie_df.select_dtypes(include=['float64', 'int64']).columns
z_scores = movie_df[numeric_cols].apply(zscore)
outliers = (z_scores.abs() > 3).any(axis=1)
print("Outliers Detected (Rows):")
print(movie_df[outliers])  # Replace display with print




#Machine Learning

st.title("Movie Data Analysis")
# By CHAN, Jeska Ashley B. - Unsupervised Learning: Clustering based on Budget and IMDb Scores
st.subheader("Relationship Between Budget and IMDb Score")
fig1, ax1 = plt.subplots()
ax1.scatter(movie_df['Budget'], movie_df['IMDb_Rounded'], color='mediumvioletred', alpha=0.6)
ax1.set_title('Relationship Between Budget and IMDb Score')
ax1.set_xlabel('Budget (in millions)')
ax1.set_ylabel('IMDb Score')
st.pyplot(fig1)


#The scatterplot showing the relationship between the Budget and IMDb scores show no strong correlation between them. Based on the results of the plot, regardless of the movie's budget, its IMDb scores can vary. More movies with lesser budget exhibit a wide range of IMDb scores. This suggests that the audience rating are not directly tied to the financial resources invested in the movies. Perhaps there are other factors beyong budget that could determine IMDb ratings better such as direction, cinematography, and acting.

kmeans = KMeans(n_clusters=2, random_state=0)
movie_df['Cluster'] = kmeans.fit_predict(movie_df[['Budget', 'IMDb_Rounded']])

st.subheader("Data with Cluster Labels")
st.write(movie_df)

# Plot with KMeans clusters
st.subheader("KMeans Clustering of Budget and IMDb Scores")
fig2, ax2 = plt.subplots(figsize=(8, 6))
scatter = ax2.scatter(movie_df['Budget'], movie_df['IMDb_Rounded'], c=movie_df['Cluster'], cmap='PiYG', s=100)
ax2.set_title('KMeans Clustering of Budget and IMDb Scores')
ax2.set_xlabel('Budget')
ax2.set_ylabel('IMDb Scores')
plt.colorbar(scatter, label='Cluster')
ax2.grid(True)

# Display the second plot in Streamlit
st.pyplot(fig2)

#After applying K-Means clustering, the diagram revealed two distinct clusters, one colored magenta and the other green, which separated the data into lower-budget and higher-budget groups. However, the clustering did not demonstrate a clear pattern concerning the IMDb scores, as both groups exhibited a wide range of ratings. This observation supports the initial finding that budget does not significantly impact IMDb scores. High financial investment in a movie does not guarantee a high audience rating, and vice versa. Thus, producers should consider focusing on other factors that may influence a movie's standing. Consequently, this model alone is insufficient, and a more comprehensive model should incorporate additional variables to better ca\pture the relationships that may affect a movie's IMDb score.

# By CARRILLO, Nathaniel James C. - Supervised Learning: Content-based filter based on Movie, Director, Actor 1, Actor 2, Actor 3, and Genre

def get_recommendations(title):
    movie_info = movie_df.loc[movie_df['Movie'] == title]
    if movie_info.empty:
        return pd.DataFrame()  # Return an empty DataFrame if the movie is not found

    movie_genre = movie_info['Genre'].values[0]
    movie_year = movie_info['Release year'].values[0]
    directors = movie_info[['Director', 'Actor 1', 'Actor 2', 'Actor 3']].values.flatten()

    filtered_movies = movie_df[
        (movie_df['Genre'].str.contains(movie_genre, case=False)) &
        (movie_df['Movie'] != title)
    ].copy()

    filtered_movies['Score'] = 0
    filtered_movies.loc[filtered_movies['Director'].isin(directors), 'Score'] += 10
    filtered_movies.loc[filtered_movies['Actor 1'].isin(directors), 'Score'] += 5
    filtered_movies.loc[filtered_movies['Actor 2'].isin(directors), 'Score'] += 5
    filtered_movies.loc[filtered_movies['Actor 3'].isin(directors), 'Score'] += 5
    filtered_movies.loc[filtered_movies['Release year'].between(movie_year - 5, movie_year + 5), 'Score'] += 3

    top_recommendations = filtered_movies.sort_values(by=['Score', 'IMDb score'], ascending=[False, False]).head(10)
    
    return top_recommendations[['Movie', 'Director', 'Genre', 'IMDb score', 'Release year']]

st.title("Movie Recommendation System")
st.write("Content-based filter based on movie's Director, Genre, and Actors")
title = st.selectbox("Select a movie title for recommendations:", movie_df['Movie'].unique())

if title:
    recommendations = get_recommendations(title)
    if not recommendations.empty:
        st.subheader(f"Recommendations for: {title}")
        st.dataframe(recommendations)
    else:
        st.write("No recommendations found for the selected title.")


# By DRILON, Rafael Francisco V. - Supervised Learning: Linear regression relationshop based on 'Budget' and 'Box Office'
@st.cache_data
def load_data(filepath):
    data = pd.read_csv(filepath, encoding='ISO-8859-1')
    return data

data = load_data('Data/movies_data.csv')
st.title("Movie Data Analysis and Prediction")

# Drop rows with missing values for Budget, Running time, and Box Office
data_cleaned = data.dropna(subset=['Budget', 'Running time', 'Box Office'])

# Scatter plot of Budget vs. Box Office
st.subheader("Budget vs. Box Office (Raw Data)")
fig1, ax1 = plt.subplots(figsize=(10, 6))
sns.scatterplot(x='Budget', y='Box Office', data=data_cleaned, color='blue', alpha=0.6, ax=ax1)
ax1.set_title('Budget vs. Box Office (Raw Data)', fontsize=14)
ax1.set_xlabel('Budget ($)', fontsize=12)
ax1.set_ylabel('Box Office ($)', fontsize=12)
ax1.grid(True)
st.pyplot(fig1)

# Scatter plot of Running Time vs. Box Office
st.subheader("Running Time vs. Box Office (Raw Data)")
fig2, ax2 = plt.subplots(figsize=(10, 6))
sns.scatterplot(x='Running time', y='Box Office', data=data_cleaned, color='green', alpha=0.6, ax=ax2)
ax2.set_title('Running Time vs. Box Office (Raw Data)', fontsize=14)
ax2.set_xlabel('Running Time (Minutes)', fontsize=12)
ax2.set_ylabel('Box Office ($)', fontsize=12)
ax2.grid(True)
st.pyplot(fig2)

# Preprocess data for model training
def preprocess_data(data):
    data = data.dropna(subset=['Budget', 'Running time', 'Box Office'])
    scaler = StandardScaler()
    data[['Budget', 'Running time']] = scaler.fit_transform(data[['Budget', 'Running time']])
    X = data[['Budget', 'Running time']]
    y = data['Box Office']
    return X, y

X, y = preprocess_data(data)

# Train model and evaluate performance
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return model, X_test, y_test, y_pred, mse, r2

model, X_test, y_test, y_pred, mse, r2 = train_model(X, y)

# Display model evaluation metrics
st.subheader("Model Evaluation")
st.write("Mean Squared Error (MSE):", mse)
st.write("R-squared (R²):", r2)

# Plot actual vs. predicted Box Office values
st.subheader("Actual vs. Predicted Box Office")
fig3, ax3 = plt.subplots(figsize=(10, 6))
ax3.scatter(y_test, y_pred, color='blue')
ax3.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linewidth=2)
ax3.set_xlabel("Actual Box Office")
ax3.set_ylabel("Predicted Box Office")
ax3.set_title("Actual vs Predicted Box Office")
ax3.grid(True)
st.pyplot(fig3)

#The two graphs above illustrate how budget and running time relate to whether or not the film will be a box office success. Though many low-budget films also do well at the box office, the plots show that a larger budget typically translates into higher box office earnings, proving that budget alone does not ensure success. On the other hand, the majority of movies are between 90 and 150 minutes long, and their earnings are usually less than $500 million, indicating that running time has little bearing on box office performance. This implies that although budget may play a role, other factors probably have a greater influence on a film's box office performance than budget or running time alone.

#Budget vs. Box Office and Running Time vs. Box Office were the first scatter plots used in the analysis. These plots indicated that although budget and box office earnings are positively correlated, neither factor by itself significantly influences a film's financial success. Building on these findings, after using the learing model. The Actual vs. Predicted Box Office plot illustrates how well the supervised linear regression model predicts earnings for lower-grossing films while significantly underestimating earnings for higher-grossing ones. The model might be missing important non-linear relationships or other factors that contribute to blockbuster success, as indicated by the large deviation from the 1:1 line for high box office values. Future iterations of the model might benefit from investigating non-linear modeling and adding more factors to improve predictive accuracy.



# By HERRERA, Kael - Supervised Learning : Linear Regression based based on Movie Box Office Earnings Time Series Analysis and Forecasting
@st.cache_data
def load_data():
    data = movie_df.dropna(subset=['Release year', 'Box Office'])[['Release year', 'Box Office']]
    data = data.rename(columns={'Release year': 'Year', 'Box Office': 'Earnings'})
    data['Year'] = pd.to_datetime(data['Year'], format='%Y').dt.year
    data = data.sort_values('Year')
    return data

data = load_data()
st.title("Box Office Earnings Analysis and Forecasting")

# Plot: Box Office Earnings Over Time
st.subheader("Box Office Earnings Over Time")
fig1, ax1 = plt.subplots(figsize=(12, 6))
ax1.plot(data['Year'], data['Earnings'], color='blue', marker='o', linestyle='-')
ax1.set_title("Box Office Earnings Over Time")
ax1.set_xlabel("Year")
ax1.set_ylabel("Earnings")
ax1.grid(True)
st.pyplot(fig1)

# Compute and plot 5-Year Rolling Average
st.subheader("Box Office Earnings with 5-Year Rolling Average")
data['Rolling_Avg'] = data['Earnings'].rolling(window=5).mean()
fig2, ax2 = plt.subplots(figsize=(12, 6))
ax2.plot(data['Year'], data['Earnings'], label="Earnings", color='blue', marker='o', linestyle='-')
ax2.plot(data['Year'], data['Rolling_Avg'], label="5-Year Rolling Average", color='orange', linestyle='--')
ax2.set_title("Box Office Earnings with 5-Year Rolling Average")
ax2.set_xlabel("Year")
ax2.set_ylabel("Earnings")
ax2.legend()
ax2.grid(True)
st.pyplot(fig2)

# Train linear regression model
X = data[['Year']]
y = data['Earnings']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Plot Actual vs. Predicted Earnings
st.subheader("Actual vs. Predicted Box Office Earnings (Linear Regression)")
fig3, ax3 = plt.subplots(figsize=(10, 6))
ax3.plot(data['Year'], data['Earnings'], label='Actual Earnings', color='blue')
ax3.plot(X_test, y_pred, label='Predicted Earnings', color='orange', linestyle='--')
ax3.set_title("Box Office Earnings Over Time: Actual vs. Predicted (Linear Regression)")
ax3.set_xlabel("Year")
ax3.set_ylabel("Earnings")
ax3.legend()
st.pyplot(fig3)

# Display Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
st.write(f"Mean Squared Error (MSE): {mse}")

# Forecast for the next 5 years
st.subheader("Future Forecast of Box Office Earnings")
future_years = pd.DataFrame({'Year': np.arange(data['Year'].max() + 1, data['Year'].max() + 6)})
future_predictions = model.predict(future_years)

# Plot historical data and future forecast
fig4, ax4 = plt.subplots(figsize=(10, 6))
ax4.plot(data['Year'], data['Earnings'], label='Historical Earnings')
ax4.plot(future_years, future_predictions, label='Future Forecast', color='red', linestyle='--')
ax4.set_title("Box Office Earnings Forecast with Linear Regression")
ax4.set_xlabel("Year")
ax4.set_ylabel("Earnings")
ax4.legend()
st.pyplot(fig4)

#This line graph shows Box Office Earnings by release year. By plotting this trend, we observe fluctuations over the years and can identify any upward or downward trends. Notable peaks might correspond to high-grossing years, possibly influenced by successful blockbuster releases.

#The orange line represents a 5-year rolling average, smoothing the data to show a clearer long-term trend. This helps us reduce the impact of yearly fluctuations, giving us a clearer view of the trend—whether it’s increasing, decreasing, or staying relatively stable.

#This plot compares the **actual box office earnings with the predictions made by our linear regression model** on the test set. The orange line represents predicted values, helping us see how well the model aligns with real data. While linear regression may not capture complex fluctuations, it provides a useful approximation for long-term trend forecasting.

#Here, the red dashed line shows our forecast for the next five years based on the historical trend identified by linear regression. This projection assumes the same general pattern continues, with future values reflecting an estimated growth or decline based on the observed trend. This analysis provided a basis for understanding and forecasting box office earnings. However, if we observe complex seasonality or non-linear patterns in the data, we may consider more sophisticated models, such as ARIMA or Prophet, to improve predictive accuracy.


#Section IV: CONCLUSION

## Time Series System
#The ARIMA and Prophet models offer valuable insights into the forecast of box office earnings, each with unique strengths. ARIMA’s linear approach effectively captures steady trends, providing a straightforward outlook based on historical data. However, it may miss out on more complex, non-linear patterns often present in film industry revenues. Prophet, in contrast, incorporates seasonality and trend changes, capturing cyclical revenue fluctuations like those seen during holidays or blockbuster seasons. This makes Prophet particularly useful for predicting periods of increased or decreased revenue. Together, these models offer complementary perspectives, with ARIMA highlighting overall trends and Prophet providing nuanced, seasonally-adjusted forecasts. Leveraging both models can enhance strategic decision-making for release timing and revenue optimization.

#The time series model we created, using linear regression, is designed to predict future box office earnings based on historical data. By treating Release Year as the time-dependent feature and Box Office Earnings as the target variable, we can establish a trend over time and generate projections for future years.

 #This approach is particularly useful for capturing long-term trends in movie earnings, reflecting the impact of factors like industry growth, inflation, and audience demand over time. Although linear regression provides a straightforward estimation, it’s especially valuable here due to the consistent time-based structure of our data. However, if additional features like Genre, Budget, or Director Box Office % were incorporated in a multivariate time series model, we could gain insights into how specific characteristics influence earnings trends, enhancing the model's ability to predict outcomes more accurately by accounting for these other key factors.


## Clustering Systems
#The analysis of the relationship between movie budgets and IMDb scores, following clustering, reveals no significant correlation between the two variables. The initial scatterplot shows that a large financial investment does not guarantee a high IMDb score. This observation suggests that audience impact and ratings are not directly influenced by budget alone. Furthermore, K-Means clustering showed no consistent pattern in IMDb scores, as both low and high-budget films exhibited a similar range of ratings. Therefore, while budget alone is not a reliable predictor of IMDb scores, incorporating other qualitative factors into the model may improve its ability to explain audience ratings more effectively.


##Recommender Systems

#The machine learning algorithm initially uses filters to only display movies of the same genre and prevent it from recommending the same movie. It uses a scoring system for select attributes where the score increases when a match is made with the original movie. After several models, the attributes and their scoring were decided after we were satisfied with the recommendations being given out.

#We gave more points if the director was a match because they often create similar movies or direct the movie's sequel if there are any. This is not a limitation however as you can see in the 4th example with "Harry Potter" as the directors change frequently.

#A known limitation with the recommender system however, is that because we filtered out other genres, it is impossible for other genres to be recommended even if they are part of the same franchise/series,are directed by the same person, and they feature the same actors.

#One example is Big Momma's House and Big Momma's House 2, an action comedy featuring Martin Lawrence. Because the two movies are labeled action and comedy, respectively, in the dataset, the algorithm will not recommend the sequel when prompted because of the different genre. We believe the fix to this is by allowing multiple genres in the same manner as how actors are listed (actor 1, actor 2, actor 3).