![Alt text](image-1.png)

# 😲 Recommender System
A recommender system is like a helpful friend that suggests things you might like, based on your preferences. It's designed to recommend products, movies, or content to users, making their choices more personalized and enjoyable. 😊

# 🔥 Non-Personalized Recommender System
A non-personalized recommender system suggests the same items to all users, without considering individual preferences. An example is the "Trending" section on a streaming platform, recommending popular content to all users based on overall popularity rather than personal tastes. 📺

### 😁 Types:

Collaborative Filtering: What you liked before and what other similar to you have liked.

Imagine it's like connecting with others who share your interests, and the system suggests things loved by your like-minded buddies. 🎬

Content-Based Filtering: Think of it as a virtual taste bud explorer—it recommends things by focusing on the unique flavors you enjoy, like suggesting songs that match your favorite musical vibes. 🎶

# 😵 Matrix Factorization a.k.a SVD

Matrix Factorization is used in recommender systems to break down a large user-item interaction matrix into smaller matrices, helping identify latent factors and patterns that enhance the system's ability to make personalized and accurate recommendations. 

# 👋 Deep Learning
-- Keras
-- Matrix Factorization in Keras
-- Deep Neural Network
-- Residual Network
-- Unsupervised Deep learning
    a. Autoencoder
    b. Unsupervised Boltzman Machines

![Alt text](image-2.png)

# 😺 Dataset : 

📊 About Dataset:
The datasets illustrate ratings and free-text tagging activities from MovieLens, a movie recommendation service. With 20000263 ratings and 465564 tag applications spanning 27278 movies, the data, generated on October 17, 2016, captures insights from 138493 users who participated between January 09, 1995, and March 31, 2015. Users were randomly selected, ensuring inclusion of those who had rated at least 20 movies. 🎬📅

url: https://www.kaggle.com/datasets/grouplens/movielens-20m-dataset

# 👀 Basic Intution 

👉 1. Pouplarity: "Most of them like yo like as well

👉 2. Associated Product: Probability of purchase Product A individually 10% whereas if someone purchases Product B will purchase Product A, probability will be 20% in that case. 

# Different Website and their Algorithm
### 📢 HackerNews: 
Balancing Popularity with Age 

![Alt text](image-3.png)

📌 note: 
ups/downs: upvotes / downvotes
gravity: put more weights on rank of old news/post. Reduce the ranking score of it.
penalty : for popular side, controversial links/urls, no reason at all based on the owner/developer.
It can be possible for HackerNews to fairly or unfarily push down or push up the news using penalty.

📌 How Hacker News ranking really works: scoring, controversy, and penalties:
ref: https://www.righto.com/2013/11/how-hacker-news-ranking-really-works.html

### 📢 Reddit: 

![Alt text](image-4.png)

1. It has sublinearity like as HankerNews. Initial growth of vote count is faster and gets slower over time.
2. If net votes is positve(more upvotes), log result is poitive.
3. If net votes is negative(more downvotes), log result is negative.
4. 45000 is a chosen number of reddit developers and it reduces the impact of overwhelmingness of new post.
   
   **`--Big site like Reddit/Facebook/Google has a responsibility to remain neutral.--`**

👉 3. Rating:
   
   --5 Star Rating: Amazon, Best Buy, Walmart
   
   --Thumbs up / down Rating: Netflix

Problem of Average Rating: 

-- 5 star --> rated by user count 1

-- 4 star --> rated by user count 540

note: failed miserably. and the key point of this problem is **`Confidence`**

### Confidence 
![Alt text](image-5.png)

![Alt text](image-6.png)

![Alt text](image-8.png)

📌 As the sample size (N) increases, the standard error decreases, leading to a smaller spread in the confidence interval. A smaller spread indicates greater precision and more confidence in the estimate. In statistical terms, larger sample sizes tend to result in more reliable and accurate estimates of population parameters, reducing the variability and increasing the level of confidence in the results.
### 👾 Normal Distribution

![Alt text](image-9.png)

**📌 note:** sample may not be normally distributed. But sum of random variable convergence to normal distribution.

### 👾 Bernouli Distribution
Use for binary outcome

![Alt text](image-10.png)

**📌 note:**  p-hat (total upvote / total votes)

### 👾 Wilson Interval (Better for Bernouli Distribution then Bernoulli)
The Wilson interval provides more accurate estimates for binary outcomes because it adjusts for continuity correction (account for the continuity between discrete points), offering better coverage probabilities, especially when the probability of success is close to 0 or 1.
![Alt text](image-11.png)


Absolutely! Here's the complete comedy competition:

- **Confidence Interval for Normal Distribution:**
  - Imagine you're also guessing the average number of laughs per comedian. You use a confidence interval, saying, "I'm 95% confident the average laughs for both Bernie and Wacky Wilson is between 50 and 100 per performance."

- **Bernoulli Interval (Bernie):**
  - Bernie's jokes are like flipping a fair coin. You estimate the laughter interval for each joke, saying, "I'm 95% sure between 30% and 70% of the audience will laugh."

- **Wilson Interval (Wacky Wilson):**
  - Now, Wacky Wilson's jokes are a bit wild. You use the Wilson interval to predict how many people will laugh or raise an eyebrow, saying, "I'm 95% confident between 20% and 80% will react to Wilson's unpredictable humor."

So, not only are you predicting laughter intervals for the jokes, but you're also confidently estimating the average laughs for the entire comedy competition!

- **😊 5 Star Rating(5 Possible Outcome)**
   - Bernoulli Interval doesn't work
   -  Wilson Interval Still Works

🌟 note: Hacker News uses same sorting algorithm for both url and comments. Whereas Reddit uses wilson interval for sorting comments.

Key notes: Lower Boundary is pessimistic (negative perciption)
example, one 3 star rating on good product can be penalized. 

# 😵 More Problem with Average Rating
  👉 1. Smoothing (Damping): use in NLP and Page Rank
   
  In the context of **NLP and average rating**, smoothing or damping often refers to techniques used to handle cases where there are limited data points or extreme ratings.

**🔥key note:** use of Bayesian smoothing or Laplace smoothing to adjust the **sentiment score**.

  In the context of average ratings, imagine a product receiving two ratings: one very positive (5 stars) and one very negative (1 star). Without damping, the average rating would be (5 + 1) / 2 = 3, which might not accurately represent the overall satisfaction. **`Smoothing(Damping Methods)`** could be applied to mitigate the impact of extreme ratings, providing a more balanced and reliable average.

# 😈 Explore-Exploide Dilemma :

- Exploration-Exploitation Dilemma: Analyzing 10 slot machines in a casino to identify the best performer.

- Data Gathering: Playing 100 games on each machine (totaling 1000 games) for a more accurate estimate.

- Dilemma Introduction: Increasing the number of games enhances precision but introduces the explore-exploit dilemma.

- Game Distribution: Dividing the 1000 games into 900 suboptimal outcomes and 100 optimal outcomes.

- Base Machine Identification: Recognizing one machine as the baseline for comparison.

- Dilemma Cause: Acknowledging that the presence of suboptimal outcomes triggers the explore-exploit dilemma.

### 🎥 Youtube Video:
**🔔 Explore-Exploit Dilemma**
Suppose I've watched YouTube videos on how to cook eggs and learned the technique. Despite this, YouTube continues to suggest these egg-related videos due to the explore-exploit dilemma. Even when I'm exploring other topics like machine learning, the explore-exploit dilemma still leads to recommendations for egg videos, even though I no longer need them.

Strategic Adjustment: Adapting the strategy to balance exploration and exploitation based on gathered data.

# 🎅 Genaralisation of above details
Idea: rank each item by a score
Ex: Hacker News, Reddit, Bandit, PageRank
# 🎁 Collaborative Filtering

Collaborative filtering is widely used in various recommendation systems, such as those found in e-commerce platforms, streaming services, and social media platforms, to provide personalized recommendations to users based on their historical interactions and preferences.

- **Basic Average Rating:**
  score = rate user i for item J / all user who rate the item j
- **Personalize Score:**
Rating reflecting unique preferences for personalized recommendations.A personalized score might suggest a movie based on your past preferences, recommending a film with similar themes and genres that align with your viewing history.

- **Rating Matrix:**
1. user-item matrix
2. user-item matrix is reminiscent of term-document matrix in NLP
3. X(t,d) = how much times term 't' appears in a document 'd'
4. for recommender system , count of like 't' for item 'd'

# 🎑 Reletionship with NLP
1. several cross-over with NLP
2. Matrix Factorarization
3. Unsupervised Deep Learning + NLP : SVD

# 🌲 Sparsity
1. for NLP, most of them are  zero (0)
2. for recommender system, most entires are empty.
   
![Alt text](image-12.png)

📌 Sparsity helps to produce recoomender system to work perfectly. If everyone likes all the movies, there is nothing to recommend.

### Goal of Collaborative Filtering 

**Guess of user i might rate the item j**

![Alt text](image-13.png)

# 📯 Type of Collaborative Filtering
- user-user 
- item-item
### 🐇 user-user filtering
product recommendation--> similarity between users based on their historical data
Recommend **Star War** movie to Bob based on Alice Preference as both of them are similar type of users. Ratings are positive correlated.
![Alt text](image-14.png)

1. Average Rating Limitations: The average rating alone may not be a precise recommendation outcome for user-user rating.

2. Weighted Average: Introducing a weighted average can help balance the impact of ratings and derivatives, particularly when users have different rating tendencies.

3. Bias Mitigation: Deriving the difference in user rating on the item and the average rating/bias helps to address biases, such as overly optimistic or pessimistic ratings.

4. Weight Calculation: Using Pearson correlation or, preferably, Cosine correlation can help calculate weights for a more nuanced approach.

5. Derivative to Remove Bias: Employing derivatives to eliminate bias is a useful technique, and cosine correlation supports this effort.

6. K Nearest Neighbors (KNN): Utilizing KNN helps avoid considering all users for summation, focusing instead on the nearest neighbors to reduce computational costs.

### 🌐 Summary: Building Precise User-User Recommendations

The average rating alone falls short for accurate user-user recommendations. To enhance results, a weighted average is introduced, balancing rating weights and derivatives. Deriving the difference between user ratings on an item and average rating/bias mitigates biases, preventing extreme ratings(too optimistic or too pessimistic). Weight calculation can be done by Pearson Similarity and Cosine correlation, ideally through Cosine Similarity, aids in bias removal. We can avoid the sum of all users rather K Nearest Neighbors are employed for efficiency, focusing on nearest neighbors to reduce computational costs. **Generally, we used K = 25 upto 50 for collaborative filtering.**🚀

### Computational Complexity(O(N^2M))

![Alt text](image-15.png)

**📌 note:** any N^2 algorithm is not ideal. super memory comsuming. 

**💊 Overcome:**
- Subset the dataset
- consider followings
  - Top Users : Those users rated most movies
  - Top Movies: Those movies rated most times

### Movie Dataset Preprocessing



### Suggestion

**"Nothing wrong with simple, basic algorithms to start of your project. It's not a mistake."**

# Matrix Factorarization:
![Alt text](image-16.png)

![Alt text](image-17.png)

![Alt text](image-18.png)

**📌 note:** In summary, the operation (N×K)⋅(M×K)T results in a matrix of dimensions N×M.

### 😵 SVD(Single Values Decomposition)
**Singular Value Decomposition (SVD):** SVD is a mathematical technique that decomposes a matrix into three simpler matrices - U, Σ, and V^T. It is widely used in various applications, including data compression, noise reduction, and recommendation systems.

**Purpose:** discover latent factors that represent underlying patterns in user-item interactions to make personalized recommendations.

**How it works:** Given a user-item interaction matrix, SVD breaks it down into three matrices - U (user-to-concept), Σ (singular values[diagonal matrix]), and V^T (item-to-concept). Multiplying these matrices reconstructs the original matrix with estimated ratings, allowing for efficient recommendation generation.

**Example:** In a recommendation system, consider a matrix where users rate movies. SVD uncovers latent factors like "user preferences" and "movie genres." Multiplying these factors yields estimated ratings, enabling the system to suggest movies personalized to user tastes, even for films they haven't seen.

![Alt text](image-22.png)

![Alt text](image-25.png)

![Alt text](image-20.png)

### 😲 Cosine Similarity

Cosine similarity measures the cosine of the angle between two vectors and is utilized in collaborative filtering for recommendation systems to assess the similarity between users or items, providing a robust method to capture preferences even in sparse datasets.

![Alt text](image-24.png)

![Alt text](image-26.png)

**💡 Fact:** We must find the latent factor between users and items(example: movies)

### 👦 Dimensionality Reduction
Matrix Factorization is a way of dimentionality reduction. Encourage the model to **learn most important features.**

In recommendation systems, Truncated Singular Value Decomposition (SVD) is employed for dimensionality reduction to efficiently handle user-item interaction matrices (M) by reducing them from high dimensions to a more manageable size (K). This reduction not only conserves computational resources but also captures essential latent factors, enhancing collaborative filtering's accuracy and scalability.
K in a recommender system by assessing the trade-off between model complexity and performance, employing techniques like the elbow method, cross-validation, or domain knowledge to strike a balance between accuracy and computational efficiency.

![Alt text](image-27.png)