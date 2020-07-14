<img target="_blank" src="snapshots/Netflix.png" width=215>

* # Netflix Recommendation - Implementation of Prize winner's solution

- [Overview:](#overview)
- [Feature Extraction](#feature-extraction)
  - [1. Baseline](#1-baseline)
  - [2. KNNBaseline predictor](#2-knnbaseline-predictor)
  - [3. SVD](#3-svd)
  - [4. SVD++](#4-svd)
- [Key aspects](#key-aspects)
- [Best Model](#best-model)
  - [Feature Importance](#feature-importance)
- [Technologies Used:](#technologies-used)
- [Credits](#credits)
- [Creator:](#creator)


## Overview:

source: https://www.kaggle.com/netflix-inc/netflix-prize-data

Netflix is all about connecting people to the movies they love. To help customers find those movies, they developed world-class movie recommendation system: CinematchSM. Its job is to predict whether someone will enjoy a movie based on how much they liked or disliked other movies. Netflix use those predictions to make personal movie recommendations based on each customer’s unique tastes. And while Cinematch is doing pretty well, it can always be made better.

The movie rating files contain over 100 million ratings from 480 thousand randomly-chosen, anonymous Netflix customers over 17 thousand movie titles. The data were collected between October, 1998 and December, 2005 and reflect the distribution of all ratings received during this period.  The ratings are on a scale from 1 to 5 (integral) stars. To protect customer privacy, each customer id has been replaced with a randomly-assigned id.  The date of each rating and the title and year of release for each movie id are also provided.

CustomerID,Rating,Date

- MovieIDs range from 1 to 17770 sequentially.
- CustomerIDs range from 1 to 2649429, with gaps. There are 480189 users.
- Ratings are on a five star (integral) scale from 1 to 5.
- Dates have the format YYYY-MM-DD.

## Feature Extraction

Source: https://surprise.readthedocs.io/en/stable/prediction_algorithms_package.html

Basic features like Global average of all movie ratings, Average rating per user, and Average rating per movie, top 5 similar users and movies ratings. Apart from these Predictions from following Machine learning models are added.

### 1. Baseline  
__Predicted_rating : ( baseline prediction )__
  
![](https://latex.codecogs.com/gif.latex?%5Clarge%20%7B%5Chat%7Br%7D_%7Bui%7D%20%3D%20b_%7Bui%7D%20%3D%5Cmu%20&plus;%20b_u%20&plus;%20b_i%7D)


- ![](http://latex.codecogs.com/png.latex?%5Cfn_cm%20%5Cpmb%20%5Cmu) :  Average of all ratings in training data.
* ![](http://latex.codecogs.com/png.latex?%5Cfn_cm%20%5Cpmb%20b_u) : User bias
* ![](http://latex.codecogs.com/png.latex?%5Cfn_cm%20%5Cpmb%20b_i) : Item bias (movie biases)



**Solved using Optimization function (Least Squares Problem)**:

![](http://latex.codecogs.com/png.latex?%5Cfn_cm%20%5Clarge%20%5Csum_%7Br_%7Bui%7D%20%5Cin%20R_%7Btrain%7D%7D%20%5Cleft%28r_%7Bui%7D%20-%20%28%5Cmu%20&plus;%20b_u%20&plus;%20b_i%29%5Cright%29%5E2%20&plus;%20%5Clambda%20%5Cleft%28b_u%5E2%20&plus;%20b_i%5E2%20%5Cright%29.%5Ctext%20%7B%20%5Bmimimize%20%7D%20%7Bb_u%2C%20b_i%5D)

### 2. KNNBaseline predictor
 **based on User-User and movie-movie similarity**

![](http://latex.codecogs.com/png.latex?%5Cfn_cm%20%5Clarge%20%5Chat%7Br%7D_%7Bui%7D%20%3D%20b_%7Bui%7D%20&plus;%20%5Cfrac%7B%20%5Csum%5Climits_%7Bv%20%5Cin%20N%5Ek_i%28u%29%7D%20%5Ctext%7Bsim%7D%28u%2C%20v%29%20%5Ccdot%20%28r_%7Bvi%7D%20-%20b_%7Bvi%7D%29%7D%20%7B%5Csum%5Climits_%7Bv%20%5Cin%20N%5Ek_i%28u%29%7D%20%5Ctext%7Bsim%7D%28u%2C%20v%29%7D)

- ![](http://latex.codecogs.com/png.latex?%5Cfn_cm%20%5Cpmb%7Bb_%7Bui%7D%7D) -  _Baseline prediction_ of (user,movie) rating

- ![]( http://latex.codecogs.com/png.latex?%5Cpmb%20%7BN_i%5Ek%20%28u%29%7D) - Set of __K similar__ users (neighbours) of __user (u)__ who rated __movie(i)__  

- _sim (u, v)_ - __Similarity__ between users __u and v__  
    - Generally, it will be cosine similarity or Pearson correlation coefficient. 
    - But we use __shrunk Pearson-baseline correlation coefficient__, which is based on the pearsonBaseline similarity (we take base line predictions instead of mean rating of user/item)

__similarly Predicted rating based on Item Item similarity:__
  
![](http://latex.codecogs.com/png.latex?%5Clarge%20%5Chat%7Br%7D_%7Bui%7D%20%3D%20b_%7Bui%7D%20&plus;%20%5Cfrac%7B%20%5Csum%5Climits_%7Bj%20%5Cin%20N%5Ek_u%28i%29%7D%20%5Ctext%7Bsim%7D%28i%2C%20j%29%20%5Ccdot%20%28r_%7Buj%7D%20-%20b_%7Buj%7D%29%7D%20%7B%5Csum%5Climits_%7Bj%20%5Cin%20N%5Ek_u%28j%29%7D%20%5Ctext%7Bsim%7D%28i%2C%20j%29%7D)

### 3. SVD

__Predicted Rating :__

![](http://latex.codecogs.com/png.latex?%5Clarge%20%5Chat%20r_%7Bui%7D%20%3D%20%5Cmu%20&plus;%20b_u%20&plus;%20b_i%20&plus;%20q_i%5ETp_u)
    
- ![](http://latex.codecogs.com/png.latex?%5Cpmb%20q_i) - Representation of item(movie) in latent factor space
- ![](http://latex.codecogs.com/png.latex?%5Cpmb%20p_u) - Representation of user in new latent factor space

__Optimization problem with user item interactions and regularization (to avoid overfitting)__

![](http://latex.codecogs.com/png.latex?%5Clarge%20%5Csum_%7Br_%7Bui%7D%20%5Cin%20R_%7Btrain%7D%7D%20%5Cleft%28r_%7Bui%7D%20-%20%5Chat%7Br%7D_%7Bui%7D%20%5Cright%29%5E2%20&plus;%20%5Clambda%5Cleft%28b_i%5E2%20&plus;%20b_u%5E2%20&plus;%20%7C%7Cq_i%7C%7C%5E2%20&plus;%20%7C%7Cp_u%7C%7C%5E2%5Cright%29)

### 4. SVD++

__Predicted Rating :__

![](http://latex.codecogs.com/png.latex?%5Clarge%20%5Chat%7Br%7D_%7Bui%7D%20%3D%20%5Cmu%20&plus;%20b_u%20&plus;%20b_i%20&plus;%20q_i%5ET%5Cleft%28p_u%20&plus;%20%7CI_u%7C%5E%7B-%5Cfrac%7B1%7D%7B2%7D%7D%20%5Csum_%7Bj%20%5Cin%20I_u%7Dy_j%5Cright%29)

- ![](http://latex.codecogs.com/png.latex?%5Cpmb%7BI_u%7D) - the set of all items rated by user u

- ![](http://latex.codecogs.com/png.latex?%5Cpmb%7By_j%7D) - Our new set of item factors that capture implicit ratings.
   
__Optimization problem with user item interactions and regularization (to avoid overfitting)__

![](http://latex.codecogs.com/png.latex?%5Clarge%20%5Csum_%7Br_%7Bui%7D%20%5Cin%20R_%7Btrain%7D%7D%20%5Cleft%28r_%7Bui%7D%20-%20%5Chat%7Br%7D_%7Bui%7D%20%5Cright%29%5E2%20&plus;%20%5Clambda%5Cleft%28b_i%5E2%20&plus;%20b_u%5E2%20&plus;%20%7C%7Cq_i%7C%7C%5E2%20&plus;%20%7C%7Cp_u%7C%7C%5E2%20&plus;%20%7C%7Cy_j%7C%7C%5E2%5Cright%29)

## Key aspects

* Since ```date```  feature available on which rating were given Train and Test data were splited based on time in 80:20 ratio. 
* The models were trained on sample data of size 100000. In Test data, only those users and movies sampled that are appeared atleast once in the trainset.

## Best Model
Predictions from baseline models and manually extracted features are fit against ratings using **Xgboost** algorithm. Best model achived an **RMSE** of **0.967** on test data.

```py
{'n_estimators': 250,
  'learning_rate': 0.05,
  'max_depth': 6,
  'min_child_weight': 3,
  'gamma': 0.2,
  'colsample_bytree': 0.3,
  'eta': 0.1}
  ```



### Feature Importance
![](snapshots/feature_importances.png)


## Technologies Used:


![](https://forthebadge.com/images/badges/made-with-python.svg)

[<img target="_blank" src="https://scikit-learn.org/stable/_static/scikit-learn-logo-small.png" width=170>](https://scikit-learn.org/stable/#)
[<img target="_blank" src="snapshots/surprise.svg" width=215>](https://scikit-learn.org/stable/#)
 
 ## Credits
 * https://www.kaggle.com/netflix-inc/netflix-prize-data
* Netflix blog: https://medium.com/netflix-techblog/netflix-recommendations-beyond-the-5-stars-part-1-55838468f429
* surprise library: http://surpriselib.com/ (we use many models from this library)
* surprise library doc: http://surprise.readthedocs.io/en/stable/getting_started.html (we use many models from this library)
* installing surprise: https://github.com/NicolasHug/Surprise#installation
* Research paper: http://courses.ischool.berkeley.edu/i290-dm/s11/SECURE/a1-koren.pdf (most of our work was inspired by this paper)
 
</br>

------
## Creator:
[<img target="_blank" src="https://media-exp1.licdn.com/dms/image/C4D03AQG-6F3HHlCTVw/profile-displayphoto-shrink_200_200/0?e=1599091200&v=beta&t=WcZLox9lzVQqIDJ2-5DsEhNFvEE1zrZcvkmcepJ9QH8" width=150>](https://skumar-djangoblog.herokuapp.com/)