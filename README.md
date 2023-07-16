# EDA-hypothesis-testing-and-price-prediction-on-Mercari-dataset
This case study is about running EDA to understand the data better, and draw meaning insights that may be helpful in better decision making. Hypothesis testing is done to statistically support the conclusion drawn from EDA . After text processing on data model for price prediction were made. 


## Contents
- Introduction
- Exploratory Data Analysis
- Hypothesis Testing
- Pre-Processing of data before modelling
- Modelling
- Comparison of model performance and conclusions
- Further Studies
- Reference

## 1. Introduction

### 1.1 About Company
Mercari is a shopping app which is now operational in Japan, US, and India. It is Japan's biggest community-powered shopping app. Mercari allows users to buy and sell items quickly from their smartphones.  <br>

On such apps, where there is wide range of products and prices.  For e.g. A T shirt may have different price ranges depending on its condition, brand, material and so on. Hence, it can be very hard to predict the correct price of an item.
<br>
### 1.2 Objective
- Using this Mercari dataset and do some EDA to draw conclusions that will help us gain better insights about data.
- Using the concepts of hypothesis testing and statistics, support the conclusions drawn earlier for population data.
- Predict the prices of the items using regression models.
### 1.3 Concepts used:
- Data Cleaning and pre-processing (including text processing with NLTK library, and gensim library)
- EDA
- Hypothesis testing (using scipy library for statistical test)
- Predictions using regression models (using sklearn library)

### 1.4 Data Source
We will use the dataset for Mercari that is available on Kaggle. There are two types of datasets available, one for training and other for testing. For analysis we have used the training dataset, which contains 1.4 million records and use about 79.2 MB of memory. There are 7 columns present in the dataset. The dataset is in the form of .tsv format.
### 1.5 Column profile
Before we move ahead with the analysis, we need to understand what each column is about. The description of each feature is as follows:
- *train_id or test_id* - the id of the listing
- *name* - the title of the listing. Note that we have cleaned the data to remove text that look like prices (e.g. $20) to avoid leakage. These removed prices are represented as [rm]
- *item_condition_id* - the condition of the items provided by the seller
- *category_name* - category of the listing
- *brand_name*
- *price* - the price that the item was sold for. This is the target variable that you will predict. The unit is USD. This column doesn't exist in test.tsv since that is what you will predict.
- *shipping* - 1 if shipping fee is paid by seller and 0 by buyer
- *item_description* - the full description of the item. Note that we have cleaned the data to remove text that look like prices (e.g. $20) to avoid leakage. These removed prices are represented as [rm]

## 2. Exploratory Data Analysis
<br>
This is the most important and time consuming part for any data science project. As you have to use imagination to dig out the important information or insights while ensuring high interpretability. This helps us in understanding the data better and for other stakeholders too which further helps in making better business decisions. This also helps in understanding how target variable are behaving with different features. This will further help us in feature engineering. <br>

Before moving ahead with EDA we will first, import some libraries like pandas, numpy, matplotlib etc. Then we will be reading the data. Note that data is in .tsv format

This is how our data looks like:
We don't need this feature train_id for our analysis, so we will drop this feature. 
<br>
**Data type of features**
<br>
- There are 7 columns present in the data with around 1.48 million rows.
- There are 4 columns with data type object while remaining are of continuous data type.
-  
**Check for Null values**

- 42.67% of null values are present in the feature brand_name.
- Other features like item_description and category_name have less 0.5% of null values present. 

**Check for duplicates**
- There are 49 (0.0033%) rows that are duplicate in dataset of size 1.4 million rows. Since, it is such a small number so we can delete them.
<br>
As per the prices setting guidelines from Mercari, it is suggested that sellers keep the price of product in the range of $1 - $2,000. There are 874 (0.0589%) items present in the dataset whose prices doesn't belong to this range. So, we will keep these products out of our analysis.

### 2.1 Univariate Analysis

#### a. price 
We will check the distribution of price (Target variable) because this is the feature that we want to predict using regression algorithms.
<br>

![image](https://github.com/Asin-30/EDA-hypothesis-testing-and-price-prediction-on-Mercari-dataset/assets/69243814/d3f596d6-2a33-46fd-baa6-ce0a3676323c)

The distribution appears to be rightly skewed. Since, most of the statistical tests and algorithms requires the data to be normally distributed. Another advantage is that regression algorithms especially linear regression performs better for normally distributed dependent variable. So, we will use log transformation to transform the data. 
<br>

![image](https://github.com/Asin-30/EDA-hypothesis-testing-and-price-prediction-on-Mercari-dataset/assets/69243814/3cee8e80-c78c-4879-9842-30583bb701fc)

<br>

The plot we get post log transformation look like normal (not statistically proven) and right skewness has reduced. 
Since, the dataset is huge, to prevent un-necessary memory usage we will drop price from our dataset for further analysis. For our further analysis we will use (log transformed) price.

#### b. item_condition_id
If item_condition_id is 1 then it means that the item is in best possible condition.
<br>

![image](https://github.com/Asin-30/EDA-hypothesis-testing-and-price-prediction-on-Mercari-dataset/assets/69243814/09e5c1a8-8864-412f-8c57-b564b23ef445)

And as per above plot, items with condition 1 are maximum (43.20%) in number and with condition id 5 are least (0.16%). This make sense, as the product on any platform are more likely to sell if they are in their best condition.


#### c. shipping
There are two categories in shipping feature: 0 (buyer is paying for the delivery) and 1 (seller is paying for the delivery). To earn more profit, it is very understandable that sellers won't like to pay for the shipping. And this reflects in the plot (count). Shipping for 55.26% of items is paid by buyers not sellers (shipping: 0). Yet, there are a lot of sellers who believe in expanding their customer reach by making items cheaper for them, which can be possible when buyer don't have to pay for the delivery. And that is why, there are 44.73% of items for which shipping was paid by sellers.

<br>

![image](https://github.com/Asin-30/EDA-hypothesis-testing-and-price-prediction-on-Mercari-dataset/assets/69243814/de931679-afe0-409d-bb70-9307a096a05c)

<br>

#### d. brand_name

There are 4775 unique brand name present in dataset. If items of a certain brand are coming frequently than it means that the brand is popular. We can say that popularity of brand is proportional to its frequency.

<br>

There are 6.366% of the items that belong to brand PINK which comes under the umbrella of Victoria's Secret, followed by Nike (6.35%) and Vicotria's Secret (5.65%). Two out of three most popular brands are for women. This generates the possibility that women are the most frequent shoppers on the platform. 


#### e. category_name
Upon closely observing the values of this feature, you will find that there are three categories involved within separated by '/'.
First is the main category, then there are two tiers of categories ahead as well. This representation is analogus to the arrangements made in Mall. For example, Men/Tops/T-shirts means go to Men section, then tops section and then finally T-shirts section. 
We will split the data with separator as '/' and call them as 'categ_1', 'categ_2', 'categ_3'. <br>

**Now for the first level category: 'categ_1'**
<br>

![image](https://github.com/Asin-30/EDA-hypothesis-testing-and-price-prediction-on-Mercari-dataset/assets/69243814/27035357-b863-455f-a760-1f39f8ff472b)

As we anticipated earlier, Women are primary buyers on the platforms in terms of frequency. The products for the next two main categories Beauty and kids are also shopped by majorly women.
<br>

**What are women purchasing the most on the platform?**

<br>

![image](https://github.com/Asin-30/EDA-hypothesis-testing-and-price-prediction-on-Mercari-dataset/assets/69243814/13eb14bb-05e3-4321-baef-3d508452fb12)

The plot below shows that women are more into fitness and casual wear that is why they are buying Athletic and Apparel.  <br>
Around 18% of the sales in women category comes from 'Athletic Apparel'. This shows the interest of women into fitness. Approx. 16% of the sales comes from Tops & Blouses category, which shows the interest of women may be in casual and formal wears. Around 12% of the sales in women category comes from Shoes. This category involves all the foot wearables by women e.g. casual shoes, flats, sport shoes, flip flops etc.
<br>

**What is purchased most by women in sub-category of Athletic & Apparel?**

As per the plot below, one can observe that Pants, Tights, Leggings are being sold the most followed by shorts and Shirts & Tops. While Snowsuits & bibs and Vests are least popular in this category.
<br>

![image](https://github.com/Asin-30/EDA-hypothesis-testing-and-price-prediction-on-Mercari-dataset/assets/69243814/84c55881-a72c-4c29-9ffa-9e7fbb98250f)

<br>


**What sort of products women are purchasing in subcategory of Beauty?**
<br>
Women are purchasing Makeup related products the most in category of Beauty. Make up contributes around 60% of sales by women]. Next comes, the Skin Care (15% approx.) and Fragrance (12% approx.) by women in this category.
<br>

![image](https://github.com/Asin-30/EDA-hypothesis-testing-and-price-prediction-on-Mercari-dataset/assets/69243814/e40b9a69-37a6-4fc7-988f-db075defbfc7)

<br>

**What are the most popular items in sub-category, categ_2?**
<br>
The plot below make sense, because more frequent buyers are women, and they are very frequent purchasing, Athletic Apparel, Makeup and Tops & Blouses. But apart from women related items (Jewlery, Women's Handbags, Skin Care) there are also products from Electronics, which involves Cell Phones & Accessories, and Video Games & Consoles.
<br>

![image](https://github.com/Asin-30/EDA-hypothesis-testing-and-price-prediction-on-Mercari-dataset/assets/69243814/2c5cfa76-5528-4bee-954c-358f5b948090)
<br>
And the least popular items in this category are bookzines and quilts.

<br>
**What are the most popular items in sub-category, categ_3?**
<br>

There are 863 unique categories present in this feature. We will take the 15 most popular categories into consideration.
As per this plot, t-shirts, Pants, Tights, Leggings are the items that are bought most frequently. Most of the popular products plotted above are primarily for Women as they are the main consumers of such products e.g. Face, Lips, Eyes, Bras, Blouse, Tank-Cami. As for the electronics section the most popular items are Cases, Covers & Skins, Games. There is 'missing' category also, which include all the products that aren't very frequently bought.
<br>

![image](https://github.com/Asin-30/EDA-hypothesis-testing-and-price-prediction-on-Mercari-dataset/assets/69243814/4d11a004-7bfb-401b-8de8-4c8c14043361)

<br>
There are 863 values present in this category, so for visualization we can use WordCloud.
<br>

![image](https://github.com/Asin-30/EDA-hypothesis-testing-and-price-prediction-on-Mercari-dataset/assets/69243814/b1e230a4-a8d5-4bad-8bd2-3170786a3bc2)

<br>
Note: If there are more than 1 words present then we took only the last word. e.g. Legging mean the category of Pants, Tights Leggings. It is done so that it is easy to read in word cloud plots. <br>
Next two features are name and item_description. Before we can gain any insights from them we need to make sure that data is consistent.

#### Cleaning text data
<br>

Next there are text features remaining: item_description and name. We need to first clean it first, so that python doesn't take similar words as different. For e.g. in name there is value called Bundle and bundle, for python these are different but for the sake of our analysis it should be same. So, we will clean it and bring the text data in similar format. 
We will convert the text data to lower case first then remove the 'stopwords' from English. If there are null values, then replace them with 'missing'.
### f. name
Since this is text feature thus number of unique values could be huge. In order to reduce that number, we will turn every word to lower case and remove all the punctuations from it. In order to decrease its length, we will also remove the "stopwords" in English like is, the, am, of etc. using NLTK library. 
For visual representation we will be using WordCloud. The most popular items are: bundle, legging, dress, set, bag etc.
<br>

![image](https://github.com/Asin-30/EDA-hypothesis-testing-and-price-prediction-on-Mercari-dataset/assets/69243814/4762d422-beb9-4cb5-916d-f23b4ab38c8b)


### g. item_description 
<br>
There is lot of text data corresponding to each item. It will take a lot memory and space to run analysis on this feature. But later on we'll use a library (Word2Vec) to vectorize it in numerical format.
Now, we won't need name, item_description, brand_name anymore in our analysis; we have cleaned them and stored their data in new features. So, we will drop these columns from our dataset to save up space.
<br>

## 2.2 Bivariate Analysis

### a.  item condition id vs (log) price
Below are the box plot and probability distribution of (log)prices of the items and their condition id. The standard deviation for the items with condition id 5 is maximum, followed by items with condition id 1. This make sense because where the item condition is not too good then buyers won't like to pay the standard price for the item and seller would keep on changing the price till it gets sold. As for the items with condition id 1, buyers would want to buy items in its best possible condition. And if the product is too expensive, then they will ensure that condition of the product is 1. This directs a relatively big spectrum of possible prices of items in both the mentioned condition ids and hence standard deviation is slightly higher as compared to other price of items with other condition ids.
<br>
In the box plot and PDF plot, one can notice that the price distribution of price of items with condition id 5 is relatively different from distributions of price with other condition id. Since, the item condition is poor hence the price is lower than the rest.
<br>

![image](https://github.com/Asin-30/EDA-hypothesis-testing-and-price-prediction-on-Mercari-dataset/assets/69243814/3c22c0a1-e96c-41b4-a405-f14f603d3b26)

![image](https://github.com/Asin-30/EDA-hypothesis-testing-and-price-prediction-on-Mercari-dataset/assets/69243814/d64f670e-183b-47c2-80bf-a0e2addcd932)

<br>

## b. Shipping vs log_price

One can notice from the plot below that mean price of items where the buyers had to pay for shipping is higher. And this is intuitive, if the buyer is paying for the product as well as for shipping then the overall price of the item will increase. <br>

The PDF plot for the price of items where the shipping paid by buyer is little bit shifted towards right. This shows that min price of the item is also higher for buyers. The conclusions drawn above are also reflected in the box plot below:
<br>

![image](https://github.com/Asin-30/EDA-hypothesis-testing-and-price-prediction-on-Mercari-dataset/assets/69243814/3c75c08e-c527-42c0-8389-c97eec4d0d38)

The mean, min and max of price of item where shipping is paid by buyer is higher.
<br>

## c. categories vs (log)price 
We take each sub category one at a time for the analysis. First, we will start with the main category i.e. categ_1.

### c.1 categ_1 and log_price

Categ_1 has the main categories of the platform. We will draw PDFs and bar charts to check the distribution of (log) prices and know about mean and standard deviation.
We can see that distribution for the price of handmade items is quite different from other categories. Yet, electronics has the highest standard deviation and men category has the highest mean of the prices of items purchased on the platform.
<br>

![image](https://github.com/Asin-30/EDA-hypothesis-testing-and-price-prediction-on-Mercari-dataset/assets/69243814/b1fd186f-e0da-4a15-9f5b-b3ef81163059)

Also, we saw earlier that women are most frequent customers as the items getting sold more often are women oriented. But here, we can see that the women category come second to men in terms of mean sales of the items belonging to their respective categories, followed by home and then electronics.
<br>
In terms of revenue generation for the company as expected, women category tops the list. The next two categories also primarily belong to women because beauty products are mainly for women and kids can't shop for themselves, hence it is more likely that their mothers shop for them. Next comes the electronics category followed by men. <br>
<br>

![image](https://github.com/Asin-30/EDA-hypothesis-testing-and-price-prediction-on-Mercari-dataset/assets/69243814/1f383e08-140c-4fe8-ad08-541dfc802cd6)


### c.2. categ_2 and log_price

There are 113 unique categories within categ_2. We will check what are top 10 categories in terms of frequency.
we have already seen that 9.104% sales are made from Athletics & Apparel, then makeup followed by tops & blouses. If we again assume that the categories having maximum prices will have expensive products, then we can make the following plot:
<br>

![image](https://github.com/Asin-30/EDA-hypothesis-testing-and-price-prediction-on-Mercari-dataset/assets/69243814/19c12f2f-9fff-4082-b05b-8d707836404f)

One can notice that max price for jewelry is comparable to other categories as well. This suggests that there are products of luxury brands because their prices matches with that of jewelry. 
<br>
The standard deviation is least for makeup. And cell phone accessories has maximum standard deviation. Mean is highest for computers & tablets so they are the most expensive products among these categories followed by bags & purses and then women's handbags. 
Least selling categories are bathing skin care, blazers sport coats, candles etc and have least contribution in generating revenue.<br>

### c.3  categ_3 and (log)price
Considering the same assumption as before that max price of category is because of the expensive products present in them.
<br>

![image](https://github.com/Asin-30/EDA-hypothesis-testing-and-price-prediction-on-Mercari-dataset/assets/69243814/f92eb2a6-1811-4b56-acf5-658c0141d76c)


Here in the plot above one can observe that Satchel are the most expensive products which make sense, because earlier we say that items from brands PINK, Victoria's Secret and others are frequently bought by women. And these comes under the category of luxury brands and hence are expensive. Next comes the ipad, followed by shoulder bag and then messenger crossbody.

## d. brand vs log(price)
The brands that have most expensive items are Mary Kay and Chanel. NaN may include the brand that are not very famous but expensive. <br>
<br>

![image](https://github.com/Asin-30/EDA-hypothesis-testing-and-price-prediction-on-Mercari-dataset/assets/69243814/1b0c87a5-51c7-40e4-9936-c1957f539684)

One can observe that most of the brand are for clothing. Here also, we see that since women are the primary customers, the brands also such that it will attract more women.
The distribution of the price of the items from these brands is also very different from each other. It is possible because of different prices for different items and what is popular among customers.
<br>

![image](https://github.com/Asin-30/EDA-hypothesis-testing-and-price-prediction-on-Mercari-dataset/assets/69243814/ae07a1de-176c-4e08-a94b-78462be96c69)


<br>
Among all these brands Mary Kay has the least standard deviation and most popular brand among buyers. Earlier also, we saw the makeup brands has the least standard deviation, Mary Kay is also such brand, hence it's standard deviation is least. Also makeup is the second most popular sub-category, this explaining why it has high density.
Other brands offers various products clothing, bags, watches etc this explains the high standard deviations. Hence, they have low density and high standard deviation, as well as wide price range. We can also see that from the box plot below:
<br>

![image](https://github.com/Asin-30/EDA-hypothesis-testing-and-price-prediction-on-Mercari-dataset/assets/69243814/97d4f52a-bdae-4a37-aa0c-916a0bca8b30)

Among these brands, only Apple is for electronics category. <br>

**What are the popular items (names) with best and worst condition ids?**
Since there are too many variables in 'name' features so we will use WordCloud for visualization of popular items.
<br>

![image](https://github.com/Asin-30/EDA-hypothesis-testing-and-price-prediction-on-Mercari-dataset/assets/69243814/a0a9fdcc-70fd-4172-90a1-12d6e091af50)
![image](https://github.com/Asin-30/EDA-hypothesis-testing-and-price-prediction-on-Mercari-dataset/assets/69243814/b52b790f-66a1-4f2c-9dc6-94fdb97f27ea)



The most popular item that are popular among customers that they want in best condition are : bundle, legging, set, case, dresses , bags etc.
The products that customer want to buy irrespective of their bad condition are: part (may be spare part of something), purse, bag, doll, wallet etc. 
<br>

## 3. Hypothesis Testing
Now to support the observation drawn above for population data, we will run some statistical test and perform hypothesis testing. To run these test we will use scipy library's stats module.
<br>
<br>
**Is price  of the item similar for both shipping categories (0: paid by buyer, 1: paid by seller)?**
<br>

We need to declare the null and alternative hypothesis first: <br>
<br>
H0 : Mean of the prices of products for both shipping type is similar. <br>
Ha : Mean of the prices of products for both shipping type is different <br>
<br>
<br>
We will run t-test of independence for 5% significance level to check if both have similar mean. We chose this test because we don't about population mean and standard deviation and dataset is large so it will be as good as z test.
<br>
The p-values comes out to be less than 0.05, thus we are able to reject our null hypothesis. Thus, statistically we can say that prices are affected by shipping type. Prices are not independent of shipping type.
<br>

**Does item condition affect the price of items on the platform?** 
<br>
The null hypothesis and alternative hypothesis are as follows:<br>
<br>
H0 : Prices of products for different condition id have similar mean. <br>
Ha : Prices of products for different condition id have different mean.<br>
we'll run the hypothesis for 5% significance level.<br>
<br>
For this we will choose ANOVA as our statistical test as we have to check for similarity among more than two labels.<br> 
<br>
<br>
Here as well we get p_values < 0.05, this implies that item condition is dependent on price of product.<br>
<br>
Using KS test, we will also check if the distribution of prices of items is similar for condition id 1 and 2 or not. Upon running the test,we found that even the condition ids that are close to each other e.g. 1 and 2, the distribution for (log)prices were different.

<br>
<br>

**Do items in category of men have similar prices as that of women?**
<br>

The null and alternative hypothesis for this case will be:<br>
<br>
H0 : Mean Price of the items from category of men and women are similar<br> 
Ha : Mean Price of the items from category of men and women are different<br>
We'll check again for 5% significance level.<br>
For this case, we will run t- test of independence. Here, also we get the p_value less than 0.05, which implies that price of the items depend on whether it comes from men or women category. This also shows that preferences of women and men are different from each other in terms of shopping.<br>
<br>
<br>
Using KS test we can also check if the price distribution of the two categories is same or not. The p_value for this test was 0.03 which is also less than 0.05. This supports the observation that distribution of the prices between two categories is different.
<br>
<br>
**Is condition of item is independent of categories?**
<br>

For this, the null and alternative hypothesis will be:<br>
<br>

H0: item condition and categories (categ_1) are independent,<br>
Ha: item condition and categories (categ_1) are not independent.<br>
Running statistical test for 5% significance level.<br>
Here we will choose chi square test as statistical test, as both the features are categorical.<br>
<br>
<br>
The p_value we get is again less than 0.05, thus we will reject the null hypothesis. Thus we can statistically say that the item condition id is NOT independent of categories in the feature categ_1.
<br>
<br>

## 4. Pre-processing of data before modelling
Before we ahead with modelling, we need to make sure that data is in acceptable format for the algorithm and perform.
### Text Processing
 The features 'name' and 'item_description' have text as values. We have already converted it to lower case, removed punctations and "stopwords" present in English language using NLTK library.
Since both the features are in text format, so we can do the following: <br>

- Combine them into one, to reduce count of features.
- Perform tokenization on this contents of this feature. This will help to split each word from the sentence. 
- Next we will do stemming, which will help to keep only the words with same meaning. For e.g. moved and move, rotating and rotation etc. these words have same meaning but are different for python. Stemming will help to keep one word and remove other and restricting repetition of words. For this we will use PorterStemmer(). However, note that this will make the sentences grammatically incorrect. But for now we can let got of this concern.
- After stemming we will use "CountVectorizer()" to vectorize our values of a feature.
- Then we will use "Word2Vec" to switch these values from string to numerical values.
- Concatenate this data-frame with the original dataframe, removing the features that are no longer useful.
- Finally, change the column names and convert them all to string data type.
<br>
<br>
<br>

## 5. Modelling
We need to predict prices, thus it is a regression problem. We will use different regression models and then finally compare their performance using RMSE (Root mean square error) and R² score. <br>

RMSE metric is not available in sklearn.metrics. So, we will use its mean_square_error and then take its square root. In order to build these models we will use sklearn library. We will follow the following steps: 
- Start with hyper-tuning its parameters, 
- Find the best parameters,
- Build the model with those parameters, 
- Make the predictions for the test data, 
- And finally test its performance using the metrics RMSE and R² score.

### A. Linear Regression with L1 (Lasso) Regularization

As the name suggests, this is linear model. In linear regression, we use regularization to prevent overfitting. This regularizing term helps in optimizing loss function: squared error.
Post the hyperparametric tuning using GridSearchCV, we get the following results:
<br>

![image](https://github.com/Asin-30/EDA-hypothesis-testing-and-price-prediction-on-Mercari-dataset/assets/69243814/78cc3d71-4c02-46c9-933f-223ff4246293)
<br>

Upon building model with best parameters, we get R² score for test data of 0.3696 and is not a very good score. The RMSE that we get for this model for test data is 0.626.

### B. Linear Regression with L2 (Ridge) Regularization 
This is also linear model of linear regression but with different regularization term.  <br>

Post the hyperparametric tuning using GridSearchCV, we get the following results:
<br>

![image](https://github.com/Asin-30/EDA-hypothesis-testing-and-price-prediction-on-Mercari-dataset/assets/69243814/ce3953d7-047b-4341-bf40-dba9bb47927a)



For this model too the evaluation metrics are same i.e. R² score and RMSE value for test data are 0.3696 and 0.626 respectively.
### C. Decision Tree Regressor
This model is tree-based model used to make prediction. Mainly used for classification problems but can also be used in regression problems. 
After hyperparametric tuning, the results are shown below. 
<br>

![image](https://github.com/Asin-30/EDA-hypothesis-testing-and-price-prediction-on-Mercari-dataset/assets/69243814/10fa4550-ed70-4eac-9a16-5581fbf34b51)


With the help of these plots we can observe the best values for the hyperparameter.

For Decision Tree Regressor, the R² score and RMSE values are 0.411 and 0.605 respectively. As one can notice, with this model R² value has increased a little and RMSE has too decreased a little. Hence, we can say that the model performance has improved a little as compared to other models above.
### D. Hist Gradient Boosting Regressor
This model is an ensemble model. We used this model instead of Gradient Boosting Regressor because as per the documentation when rows are greater than 10,000 then histogram based Gradient Boosting works faster.
Post hyperparametric tuning, the visualization for GridSearchCV results is as follows:
<br>

![image](https://github.com/Asin-30/EDA-hypothesis-testing-and-price-prediction-on-Mercari-dataset/assets/69243814/8008ff7e-d149-42b9-a6b3-7983f731ad24)

For this model R² score and RMSE value for the test data are:0.411 and 0.6049 which is almost same as the results that we got from decision tree regressor.

## 6. Comparing the model performance and conclusions

With the help of a data-frame we can compare the values of R² score and RMSE value for different models at the same time. Note that the results of these metrics is for test data.
I skipped the model validation as it was taking a lot of time, because of data size. This also provides motivation to look for better approaches, available in Deep Learning. <br>

But neither of the models are very good models. All the models have 60–62% error. R² score is also less than 0.5. This shows that our model won't give very good results. But still, Histogram Gradient Boosting Regressor perform relatively well as compared to linear models. The performance of both tree-based model is same. <br>

## 7. Further Studies
We have trained our model on only 20% of the data for fast computation. In Text processing we have converted only 300 words to vector. Because of the huge dataset, computation takes a lot of time. Because of huge data and too many feature the data becomes very sparse. 
- To counter these challenges and for model building, in I want to try deep learning approach for fast and high-performance modelling.
- To try different libraries for text processing that would be faster.

<br>

## 8. Reference
- Probability Distributions in Data Science - Scaler Topics: https://www.scaler.com/topics/probability-distributions-in-data-science/
- How to Vectorize Text in DataFrames for NLP Tasks - 3 Simple Techniques | by Eric Kleppen | Towards Data Science: https://towardsdatascience.com/how-to-vectorize-text-in-dataframes-for-nlp-tasks-3-simple-techniques-82925a5600db
- Mercari Price Suggestion Challenge | Kaggle: https://www.kaggle.com/competitions/mercari-price-suggestion-challenge
- Mercari Price Suggestion Challenge | by PUSHAP GANDHI | Towards Data Science: https://towardsdatascience.com/mercari-price-prediction-challenge-3a8ea00a7d33
- 3.3. Metrics and scoring: quantifying the quality of predictions - scikit-learn 1.3.0 documentation: https://scikit-learn.org/stable/modules/model_evaluation.html
