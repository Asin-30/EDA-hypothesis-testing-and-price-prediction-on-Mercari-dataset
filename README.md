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
- GitHub Repository and LinkedIn
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

# sellers rating to their products

# percentage of the products with their rating provided by the sellers
print(df['item_condition_id'].value_counts(normalize = True)*100)

((df['item_condition_id'].value_counts(normalize = True))*100).plot(kind= 'bar')
plt.title('count plot of items as per their condition id')
plt.xlabel('item condition id')
plt.ylabel('% of the ite
And as per above plot, items with condition 1 are maximum (43.20%) in number and with condition id 5 are least (0.16%). This make sense, as the product on any platform are more likely to sell if they are in their best condition.
c. shipping
There are two categories in shipping feature: 0 (buyer is paying for the delivery) and 1 (seller is paying for the delivery). To earn more profit, it is very understandable that sellers won't like to pay for the shipping. And this reflects in the plot (count). Shipping for 55.26% of items is paid by buyers not sellers (shipping: 0). Yet, there are a lot of sellers who believe in expanding their customer reach by making items cheaper for them, which can be possible when buyer don't have to pay for the delivery. And that is why, there are 44.73% of items for which shipping was paid by sellers.

d. brand_name
There are 4775 unique brand name present in dataset. If items of a certain brand are coming frequently than it means that the brand is popular. We can say that popularity of brand is proportional to its frequency. Using this analogy we can say that the top 10 most popular brands present in dataset are as follows:
There are 6.366% of the items that belong to brand PINK which comes under the umbrella of Victoria's Secret, followed by Nike (6.35%) and Vicotria's Secret (5.65%). Two out of three most popular brands are for women. This generates the possibility that women are the most frequent shoppers on the platform. 
e. category_name
Upon closely observing the values of this feature, you will find that there are three categories involved within separated by '/'.
First is the main category, then there are two tiers of categories ahead as well. This representation is analogus to the arrangements made in Mall. For example, Men/Tops/T-shirts means go to Men section, then tops section and then finally T-shirts section. 
We will split the data with separator as '/' and call them as 'categ_1', 'categ_2', 'categ_3'.
Now for the first level category: 'categ_1'

As we anticipated earlier, Women are primary buyers on the platforms in terms of frequency. The products for the next two main categories Beauty and kids are also shopped by majorly women.
What are women purchasing the most on the platform?
The plot below shows that women are more into fitness and casual wear that is why they are buying Athletic and Apparel. 
Around 18% of the sales in women category comes from 'Athletic Apparel'. This shows the interest of women into fitness. Approx. 16% of the sales comes from Tops & Blouses category, which shows the interest of women may be in casual and formal wears. Around 12% of the sales in women category comes from Shoes. This category involves all the foot wearables by women e.g. casual shoes, flats, sport shoes, flip flops etc.
What is purchased most by women in sub-category of Athletic & Apparel?
As per the plot below, one can observe that Pants, Tights, Leggings are being sold the most followed by shorts and Shirts & Tops. While Snowsuits & bibs and Vests are least popular in this category.
What sort of products women are purchasing in subcategory of Beauty?
Women are purchasing Makeup related products the most in category of Beauty. Make up contributes around 60% of sales by women]. Next comes, the Skin Care (15% approx.) and Fragrance (12% approx.) by women in this category.
What are the most popular items in sub-category, categ_2?
The plot below make sense, because more frequent buyers are women, and they are very frequent purchasing, Athletic Apparel, Makeup and Tops & Blouses. But apart from women related items (Jewlery, Women's Handbags, Skin Care) there are also products from Electronics, which involves Cell Phones & Accessories, and Video Games & Consoles.
And the least popular items in this category are bookzines and quilts.
What are the most popular items in sub-category, categ_3?
There are 863 unique categories present in this feature. We will take the 15 most popular categories into consideration.
As per this plot, t-shirts, Pants, Tights, Leggings are the items that are bought most frequently. Most of the popular products plotted above are primarily for Women as they are the main consumers of such products e.g. Face, Lips, Eyes, Bras, Blouse, Tank-Cami. As for the electronics section the most popular items are Cases, Covers & Skins, Games. There is 'missing' category also, which include all the products that aren't very frequently bought.
There are 863 values present in this category, so for visualization we can use WordCloud.
Note: If there are more than 1 words present then we took only the last word. e.g. Legging mean the category of Pants, Tights Leggings. It is done so that it is easy to read in word cloud plots.
Next two features are name and item_description. Before we can gain any insights from them we need to make sure that data is consistent.
Cleaning text data
Next there are text features remaining: item_description and name. We need to first clean it first, so that python doesn't take similar words as different. For e.g. in name there is value called Bundle and bundle, for python these are different but for the sake of our analysis it should be same. So, we will clean it and bring the text data in similar format. 
We will convert the text data to lower case first then remove the 'stopwords' from English. If there are null values, then replace them with 'missing'.
f. name
Since this is text feature thus number of unique values could be huge. In order to reduce that number, we will turn every word to lower case and remove all the punctuations from it. In order to decrease its length, we will also remove the "stopwords" in English like is, the, am, of etc. using NLTK library. 
For visual representation we will be using WordCloud. The most popular items are: bundle, legging, dress, set, bag etc.
g. item_description
There is lot of text data corresponding to each item. It will take a lot memory and space to run analysis on this feature. But later on we'll use a library (Word2Vec) to vectorize it in numerical format.
Now, we won't need name, item_description, brand_name anymore in our analysis; we have cleaned them and stored their data in new features. So, we will drop these columns from our dataset to save up space.
2.2 Bivariate Analysis

a.  item condition id vs (log) price
Below are the box plot and probability distribution of (log)prices of the items and their condition id. The standard deviation for the items with condition id 5 is maximum, followed by items with condition id 1. This make sense because where the item condition is not too good then buyers won't like to pay the standard price for the item and seller would keep on changing the price till it gets sold. As for the items with condition id 1, buyers would want to buy items in its best possible condition. And if the product is too expensive, then they will ensure that condition of the product is 1. This directs a relatively big spectrum of possible prices of items in both the mentioned condition ids and hence standard deviation is slightly higher as compared to other price of items with other condition ids.
In the box plot and PDF plot, one can notice that the price distribution of price of items with condition id 5 is relatively different from distributions of price with other condition id. Since, the item condition is poor hence the price is lower than the rest.
b. Shipping vs log_price
Next, we will do plotting for shipping and (log) price.
One can notice from the plot below that mean price of items where the buyers had to pay for shipping is higher. And this is intuitive, if the buyer is paying for the product as well as for shipping then the overall price of the item will increase.
The PDF plot for the price of items where the shipping paid by buyer is little bit shifted towards right. This shows that min price of the item is also higher for buyers. The conclusions drawn above are also reflected in the box plot below:
The mean, min and max of price of item where shipping is paid by buyer is higher.
c. categories vs (log)price 
We take each sub category one at a time for the analysis. First, we will start with the main category i.e. categ_1.
c.1 categ_1 and log_price
Categ_1 has the main categories of the platform. We will draw PDFs and bar charts to check the distribution of (log) prices and know about mean and standard deviation.
We can see that distribution for the price of handmade items is quite different from other categories. Yet, electronics has the highest standard deviation and men category has the highest mean of the prices of items purchased on the platform.
Also, we saw earlier that women are most frequent customers as the items getting sold more often are women oriented. But here, we can see that the women category come second to men in terms of mean sales of the items belonging to their respective categories, followed by home and then electronics.
In terms of revenue generation for the company as expected, women category tops the list. The next two categories also primarily belong to women because beauty products are mainly for women and kids can't shop for themselves, hence it is more likely that their mothers shop for them. Next comes the electronics category followed by men.

c.2. categ_2 and log_price
There are 113 unique categories within categ_2. We will check what are top 10 categories in terms of frequency.
we have already seen that 9.104% sales are made from Athletics & Apparel, then makeup followed by tops & blouses. If we again assume that the categories having maximum prices will have expensive products, then we can make the following plot:
One can notice that max price for jewelry is comparable to other categories as well. This suggests that there are products of luxury brands because their prices matches with that of jewelry.
The standard deviation is least for makeup. And cell phone accessories has maximum standard deviation. Mean is highest for computers & tablets so they are the most expensive products among these categories followed by bags & purses and then women's handbags. 
Least selling categories are bathing skin care, blazers sport coats, candles etc and have least contribution in generating revenue.
c.3  categ_3 and (log)price
Considering the same assumption as before that max price of category is because of the expensive products present in them.

Here in the plot above one can observe that Satchel are the most expensive products which make sense, because earlier we say that items from brands PINK, Victoria's Secret and others are frequently bought by women. And these comes under the category of luxury brands and hence are expensive. Next comes the ipad, followed by shoulder bag and then messenger crossbody.
d. brand vs log(price)
The brands that have most expensive items are Mary Kay and Chanel. NaN may include the brand that are not very famous but expensive.
One can observe that most of the brand are for clothing. Here also, we see that since women are the primary customers, the brands also such that it will attract more women.
The distribution of the price of the items from these brands is also very different from each other. It is possible because of different prices for different items and what is popular among customers.
Among all these brands Mary Kay has the least standard deviation and most popular brand among buyers. Earlier also, we saw the makeup brands has the least standard deviation, Mary Kay is also such brand, hence it's standard deviation is least. Also makeup is the second most popular sub-category, this explaining why it has high density.
Other brands offers various products clothing, bags, watches etc this explains the high standard deviations. Hence, they have low density and high standard deviation, as well as wide price range. We can also see that from the box plot below:
Among these brands, only Apple is for electronics category.
What are the popular items (names) with best and worst condition ids?
Since there are too many variables in 'name' features so we will use WordCloud for visualization of popular items.
The most popular item that are popular among customers that they want in best condition are : bundle, legging, set, case, dresses , bags etc.
The products that customer want to buy irrespective of their bad condition are: part (may be spare part of something), purse, bag, doll, wallet etc.
3. Hypothesis Testing
Now to support the observation drawn above for population data, we will run some statistical test and perform hypothesis testing. To run these test we will use scipy library's stats module. Let's start.
Is price  of the item similar for both shipping categories (0: paid by buyer, 1: paid by seller)?
We need to declare the null and alternative hypothesis first:
H0 : Mean of the prices of products for both shipping type is similar. 
Ha : Mean of the prices of products for both shipping type is different
We will run t-test of independence for 5% significance level to check if both have similar mean. We chose this test because we don't about population mean and standard deviation and dataset is large so it will be as good as z test.
a = df1.loc[df1['shipping'] == 1, 'log_price']
b = df1.loc[df1['shipping'] == 0, 'log_price']
stat_value,p_value = ttest_ind(a,b)

if p_value < 0.05:
    print('reject the null hypothesis (H0) with p-value:',p_value," ,this implies that shipping type  affects the price of product" )
else:
    print('Fail to  reject the null hypothesis (H0) with p-value:',p_value," ,this implies that shipping type doesn't affect the price of product" ) 
The p-values comes out to be less than 0.05, thus we are able to reject our null hypothesis. Thus, statistically we can say that prices are affected by shipping type. Prices are not independent of shipping type.
Does item condition affect the price of items on the platform?
The null hypothesis and alternative hypothesis are as follows:
H0 : Prices of products for different condition id have similar mean. 
Ha : Prices of products for different condition id have different mean.
we'll run the hypothesis for 5% significance level.
For this we will choose ANOVA as our statistical test as we have to check for similarity among more than two labels.
a = df1.loc[df1['item_condition_id'] == 1, 'log_price']
b = df1.loc[df1['item_condition_id'] == 2, 'log_price']
c = df1.loc[df1['item_condition_id'] == 3, 'log_price']
d = df1.loc[df1['item_condition_id'] == 4, 'log_price']
e = df1.loc[df1['item_condition_id'] == 5, 'log_price']

stat_value,p_value = f_oneway(a,b,c,d,e)

if p_value < 0.05:
    print('reject the null hypothesis (H0) with p-value:',p_value," ,this implies that item condition have impact on the  price of product" )
else:
    print('Fail to  reject the null hypothesis (H0) with p-value:',p_value," ,this implies that item condition don't have any impact on price of product" ) 
Here as well we get p_values < 0.05, this implies that item condition is dependent on price of product.
Using KS test, we will also check if the distribution of prices of items is similar for condition id 1 and 2 or not. Upon running the test,we found that even the condition ids that are close to each other e.g. 1 and 2, the distribution for (log)prices were different.
Do items in category of men have similar prices as that of women?
The null and alternative hypothesis for this case will be:
H0 : Mean Price of the items from category of men and women are similar 
Ha : Mean Price of the items from category of men and women are different
We'll check again for 5% significance level.
For this case, we will run t- test of independence. Here, also we get the p_value less than 0.05, which implies that price of the items depend on whether it comes from men or women category. This also shows that preferences of women and men are different from each other in terms of shopping.
Using KS test we can also check if the price distribution of the two categories is same or not. The p_value for this test was 0.03 which is also less than 0.05. This supports the observation that distribution of the prices between two categories is different.
Is condition of item is independent of categories?
For this, the null and alternative hypothesis will be:
H0: item condition and categories (categ_1) are independent,
Ha: item condition and categories (categ_1) are not independent.
Running statistical test for 5% significance level.
Here we will choose chi square test as statistical test, as both the features are categorical.
a =  pd.crosstab(index = df1['item_condition_id'], columns = df1['categ_1'])

stat_value,p_value,_,_ = chi2_contingency(a)

if p_value < 0.05:
    print('reject the null hypothesis (H0) with p-value:',p_value," ,item conditions and categories are not independent" )
else:
    print('Fail to  reject the null hypothesis (H0) with p-value:',p_value," ,item conditions and categories are independent" ) 
The p_value we get is again less than 0.05, thus we will reject the null hypothesis. Thus we can statistically say that the item condition id is NOT independent of categories in the feature categ_1.
4. Pre-processing of data before modelling
Text Processing
Before we ahead with modelling, we need to make sure that data is in acceptable format for the algorithm and perform. The features 'name' and 'item_description' have text as values. We have already converted it to lower case, removed punctations and "stopwords" present in English language using NLTK library.
Since both the features are in text format, so we can do the following:
Combine them into one, to reduce count of features. Then
Perform tokenization on this contents of this feature. This will help to split each word from the sentence. 
Next we will do stemming, which will help to keep only the words with same meaning. For e.g. moved and move, rotating and rotation etc. these words have same meaning but are different for python. Stemming will help to keep one word and remove other and restricting repetition of words. For this we will use PorterStemmer(). However, note that this will make the sentences grammatically incorrect. But for now we can let got of this concern.
After stemming we will use "CountVectorizer()" to vectorize our values of a feature.
Then we will use "Word2Vec" to switch these values from string to numerical values.

6. Concatenate this data-frame with the original dataframe, removing the features that are no longer useful.
7. Finally, change the column names and convert them all to string data type.
def sent_to_vec(text_list):
    vec = np.zeros(shape = (300,), dtype = 'float32')
    l = len(text_list)
    for i in text_list:
        try:
            vec += item_and_name_w2v.wv[i]
        except:
            continue
    if l != 0:
        avg_vec = vec/l
        
    return avg_vec

# integrating the two text columns together.
X['name_and_item_desc'] = X['cleaned_name'] +' '+ X['clean_item_description']
# dropping the text columns : 'cleaned names' and 'clean_item_description'
X.drop(['cleaned_name','clean_item_description'], axis = 1, inplace = True)
# to remove extra missing from the values caused by concatenation
X['name_and_item_desc'].loc[X['name_and_item_desc'] == 'missing missing'] = 'missing'
# splitting into words: tokenization 
X['name_and_item_desc'] = X['name_and_item_desc'].apply(lambda x: x.split())
#stemming
ps = PorterStemmer()
X['name_and_item_desc'] = X['name_and_item_desc'].apply(lambda x:  [ps.stem(i) for i in x] )
# creating list of unqiue words
total_words = []
for i in X['name_and_item_desc']:
    total_words += i

#vectorization
vectorizer  = CountVectorizer()
x1 = vectorizer.fit_transform(total_words)
item_and_name_w2v = Word2Vec(total_words, min_count = 5,
                              vector_size = 300, window = 20)

# converting the vector
X['name_and_item_desc2'] = (X['name_and_item_desc']).apply(sent_to_vec)

# generating more features for words
x_name_and_item_desc = (X["name_and_item_desc2"]).apply(pd.Series)
# dropping columns for feature 'name_and_item_desc'
X.drop('name_and_item_desc', axis = 1, inplace = True)

# finally concatenating
X = pd.concat([X[['clean_brand_name', 'item_condition_id', 'shipping', 'categ_1',
                   'categ_2', 'categ_3', 'name_and_item_desc2']],x_name_and_item_desc], axis = 1)
# dropping 'name_and_desc2' feature
X.drop('name_and_item_desc2', inplace = True, axis = 1)
# changing the dytpe of columns [0,1,2.....299] from int to object
col_list  = ['clean_brand_name', 'item_condition_id', 'shipping', 'categ_1',
                   'categ_2', 'categ_3']
for i in X.columns[6:]:
    col_list.append('word_' + str(i))
5. Modelling
We need to predict prices, thus it is a regression problem. We will use different regression models and then finally compare their performance using RMSE (Root mean square error) and R² score. 
RMSE metric is not available in sklearn.metrics. So, we will use its mean_square_error and then take its square root. In order to build these models we will use sklearn library. We will follow the following steps: 
a. Start with hyper-tuning its parameters, 
b. Find the best parameters, 
c. Build the model with those parameters, 
d. Make the predictions for the test data, 
e. And finally test its performance using the metrics RMSE and R² score.
Let's get started.
Linear Regression with L1 (Lasso) Regularization

As the name suggests, this is linear model. In linear regression, we use regularization to prevent overfitting. This regularizing term helps in optimizing loss function: squared error.
Post the hyperparametric tuning using GridSearchCV, we get the following results:
Upon building model with best parameters, we get R² score for test data of 0.3696 and is not a very good score. The RMSE that we get for this model for test data is 0.626.
2. Linear Regression with L2 (Ridge) Regularization 
This is also linear model of linear regression but with different regularization term. 
Post the hyperparametric tuning using GridSearchCV, we get the following results:
For this model too the evaluation metrics are same i.e. R² score and RMSE value for test data are 0.3696 and 0.626 respectively.
3. Decision Tree Regressor
This model is tree-based model used to make prediction. Mainly used for classification problems but can also be used in regression problems. 
After hyperparametric tuning, the results are show below. With the help of these plots we can observe the best values for the hyperparameter.
For Decision Tree Regressor, the R² score and RMSE values are 0.411 and 0.605 respectively. As one can notice, with this model R² value has increased a little and RMSE has too decreased a little. Hence, we can say that the model performance has improved a little as compared to other models above.
4. Hist Gradient Boosting Regressor
This model is an ensemble model. We used this model instead of Gradient Boosting Regressor because as per the documentation when rows are greater than 10,000 then histogram based Gradient Boosting works faster.
Post hyperparametric tuning, the visualization for GridSearchCV results is as follows:
For this model R² score and RMSE value for the test data are:0.411 and 0.6049 which is almost same as the results that we got from decision tree regressor.
6. Comparing the model performance and conclusions
With the help of a data-frame we can compare the values of R² score and RMSE value for different models at the same time. Note that the results of these metrics is for test data.
I skipped the model validation as it was taking a lot of time, because of data size. This also provides motivation to look for better approaches, available in Deep Learning.
But neither of the models are very good models. All the models have 60–62% error. R² score is also less than 0.5. This shows that our model won't give very good results. But still, Histogram Gradient Boosting Regressor perform relatively well as compared to linear models. The performance of both tree-based model is same.
7. Further Studies
We have trained our model on only 20% of the data for fast computation. In Text processing we have converted only 300 words to vector. Because of the huge dataset, computation takes a lot of time. Because of huge data and too many feature the data becomes very sparse. 
a. To counter these challenges and for model building, in I want to try deep learning approach for fast and high-performance modelling.
b. To try different libraries for text processing that would be faster.
8. Github Repository and LinkedIn
Here is the Github Repository to refer to the full code and to connect with me on LinkedIn.
9. Reference
Probability Distributions in Data Science - Scaler Topics
How to Vectorize Text in DataFrames for NLP Tasks - 3 Simple Techniques | by Eric Kleppen | Towards Data Science
Mercari Price Suggestion Challenge | Kaggle
Mercari Price Suggestion Challenge | by PUSHAP GANDHI | Towards Data Science
3.3. Metrics and scoring: quantifying the quality of predictions - scikit-learn 1.3.0 documentation
