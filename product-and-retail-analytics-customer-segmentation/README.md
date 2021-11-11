# Product-and-Retail-analytics-Customer-segmentation


This project aims at analyzing the content of an E-commerce database that lists purchases made by  ∼ 4000 customers over a period of one year (from 2010/12/01 to 2011/12/09). Based on this analysis, I develop a model that allows to anticipate the purchases that will be made by a new customer, during the following year and this, from its first purchase.

1. Data Preparation

2. Exploring the content of variables
<br>2.1 Countries
<br>2.2 Customers and products
<br>2.2.1 Cancelling orders
<br>2.2.2 StockCode
<br>2.2.3 Basket price
3. Insight on product categories
<br>3.1 Product description
<br>3.2 Defining product categories
<br>3.2.1 Data encoding
<br>3.2.2 Clusters of products
<br>3.2.3 Characterizing the content of clusters
4. Customer categories
<br>4.1 Formating data
<br>4.1.1 Grouping products
<br>4.1.2 Time spliting of the dataset
<br>4.1.3 Grouping orders
<br>4.2 Creating customer categories
<br>4.2.1 Data enconding
<br>4.2.2 Creating categories
5. Classifying customers
<br>5.1 Support Vector Machine Classifier (SVC)
<br>5.1.1 Confusion matrix
<br>5.1.2 Leraning curves
<br>5.2 Logistic regression
<br>5.3 k-Nearest Neighbors
<br>5.4 Decision Tree
<br>5.5 Random Forest
<br>5.6 AdaBoost
<br>5.7 Gradient Boosting Classifier
<br>5.8 Let's vote !
6. Testing the predictions

7. Conclusion

<h2>Data preparation</h2>
As a first step, I load all the modules that will be used in this notebook<br>
Then, I load the data. Once done, I also give some basic informations on the content of the dataframe: the type of the various variables, the number of null values and their percentage with respect to the total number of entries.

<h2>Exploring the content of variables</h2>
This dataframe contains 8 variables that correspond to:
InvoiceNo: Invoice number. Nominal, a 6-digit integral number uniquely assigned to each transaction. If this code starts with letter 'c', it indicates a cancellation.
StockCode: Product (item) code. Nominal, a 5-digit integral number uniquely assigned to each distinct product.
Description: Product (item) name. Nominal.
Quantity: The quantities of each product (item) per transaction. Numeric.
InvoiceDate: Invice Date and time. Numeric, the day and time when each transaction was generated.
UnitPrice: Unit price. Numeric, Product price per unit in sterling.
CustomerID: Customer number. Nominal, a 5-digit integral number uniquely assigned to each customer.
Country: Country name. Nominal, the name of the country where each customer resides.

<h2> Insight on product categories</h2>
In the dataframe, products are uniquely identified through the StockCode variable. A shrort description of the products is given in the Description variable. In this section, I intend to use the content of this latter variable in order to group the products into different categories.

<h2>Classification of customers</h2>
In this part, the objective will be to adjust a classifier that will classify consumers in the different client categories that were established in the previous section. The objective is to make this classification possible at the first visit. To fulfill this objective, I will test several classifiers implemented in scikit-learn. First, in order to simplify their use, I define a class that allows to interface several of the functionalities common to these different classifiers.


<h3>Prediction accuracys</h3>

Support Vector Machine
Precision: 65.93 % 

Logostic Regression
Precision: 71.34 % 

k-Nearest Neighbors
Precision: 67.58 % 

Decision Tree
Precision: 71.38 % 

Random Forest
Precision: 75.38 % 

Gradient Boosting
Precision: 75.23 % 


<h2>Testing predictions</h2>
In the previous section, a few classifiers were trained in order to categorize customers. Until that point, the whole analysis was based on the data of the first 10 months. In this section, I test the model the last two months of the dataset, that has been stored in the set_test dataframe

<h2>Conclusion</h2>
The work described in this notebook is based on a database providing details on purchases made on an E-commerce platform over a period of one year. Each entry in the dataset describes the purchase of a product, by a particular customer and at a given date. In total, approximately  ∼ 4000 clients appear in the database. Given the available information, I decided to develop a classifier that allows to anticipate the type of purchase that a customer will make, as well as the number of visits that he will make during a year, and this from its first visit to the E-commerce site.

The first stage of this work consisted in describing the different products sold by the site, which was the subject of a first classification. There, I grouped the different products into 5 main categories of goods. In a second step, I performed a classification of the customers by analyzing their consumption habits over a period of 10 months. I have classified clients into 11 major categories based on the type of products they usually buy, the number of visits they make and the amount they spent during the 10 months. Once these categories established, I finally trained several classifiers whose objective is to be able to classify consumers in one of these 11 categories and this from their first purchase. For this, the classifier is based on 5 variables which are:

mean : amount of the basket of the current purchase
categ_N with  N∈[0:4] : percentage spent in product category with index  N 
Finally, the quality of the predictions of the different classifiers was tested over the last two months of the dataset. The data were then processed in two steps: first, all the data was considered (ober the 2 months) to define the category to which each client belongs, and then, the classifier predictions were compared with this category assignment. I then found that 75% of clients are awarded the right classes. The performance of the classifier therefore seems correct given the potential shortcomings of the current model. In particular, a bias that has not been dealt with concerns the seasonality of purchases and the fact that purchasing habits will potentially depend on the time of year (for example, Christmas ). In practice, this seasonal effect may cause the categories defined over a 10-month period to be quite different from those extrapolated from the last two months. In order to correct such bias, it would be beneficial to have data that would cover a longer period of time.
