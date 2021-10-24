#!/usr/bin/env python
# coding: utf-8

# # Mount dataset resources

# In[1]:


import re
import pandas as pd
import numpy as np
import warnings

import datetime
from datetime import datetime, date
import time

# Visualisation
from matplotlib import pyplot as plt

# Suppress warnings
warnings.filterwarnings("ignore")

# чтениие/запись файлов в feather-формате
import feather


# In[2]:


from google.colab import drive
drive.mount('/content/drive')


# In[3]:


get_ipython().magic(u"cd '/content/drive/MyDrive/github/eda_examples/Kaggle_Flight_Data_Analysis/'")
get_ipython().magic(u'pwd')


# # Kaggle 2015 Flight Delay Data Analysis
# 

# In[4]:


# Importing the packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
# Display settings
pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 200)


# In[5]:


# Loading the file
flights = pd.read_csv('./flights.csv')
flights.head()


# ### Part 1: Exploratory Analysis
# #### 1. How many observations are there? How many features are there?

# In[17]:


# .shape attribute gives us the count of observations.
print(f'{flights.shape[0]:5} - Total number of observations')
# Similarly we can get the count of features
print(f'{flights.shape[1]:5} - Total number of features')
# list of columns name
print(f"\nThe column names in the flights dataset are:\n {flights.columns.values}")


# #### 2. How many different airlines are there? What are their counts?

# In[23]:


# Here we are trying to get the distinct/unique values in the airline column.
print(f'{flights.AIRLINE.nunique():5} - total num of different airlines in the dataset')

# Here we are trying to get their counts, we will use group by
airline_counts = flights.groupby('AIRLINE')['AIRLINE'].count().sort_values(ascending = False)
print("\nThe count for the airlines in the dataset in the descending order are:")
airline_counts


# #### 3. How many missing values are there in the departure delays? How about arrival delays? Do they match? Why or why not? Remove these observations afterwards.

# In[24]:


# Calculating the missing values in departure delays
print(f"{flights['DEPARTURE_DELAY'].isnull().sum():5} - total number of missing values in departure delays")

# Calculating the missing values in the arrival delays
print(f"{flights['ARRIVAL_DELAY'].isnull().sum():5} - total number of missing values in arrival delays")


# **The number of missing values for departure delays and arrival delays DO NOT match. We have more missing values for arrival delays.**

# In[25]:


# Analysis: Checking if there are flights which are missing ARRIVAL_DELAY values actually departed.
flights[flights['ARRIVAL_DELAY'].isnull()][['DEPARTURE_TIME','DEPARTURE_DELAY','ARRIVAL_TIME','ARRIVAL_DELAY']]


# **From the above subset of data we can see that there are flights with departure time but are missing arrival delay values.**

# In[26]:


# Checking the status of such flights. It can be due to flight returning to base after take off or there can be a diversion.
flights[flights['ARRIVAL_DELAY'].isnull()][['DEPARTURE_TIME','DEPARTURE_DELAY','ARRIVAL_TIME','ARRIVAL_DELAY','DIVERTED','CANCELLED']]


# **From the above subset of data we can conclude that this mismatch in the missing values is due to the flight diversion.**

# In[27]:


# Dropping the missing values for DEPARTURE_DELAY AND ARRIVAL_DELAY columns
flights.dropna(subset=['DEPARTURE_DELAY','ARRIVAL_DELAY'], axis=0, inplace= True)


# In[30]:


# Checking if the missing values for DEPARTURE_DELAY have been dropped from the dataset.
print(f"{flights['DEPARTURE_DELAY'].isnull().sum():5} - total number of missing values in departure delays")

# Checking if the missing values for ARRIVAL_DELAY have been dropped from the dataset.
print(f"{flights['ARRIVAL_DELAY'].isnull().sum():5} - total number of missing values in arrival delays")


# #### 4. What is the average and median departure and arrival delay? What do you observe?

# In[33]:


# Calculating the average departure delay
print(f"{flights.DEPARTURE_DELAY.mean():5.3f} - Average departure delay")

# Calculating the average arrival delay
print(f"{flights.ARRIVAL_DELAY.mean():5.3f} - Average arrival delay")

# Calculating the median for departure delay.
print(f"{flights.DEPARTURE_DELAY.median():5.3f} - Median departure delay")

# Calculating the median for arrival delay.
print(f"{flights.ARRIVAL_DELAY.median():5.3f} - Median arrival delay")


# **Based on the values above we find that the mean is greater than median for both departure and arrival delay**

# In[34]:


# Analysis
# Checking skewness to determine the distribution
print(f'Skew DEPARTURE_DELAY: {flights.DEPARTURE_DELAY.skew():.3f}')
print(f'Skew ARRIVAL_DELAY:   {flights.ARRIVAL_DELAY.skew():.3f}')


# In[35]:


# Creating the boxplot.
plt.boxplot([flights['DEPARTURE_DELAY'], flights['ARRIVAL_DELAY']])
plt.ylabel('Delay')
plt.xticks([1, 2],['Departure delay', 'Arrival delay'])
plt.title('Boxplot for Departure and Arrival Delay')
plt.show()


# **Observations:**  
# * From the above box plot we can see that there are a lot of outliers and extreme values in the dataset.  
# * The coefficient of skewness is also significantly higher than zero.  
# * The distribution is skewed to the right and extremely high values have a significant impact on the mean.

# #### 5. Display graphically the departure delays and arrival delays for each airline. What do you notice? Explain

# In[36]:


# Creating a category plot for departure delays using seaborn library.
ax = sns.catplot(x = 'AIRLINE', y = 'DEPARTURE_DELAY', data = flights, palette = "muted", aspect=2)
ax.set_xlabels('Airlines')
ax.set_ylabels('Departure Delay')
ax.fig.suptitle('Departure delay for Airlines', fontsize=14)
plt.show()


# In[37]:


# Creating a category plot for arrival delays using seaborn library.
ax2 = sns.catplot(x = 'AIRLINE', y = 'ARRIVAL_DELAY', data = flights, palette = "muted", aspect=2)
ax2.set_xlabels('Airlines')
ax2.set_ylabels('Arrival Delay')
ax2.fig.suptitle('Arrival delay for Airlines', fontsize=14)
plt.show()


# In[38]:


# Checking correlation
print("Correlation Matrix")
flights[['DISTANCE','DEPARTURE_DELAY','ARRIVAL_DELAY']].corr()


# #### Observations:
# * We can see that the arrival and departure delay follow the same trend.
# * This trend indicate that there might be a strong correlation between the arrival and departure delay.
# * From the above correlation matrix we can see that there is no correlation between the distance and delays (0.02 & -0.02).
# * However there is a strong positive correlation between the departure and arrival delays (0.93). Hence, delayed flights arrive late.
# 

# #### 6. Now calculate the 5 number summary (min, Q1, median, Q3, max) of departure delay for each airline. Arrange it by median delay (descending order). Do the same for arrival delay.
# **Departure delay 5 number summary**

# In[39]:


# Calculating departure delay summary, concatenating median to the summary.
departure_delay_data = pd.concat([flights.groupby('AIRLINE')['DEPARTURE_DELAY'].describe(),flights.groupby('AIRLINE')['DEPARTURE_DELAY'].aggregate([np.median])],axis =1)
# Dropping the columns which are not needed.
departure_delay_summary = departure_delay_data.drop(['count','std','mean','50%'], axis=1)
# Sorting by median delay in descending order.
departure_delay_summary.sort_values(by= 'median', axis=0, ascending = False, inplace=True)
# Displaying the results
departure_delay_summary.rename(columns={'min': 'Min', '25%': 'Q1', '75%': 'Q3', 'max':'Max','median': 'Median'})


# **Arrival delay 5 number summary**

# In[40]:


# Calculating arrival delay summary, concatenating median to the summary.
arrival_delay_data = pd.concat([flights.groupby('AIRLINE')['ARRIVAL_DELAY'].describe(),flights.groupby('AIRLINE')['ARRIVAL_DELAY'].aggregate([np.median])],axis =1)
# Dropping the columns which are not needed.
arrival_delay_summary = arrival_delay_data.drop(['count','std','mean','50%'], axis=1)
# Sorting by median delay in descending order.
arrival_delay_summary.sort_values(by= 'median', axis=0, ascending = False, inplace=True)
# Displaying the results
arrival_delay_summary.rename(columns={'min': 'Min', '25%': 'Q1', '75%': 'Q3', 'max':'Max','median': 'Median'})


# #### 7. Which airport has the most averaged departure delay? Give me the top 10 airports. Why do you think the number 1 airport has that much delay?

# In[41]:


# To calculate the average departure delay we need to group by on origin airport.
# Creating a subset of data.
mean_departure_delay = flights.groupby('ORIGIN_AIRPORT')['DEPARTURE_DELAY'].aggregate([np.mean])
print("The airport with the most averaged departure delay is")
mean_departure_delay[mean_departure_delay['mean'] == mean_departure_delay['mean'].max()]


# In[42]:


# Displaying top 10 airports
mean_departure_delay.sort_values(by= 'mean', axis=0, ascending = False).head(10)


# In[43]:


# Checking the number of observations of airport == FAR in the dataset.
flights[flights['ORIGIN_AIRPORT'] == 'FAR']


# #### Observation:
# * Here, we can see that the airport FAR has only one observation in the dataset. Hence, the reason for it being the airport with the maximum average delay.

# #### 8. Do you expect the departure delay has anything to do with distance of trip? What about arrival delay and distance? Prove your claims.

# In[44]:


# Checking if there is a correlation between distance and departure delay as well as between distance and arrival delay.
# Creating the correlation matrix for the above
flights[['DISTANCE','DEPARTURE_DELAY','ARRIVAL_DELAY']].corr()


# #### Observations:
# * The above correlation matrix proves that the distance has nothing to do with the departure and arrival delays.
# * There is no correlation between the distance and the departure and arrival delays.

# #### 9. What about day of week vs departure delay?

# In[45]:


# Plotting a graph of week vs departure delay
ax3 = sns.catplot(x = 'DAY_OF_WEEK', y = 'DEPARTURE_DELAY', data = flights, kind="bar", palette = "muted", aspect=2)
ax3.set_xlabels('Day of Week')
ax3.set_ylabels('Departure Delay')
ax3.fig.suptitle('Departure delay vs Day of week for Airlines', fontsize=14)
plt.show()


# In[46]:


# Checking correlation
flights[['DAY_OF_WEEK','DEPARTURE_DELAY']].corr()


# #### Observations:
# * From the above graph we can see that the average departure delay for each day of the week is nearly same.
# * The correlation matrix also proves that there is no correlation between the departure delay and day of the week.

# #### 10. If there is a departure delay (i.e. positive values for departure delay), does distance have anything to do with arrival delay? Explain. (My experience has been that longer distance flights can make up more time.)

# In[47]:


# Creating a subset of dataframe comprising of positive departure values
positive_departure_delay_subset = flights[flights['DEPARTURE_DELAY'] > 0]
positive_departure_delay_subset.head()


# In[48]:


# Creating a scatter plot to see the relation between distance and arrival delay
plt.scatter(positive_departure_delay_subset['DISTANCE'],positive_departure_delay_subset['ARRIVAL_DELAY'])
plt.ylabel("Arrival delay")
plt.xlabel("Distance")
plt.title('Distance Vs Arrival delay', fontsize=14)
plt.show()


# In[49]:


# Checking correlation
positive_departure_delay_subset[['DISTANCE','ARRIVAL_DELAY']].corr()


# #### Observations:
# * Distance has nothing to do with arrival delay.
# * The scatter plot and the correlation matrix suggests the same. All the long distance flight may or may not be able to makeup the lost time.

# #### 11. Come up with two interesting questions that you want to answer, then explore it in using this data set. Use any numerical or graphical methods to support your answers. (preferably both).

# **Q1. From which airport does most flights originate?**

# In[50]:


# Creating a subset of values
originating_flights = flights.groupby('ORIGIN_AIRPORT').agg({'AIRLINE': 'count'})
originating_flights = originating_flights.reset_index()
# Picking only top 10 airports.
originating_flights = originating_flights.sort_values(by = 'AIRLINE', ascending = False).head(10)
originating_flights.head()


# In[51]:


sns.barplot(x=originating_flights['ORIGIN_AIRPORT'], y=originating_flights['AIRLINE'])
plt.xlabel('Origin Airport')
plt.ylabel('Flight Count')
plt.title('Top 10 airports with most originating flights')
plt.show()


# **Answer: From the analysis above, we can see that the most flights originate from ATL airport**

# **Q2. Which is the most visited city?**

# In[52]:


# Creating a subset of destination flights
destination_flights = flights.groupby('DESTINATION_AIRPORT').agg({'AIRLINE': 'count'})
destination_flights = destination_flights.reset_index()
# Picking only top 10 airports.
destination_flights = destination_flights.sort_values(by = 'AIRLINE', ascending = False).head(10)
destination_flights.head()


# In[53]:


sns.barplot(x=destination_flights['DESTINATION_AIRPORT'], y=destination_flights['AIRLINE'])
plt.xlabel('Destination Airport')
plt.ylabel('Flight Count')
plt.title('Top 10 destination airports')
plt.show()


# **Answer: From the above analysis, we can deduce that ATL is the most visited city. If we combine the results of both Q1 and Q2 we can say ATL is the busiest airport.**

# **Q3. Since ATL is the most visited destination. In which part of the year do people visit it most?**

# In[54]:


# Creating a subset of data with ATL as the only destination airport
atl_data = flights[flights['DESTINATION_AIRPORT'] == 'ATL']
atl_monthly_count = atl_data.groupby('MONTH').agg({'AIRLINE':'count'})
atl_monthly_count = atl_monthly_count.reset_index()
atl_monthly_count


# In[55]:


sns.barplot(x=atl_monthly_count['MONTH'], y=atl_monthly_count['AIRLINE'])
plt.xlabel('Months')
plt.ylabel('Flight Counts')
plt.title('Monthly count of flights to ATL airport')
plt.show()


# **Answer: There are maximum flights in the month of March and in the summer months the count of flights is more. Hence people visit ATL mostly during Spring and Summer.**

# In[56]:


flights.head()


# **Q4. Does all airlines have same taxi in and taxi out times?**

# In[57]:


ax4 = plt.subplots(figsize=(6,9))
ax4 = sns.barplot(x="TAXI_OUT", y="AIRLINE", data=flights, color="y")
ax4 = sns.barplot(x="TAXI_IN", y="AIRLINE", data=flights, color="g")
plt.ylabel('Airline')
plt.xlabel('Taxi Out (yellow), Taxi In (green)')
plt.title('Taxi Out and Taxi In times for Airlines')
plt.show()


# **Answer: Taxi out and Taxi in times for all the airlines is different. However, for all the airlines taxi in times is significantly less than taxi out.**

# ### Part 2: Regression Analysis
# #### Subpart 1

# #### 1. Your response is ARRIVAL_DELAY. First, remove all the missing data in the WEATHER_DELAY column. Once you do this, there shouldn't be anymore missing values in the data set(except for the cancellation reason feature). Check that.

# In[58]:


# Checking missing values
print(f"{flights['WEATHER_DELAY'].isnull().sum():5} - Total missing values")


# In[59]:


# Dropping missing values
flights.dropna(axis=0, subset=['WEATHER_DELAY'], inplace=True)
# Checking the result.
flights.isnull().sum()


# #### 2. Build a regression model using all the observations, and the following predictors: [LATE_AIRCRAFT_DELAY, AIRLINE_DELAY, AIR_SYSTEM_DELAY, WEATHER_DELAY, DAY_OF_WEEK, DEPARTURE_TIME, DEPARTURE_DELAY, DISTANCE, AIRLINE] a total of 9 predictors. Notice the AIRLINE variable is a categorical variable.

# In[60]:


# Creating dummy variables for airline codes.
AIRLINE_CODE = pd.get_dummies(flights['AIRLINE'], drop_first=True)
flights = pd.concat([flights, AIRLINE_CODE], axis=1)
flights.head()


# In[61]:


# Creating dummy variables for days as well.
DAY_CODE = pd.get_dummies(flights['DAY_OF_WEEK'], drop_first=True, prefix = 'DAY')
flights = pd.concat([flights, DAY_CODE], axis=1)
flights.head()


# In[62]:


# Regression Model
y = flights['ARRIVAL_DELAY']
X = flights[['LATE_AIRCRAFT_DELAY', 'AIRLINE_DELAY', 'AIR_SYSTEM_DELAY', 'WEATHER_DELAY', 'DEPARTURE_TIME', 'DEPARTURE_DELAY', 'DISTANCE', 'AS', 'B6', 'DL', 'EV', 'F9', 'HA', 'MQ', 'NK', 'OO', 'UA', 'US', 'VX', 'WN', 'DAY_2', 'DAY_3', 'DAY_4', 'DAY_5', 'DAY_6', 'DAY_7']]
X_int = sm.add_constant(X) 
linreg = sm.OLS(y,X_int).fit() 
linreg.summary()


# #### 3. Perform model diagnostics. What do you observe? Explain.

# In[63]:


# Creating residual plot
sns.residplot(linreg.fittedvalues, linreg.resid)
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs Fitted Values')
plt.show()


# In[64]:


# QQPlot
figure = sm.qqplot(linreg.resid)


# #### Observations:
# * There are outliers.
# * The model does not satisfy the linearity, constant variance and normality.

# #### 4. Provide interpretations for a few of the coeffcients, and comment on whether they make sense.

# #### Interpretations:
# * Every one minute increase in airline delay, results in 0.98 minute increase in arrival (arrival delay).
# * There is an impact of 'late aircraft delay', 'air system delay', 'weather delay' and 'departure delay' on aircraft arrivals (arrival delay).
# * There is no effect of day of the week on arrivals. This is evident from the high p-values.
# * For every one minute increase in departure delay, arrival delay increases by 0.018 minutes.

# #### Subpart 2
# If you have done the above steps correctly, you will notice a lot of things "doesn't seem right". We will try to fix a couple of these things here.

# #### 1. Removing outliers: _first is to remove outliers. Using the boxplot method, remove the outliers in the ARRIVAL_DELAY variable.

# In[65]:


# Creating the box plot
plt.boxplot(flights['ARRIVAL_DELAY'])
plt.xticks([1],['Arrival Delay'])
plt.title('Boxplot of Arrival Delay')
plt.show()


# In[66]:


# Finding IQR
flights['ARRIVAL_DELAY'].quantile()
Q1 = flights['ARRIVAL_DELAY'].quantile(0.25)
Q3 = flights['ARRIVAL_DELAY'].quantile(0.75)
IQR = Q3 - Q1
print(IQR)


# In[67]:


# Creating threshold values.
threshold_1 = Q3 + (1.5 * IQR)
threshold_2 = Q1 - (1.5 * IQR)
# Dropping values
flights.drop((flights[(flights['ARRIVAL_DELAY'] > threshold_1) | (flights['ARRIVAL_DELAY'] < threshold_2)]).index, inplace = True)


# In[68]:


# Checking values in the dataset
flights.shape[0]


# In[69]:


# Creating the box plot again
plt.boxplot(flights['ARRIVAL_DELAY'])
plt.xticks([1],['Arrival Delay'])
plt.title('Boxplot of Arrival Delay')
plt.show()


# #### 2. Refit the linear regression model, but now with log(ARRIVAL_DELAY) as your response. Also, remove the nonsignificant predictors from the previous model (with p-values larger than 0.05) and the AIRLINE variable. (Remember that when removing nonsignificant predictors one can only eliminate one variable per step.)

# In[70]:


flights['LOG_ARRIVAL_DELAY'] = np.log(flights['ARRIVAL_DELAY'] +1)
flights.head()


# In[71]:


y = flights['LOG_ARRIVAL_DELAY']
X = flights[['LATE_AIRCRAFT_DELAY', 'AIRLINE_DELAY', 'AIR_SYSTEM_DELAY', 'WEATHER_DELAY', 'DEPARTURE_DELAY']]
X_int = sm.add_constant(X) 
linreg = sm.OLS(y,X_int).fit() 
linreg.summary()


# #### 3. Perform model diagnostics. Did anything improve?

# In[72]:


sns.residplot(linreg.fittedvalues, linreg.resid)
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs Fitted Values')
plt.show()


# In[73]:


figure = sm.qqplot(linreg.resid)


# #### Observations:
# * The model needs improvement.
# * The model does not satisfy the constraints of linearity, constant variance and normality.

# #### 4. Provide interpretations to a few of the coeffcients. Do you think they make sense?

# #### Interpretations:
# * Weather delay has an impact on arrival delays. For every one minute increase in weather delay there is an increase of 0.0190 minutes in arrival delay.
# * For every one minute increase in air system delay, there is an increase of 0.0198 minutes in arrival delay.

# #### 5. Obviously there's still a lot that needs to be done. Provide a few suggestions on how we can further improve the model fit (you don't need to implement them).

# #### Suggestions:
# * We can add interaction among the independent variables in the model.
# * Using Tukey;s ladder transformation, we may increase or decrease the power of independent variables and use them in the model.
