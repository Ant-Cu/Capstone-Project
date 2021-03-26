# Capstone-Project
This project was done for the purpose of completing my journey at General Assembly in Data Science. I chose to focus my project on CryptoCurrency trading with BTC and ETH prices. The goal of that study is to identify supports and resistances within a 15min, 60min and daily price timeframe then use the pivot points as indicators for predicting return direction.

The different phases are as below:

![image](https://user-images.githubusercontent.com/81016049/112630524-9afee280-8e2d-11eb-9bb9-528ada8d42a5.png)

The CoinBase Pro API is used for grabbing the historical data for BTC and ETH. CoinBase is the oldest exchange still around with a decent liquidity so that the data are smooth for the past 5 years. You can find the code to download the data in that folder.

Once the data is sorted, all the following technical indicators need to be created:

* Moving average
* Bollinger Bands
* Relative Strength Indicator
* Average True Range

Technical analysis is a tool to study price and volume action with statistical tools and identify trends and patterns. 

#### Moving averages:

Simple Moving Average and Exponentially Moving Average are used in the study.

![image](https://user-images.githubusercontent.com/81016049/112630567-aa7e2b80-8e2d-11eb-8714-7ffe9ac18327.png)
 
#### Bollinger bands:

Bollinger Bands are simply two bands produced by adding n time the rolling standard deviation of an asset to a specified Moving Average for the upper band and subtracting for the lower band. In general, it will be 20period Moving Average and 2 times the standard deviation.
* Upper Band: SMA(20) + 2 x Std(20)
* Lower Band: SMA(20) – 2 x Std(20)

_Example below for ETHUSD Hourly chart with 96-period Moving Average and 2.5x the standard deviation. _ 
![image](https://user-images.githubusercontent.com/81016049/112630591-b23dd000-8e2d-11eb-86d7-3f6acf8d931c.png)

#### Relative Strength Indicator:

It indicates if an asset is overbought or oversold.
* Formula =100 -100/(1+  (EWM(P_rt,14))/|EWM(N_rt,14)| )
* Where EWM(P_rt,14) is the exponential moving average of the positive return with 14-period and EWM(P_rt,14) is the exponential moving average 	of the negative return with 14-period.

#### Average True Range:
It is an indicator of volatility. 
* True Range = max(High-Close(t-1), Low-Close(t-1),High-Low) 
* Then take the moving average of the True Range on a 14-period.
Additionally I created few columns to specified where the Close Price is located versus the Moving Averages and Bollinger Bands by simply subtracting them.  As Prices are usually mean reverting and do not move in a straight line.
I also generated another indicator which is simply the subtraction of the Moving Average with longest period and the Moving Average with shortest period. The idea is that when:
* MA(T) < MA(t) where T > t, the trend is upward

![image](https://user-images.githubusercontent.com/81016049/112630669-d4375280-8e2d-11eb-8fa1-74d23ec6d1bd.png)

The following charts will show the relationship between the some features and the price return.

![image](https://user-images.githubusercontent.com/81016049/112630706-e0231480-8e2d-11eb-9da3-bfdc7b124b57.png)

![image](https://user-images.githubusercontent.com/81016049/112630711-e31e0500-8e2d-11eb-8ffe-ab780cfebce7.png)

First a K-means Clustering unsupervised model will be used to check if any patterns can be extracted from the features for each timeframe and especially if we can capture those pivot points called Resistance and Support.
My features used are as follow:
* Difference between EMA(14) on Close Price and EMA(7) on Close Price
* Relative Strength Indicator
* Close Price and MA ratio
* Distance from Upper Bollinger band
* Distance from Lower Bollinger band
* Average True Range
Difference between long term moving averages. (for ex daily is 50day vs 100day)

All the features are scaled using MinMaxScaler as not all the features are normaly distributed. The range is [0,1].
x = [(value - min)/(Max- Min)]

Below is the silhouette analysis and the Elbow plot resulting from the model using BTC prices. 

![image](https://user-images.githubusercontent.com/81016049/112630747-f16c2100-8e2d-11eb-96ac-8db7fc5c3d01.png)
 
The 2 labels are plot using Plotly. First on daily prices, then 60min and 15min. 
![image](https://user-images.githubusercontent.com/81016049/112630769-f92bc580-8e2d-11eb-922d-4a698b54dc90.png) 
*Figure 1 BTC Daily Prices*

![image](https://user-images.githubusercontent.com/81016049/112630783-fdf07980-8e2d-11eb-9025-4c75947de962.png)
*Figure 2 BTC 60min Prices*

![image](https://user-images.githubusercontent.com/81016049/112630793-034dc400-8e2e-11eb-8f68-6733f3515317.png)
*Figure 3 BTC 15min Prices*

Once the labels are retrieved and added as a new feature to the dataframe, a prediction study was performed. The Target Variable is set as follow:
* Created a column with the Return for the next n periods (t,t+4) shift this return to t-1. Create another column with indicating if the return is positive or negative (1 or -1 respectively, 0% return is classified as negative)
- Daily data the target variable is a 4-day return at t0
- Hourly data target variable is a 4-hour return at t0
- 15min data the target variable is a 1hour return at t0

Several models were utilised:
* RandomForest Classifier
* Bagging Classifier with DecisionTree Classifier
* Bagging Classifier with Logistic Regression
* Support Vector Machine

All the models were run with GridSearchCV to get the best parameters. The RandomForest Classifier was the best model and the results for the test set were as follow:

![image](https://user-images.githubusercontent.com/81016049/112630816-0c3e9580-8e2e-11eb-9b8e-bf8833c21b7e.png) 
*Figure 4 Results for Daily Data*

![image](https://user-images.githubusercontent.com/81016049/112630831-11034980-8e2e-11eb-88a9-9e8b4b2acc07.png)
*Figure 5 Results for 60min Data*

![image](https://user-images.githubusercontent.com/81016049/112630848-15c7fd80-8e2e-11eb-904a-66fe38b14c82.png)
*Figure 6 Results for 15min Data*

![image](https://user-images.githubusercontent.com/81016049/112630867-1b254800-8e2e-11eb-9cdc-773fb89e1e2e.png)
*Figure 7 Features’ importance for Daily Data*

![image](https://user-images.githubusercontent.com/81016049/112630883-21b3bf80-8e2e-11eb-8465-9e1fb3c1a777.png)
*Figure 8 Features' importance 60min Data*

The feature importance for the 15min Data were very similar to the 60min Data.
 
As all the R scores on the train set were equal to 1, only the confusion matrix for each data set on the test set were published below.

![image](https://user-images.githubusercontent.com/81016049/112631081-59bb0280-8e2e-11eb-85b7-da84f9dd20fe.png)

### Conclusion:
The clustering model extracted to an extent the resistances and supports but also the different up and down trends from the price fluctuation. The labels did not have any impact on the prediction whether the return will be positive or negative. The train data set R-score could indicate some overfitting. The models didn’t have much edge on predicting the return direction.   



