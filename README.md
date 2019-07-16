# Generative Adversarial Networks for Stock Market prediction

### Libraries used:
- TensorFlow 1.12.0
- Keras 2.2.4
- numpy 1.15.4
- pandas 0.24.2
- matplotlib 3.0.2
- seaborn 0.9.0

Please, refer to [requirements.txt](requirements.txt) for other libraries

## Project Overview
With the boom of AI the financial industry is experiencing interesting changes and improvements. Many companies are starting to integrate [Machine Learning in their businesses](https://towardsdatascience.com/the-growing-impact-of-ai-in-financial-services-six-examples-da386c0301b2). Fraud prevention, risk management, portfolio management, investment prediction, process automation, are some of the applications of ML in finance. The Stock Market and Algorithmic Trading are other areas benefiting from it. Traders are embracing some of the new techniques and algorithms to improve their performance.

Time series forecasting has become a [hot topic](https://eu.udacity.com/course/time-series-forecasting--ud980) with the advances in Machine Learning. [Time series analysis and predictions](http://www.statsoft.com/Textbook/Time-Series-Analysis) is one of the oldest and most applied data science techniques in finance, but [new ones](https://www.xenonstack.com/blog/time-series-forecasting-machine-deep-learning/) are emerging, more [sophisticated and powerful](https://medium.com/neuronio/predicting-stock-prices-with-lstm-349f5a0974d4), providing with more accuracy, using the power of Deep Learning... But before we start rubbing our hands in glee and greediness thinking we can all use these algorithms to easily get rich, I need to warn you: Stock market prices are highly unpredictable and volatile. There are no consistent patterns in the data that would allow you to model stock prices over time. Princeton University economist Burton Malkiel wrote in his 1973 book, “A Random Walk Down Wall Street”, that “[A blindfolded monkey](https://www.forbes.com/sites/rickferri/2012/12/20/any-monkey-can-beat-the-market) throwing darts at a newspaper's financial pages could select a portfolio that would do just as well as one carefully selected by experts.”

## Problem Statement
There is a lot of literature on internet about Stock Market prices forecasting using different Machine Learning algorithms, most recently using complex LSTM Networks, or combination of LSTM-CNN. They all show incredible performance, predicting almost the same behaviour as the price. But these are mostly optical illusions due to the fact that most of them are simply predicting one day ahead. Simple methods such as Simple Moving Average or a bit more sophisticated Exponential Moving Average could also achieve similar or even better performance, and this is because prices normally don’t change drastically overnight. Predicting more than one day is a whole different story. I intend to predict one week of prices, that is, 5 trading days.

As mentioned above, Stock Market prediction is a difficult task. There are [many variables](https://www.investopedia.com/articles/basics/04/100804.asp) that affect how the share prices change over time. Combination of [different factors](https://www.getsmarteraboutmoney.ca/invest/investment-products/stocks/factors-that-can-affect-stock-prices/) make them extremely volatile and therefore quite unpredictable, or at least hard to predict with some level of accuracy.

According to financial theory, the stock market prices evolve following the so called [random walk](https://en.wikipedia.org/wiki/Random_walk), but there is [evidence](http://www.turingfinance.com/stock-market-prices-do-not-follow-random-walks/) that dismisses this theory, and they could be somehow modeled based on historical information. 

Stock Market analysis is divided into two parts: [Fundamental Analysis](https://www.investopedia.com/terms/f/fundamentalanalysis.asp) and [Technical Analysis](https://www.investopedia.com/terms/t/technicalanalysis.asp). Fundamental Analysis involves analyzing the company’s future profitability on the basis of its current business environment and financial performance. Technical Analysis, on the other hand, includes reading the charts and using statistical figures to identify the trends in the stock market. In this project I’m gonna focus on the technical analysis part.

There are many solutions out there for time series forecasting such as [ARIMA models](https://towardsdatascience.com/time-series-forecasting-arima-models-7f221e9eee06), [Support Vector Machines](https://thesai.org/Downloads/IJARAI/Volume4No7/Paper_10-A_Comparison_between_Regression_Artificial.pdf), or state-of-the-art [LSTM models](https://www.kaggle.com/amirrezaeian/time-series-data-analysis-using-lstm-tutorial), [CNN model](https://machinelearningmastery.com/how-to-develop-convolutional-neural-network-models-for-time-series-forecasting/), or a [combination of both](https://towardsdatascience.com/get-started-with-using-cnn-lstm-for-forecasting-6f0f4dde5826). Unfortunately there is no much written – only found a few articles but no examples – about using [Generative Adversarial Networks](https://arxiv.org/pdf/1406.2661.pdf) for such problems. GANs are mostly used to generate very realistic images, regardless of the type of image. In this project we’re gonna explore these amazing networks and see how we can use them to predict future prices.
