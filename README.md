# lstm_pipeline
LSTM-Based Stock Price Prediction

The developed data product is a machine learning-driven stock price tool to predict the next-day closing prices of a given stock in a given time window. The model is trained using a Long Short-Term Memory (LSTM) neural network with dense layers for time-series forecasting, where capturing sequential dependencies and market trends is critical for accurate predictions. 

The core of the application uses deep learning techniques to go beyond basic trend detection or binary classification. Rather than only identifying whether a stock or index will go up or down, the sequential model provides a continuous predicted closing price, allowing for more granular insights into potential market movements. Given multiple stocks and indices on which to test the program, the user can explore how a set of twenty features affects the behavior of different stocks.

The raw data used for the application is downloaded from Yahoo Finance. Each individual stock’s data has columns: Date, High, Low, Open, Close, and Volume for the specified time frame. The market index data is also downloaded from Yahoo Finance but includes only the closing prices for Nasdaq Futures and S&P Global. The following is an outline of how the data was processed and managed throughout the development life cycle, organized by the section headers in the Google Colab notebook:

I. Exploratory Data Analysis
  A. Create DataFrame
    1. Stock and market index data is downloaded from Yahoo Finance for six years, from 2019-03-01 to 2025-03-01
      a) All data is merged into one DataFrame on the ‘Date’ column
      b) Stocks and Indices available in the dashboard:
II. Feature Engineering
  A. Default Features and Technical Indicators
    a) Volume: moving averages, on-balance volume, volume-weighted average price
    b) Volatility: rolling standard deviation, Bollinger Bands
    c) Momentum: rate of change, Williams %R
  B. Market Index Features
    a) Rolling correlation between chosen stock and Nasdaq/S&P Global closing prices
    b) Moving averages
    c) market rate of change
  C. Global definition of full feature set: [‘Close', 'NQ_Close', 'SPGI_Close', 'MA_10',
  'MA_50', ‘MA_200', 'OBV', 'VWAP', 'Rolling_std_10', 'Rolling_std_20', ‘BB_Middle',
    'BB_Upper', 'BB_Lower', 'ROC_10', 'ROC_20', ‘WilliamsR', 'Corr_NQ', 'Corr_SPGI',
    'NQ_MA_50', 'SPGI_MA_50']
III. Data Preprocessing
  A. Prepare Data
    1. DataFrame is merged, cleaned, and prepared for preprocessing
      a) Global feature set is added to merged DataFrame
      b) Extraneous columns (High, Low, Open, Volume) are dropped

The following key tools and libraries power the machine learning implementation:
• TensorFlow & Keras: for building and training the LSTM neural network
• scikit-learn: for data preprocessing and loss metrics
• Numpy & Pandas: for data manipulation and time-series structuring
• Matplotlib & Seaborn: for model performance and feature importance visualization

The model takes advantage of a wide range of engineered features, including:
• Technical Indicators: moving averages, Bollinger Bands, on-balance volume
• Momentum & Volatility Signals: rate of change, rolling standard deviation, Williams %R
• Market Index Data: Nasdaq and S&P Global closing prices
• Correlation trends between the individual stock and macro indicators

The LSTM sequential model consists of the following layers:
Input Shape: (X_train.shape[1], X_train.shape[2])
• X_train.shape[1] —> T —> time
• X_train.shape[2] —> D —> features
LSTM(100): 100 units for longer-term pattern detection, returns predictions in 60-day
sequences
Dense(—): variable units depending on which ticker is selected
• Dense layers refine predictions when dealing with a high volume of input features
• fully connected between the LSTM layer and the final prediction
Dropout(—): variable, randomly deactivates a percentage of the neuron connections
• regularizes sequences and prevents overfitting, adds noise to the training process
• less aggressive dropout rates suit broader markets
• more aggressive dropout rates suit stocks belonging mainly to one sector
LSTM(50): 50 units to compress and filter the learned patterns
Dense(—): variable units to finalize the transformation before output
Dropout(—): variable, more regularization before output
Dense(1): outputs the predicted price

The model uses the Adam optimizer with a customized learning rate (0.0001), offering adaptive gradient descent for stable convergence.
• The loss function is Mean Squared Error (MSE), optimal for continuous numeric prediction
• Training is controlled by:
    • epochs: 75, tested from 10 to 100
    • batch_size: 32, tested from 16 to 128
    • verbose=2: outputs progress with loss and validation loss at each epoch

The training strategy was optimized to balance performance and interpretability, as the look-back window of 60 days gives the model sufficient history to learn from trends. The dropout layers and validation loss help identify and prevent overfitting, while the closing prices and input features create a well-rounded feature space for price prediction.
