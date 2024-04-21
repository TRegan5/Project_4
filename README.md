
# George Washington University Bootcampspot - Project 3
Created by Breakout Room 4 Members:
- Andrew Kemp
- Bryan P Johnson
- Reed Erickson
- Tom Regan

Our Project 4 presentation can be found [here](https://www.canva.com/design/DAGCzUtw8jc/kQuNKfmgiiwtoXon-PFA8A/edit?utm_content=DAGCzUtw8jc&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton).

## An overview of the project and its purpose
Our project aims to analyze mutual funds and ETFs using machine learning techniques to uncover insights into fund performance, portfolio composition, and financial indicators. We plan to explore various targets to assess, such as identifying factors influencing fund performance, understanding the impact of portfolio composition on returns, and evaluating the significance of financial indicators in predicting fund behavior. Through our analysis, we aim to provide valuable insights for investors, fund managers, and financial analysts to make informed decisions regarding fund selection and portfolio management.
### Tools and Libraries
We will utilize Python for data cleaning, analysis, and backend development. Additionally, the following Python libraries will be employed: Matplotlib, Seaborn, and Plotly for visualization. For database management, we will use Spark/PySpark.

### The Dataset
The dataset comprises 23,783 Mutual Funds and 2,310 ETFs, featuring various attributes such as total net assets, fund family, inception date, portfolio composition (cash, stocks, bonds, sectors), historical returns (yearly and quarterly), financial ratios (price/earning, Treynor and Sharpe ratios, alpha, beta), and ESG scores.

## How we handled the initial data and cleaning
- Removed columns with only 1 unique value. There were 6 object columns with 1 unique value including "Currency", "Exchange Timezones" etc
- Removed columns with too many values. There were 49 columns with more than 14,000 NaNs
- Removed columns with more than 60% NaN
- Consistancy in Data Forms, like converting dates into dateTime
- Removed metrics we can caluculate ourselves Ex: fund return per quarter, annual return 3, 5, 10 years, alpha, beta, mean, r_squared, stdev, ratios, counts, min, max
- Replace NaN values with column means
- Remove columns with irrelevant data Ex: Management bios, Investment strategy, fund_long_name, etc.


### Binning
We had 5 columns of type "object"
- Fund Category: cutoff value > 150 to retain 82% of data
- Fund Family Type: cutoff value > 150 to retain 72% of data (and remove outliers)
- ESG peer group: cutoff value > 100 to remove outliers
- Investment Type: only had 3 unique values, Blend, Growth, Value
- Size Type: only had 3 values, Large, Medium, Small

### Merging our DataFrames
Our CSV data was zipped so before reading it in, we added a function to unzip the files.
We had Mutual Fund Prices A-Z split into chunks and then concatenated leaving us a DataFrame with 75,657,739 rows and 3 columns.
Our other DataFrame, after cleaning, had 14,680 rows and 140 columns (down from 23,783 rows and 297 columns).
These DataFrames were then merged using SparkSQL on their common index fund_symbol.

### Building the Model
- Convert Categorical Data
Using get_dummies, we converted data with dtype== “object”
- Split Preprocessed Data
We split our data using the target array of year_to_date_return (X,y)
- Split Preprocessed Data Using train_test_split
We chose to use a random_state of 42
- Scaling Data
After creating a Standard Scaler instance, we fit the data using X_train. Then, we scaled the data using .transform(X_train) and (X_test)

### The Model: First Iteration
2 Hidden Layers: We used tensorflow dense layers with 80 and 30 nodes in each hidden layer, respectively, and Relu activation
Output Layer: We went with the standard dense output layer activation of Sigmoid

#### Evaluating the First Model
Loss:
Accuracy:


### The Model: Optimized
X Hidden Layers: We used tensorflow dense layers with X and X nodes in each hidden layer, respectively, and Relu activation
Output Layer: We went with the standard dense output layer activation of Sigmoid

#### Evaluating the Optimized Model
Loss:
Accuracy:

### Takeaways



## Instructions on how to use and interact with the project
Clone the repository to your local machine by running the following command in terminal or git bash
```bash
git clone https://github.com/TRegan5/Project_4.git
```
The "fsily_etfs.ipynb" file in the repo is everything you will need to interact with our work.


## References for the data source(s)
- We found our initial data set using [Kaggle Link](https://www.kaggle.com/datasets/stefanoleone992/mutual-funds-and-etfs/data?select=MutualFunds.csv) can be found in the "Resources" folder.
- Our [project proposal](https://docs.google.com/document/d/11TZsD3AG_tvAou-sjt2WAz6kq3hwUo6OUEfyyc5nwRM/edit?usp=sharing).
- All split data can be found in the "Resources" folder.
