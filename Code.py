    # Importing Libraries
    
    # Utility Libraries
    import numpy as np
    import pandas as pd
    
    # Visualisation Libraries
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    # Data Processing Libraries
    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import chi2
    from sklearn import model_selection
    from sklearn.metrics import mean_squared_error, r2_score
    
    # Algorithm Libraries
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import RandomForestRegressor
    
    # Math Library
    import math
    
    # Importing Dataset
    df = pd.read_csv("C:/Users/suyas/Downloads/archive/HousingData.csv")
    
    # Printing Dataset
    df.head()
    # Data Analysis
    
    # Checking for columns and their respective datatypes
    df.info()

    # Getting the number of rows and columns
    df.shape

    # Calculating the mean, minimum, deviation, maximum and other factors
    df.describe()
    fig=plt.figure()
    ax=fig.add_subplot(1, 1, 1)
    ax.hist(df['RM'], bins = 35)
    plt.title("Average number of rooms Distribution ")
    plt.xlabel("RM")
    plt.ylabel("frequency")
    plt.show()
    fig=plt.figure()
    ax=fig.add_subplot(1, 1, 1)
    ax.hist(df['LSTAT'], bins = 35)
    plt.title("Homeowners distribution with low class")
    plt.xlabel("LSTAT")
    plt.ylabel("frequency")
    plt.show()
    fig=plt.figure()
    ax=fig.add_subplot(1, 1, 1)
    ax.hist(df['PTRATIO'], bins = 35)
    plt.title("Students to Teachers ratio distribution")
    plt.xlabel("PTRATIO")
    plt.ylabel("frequency")
    plt.show()
    fig=plt.figure()
    ax=fig.add_subplot(1, 1, 1)
    ax.hist(df['AGE'], bins = 35)
    plt.title("Ages of Black Owned Portion in the town")
    plt.xlabel("AGE")
    plt.ylabel("B")
    plt.show()
    N = 506
    x = df.AGE
    y = df.B
    colors = np.random.rand(N)

    plt.scatter(x, y, c=colors)
    plt.show()
    sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='Blues')
    df = df.fillna(df.mean())
    df.isnull().sum()
    df.rename(columns={'MEDV':'PRICE'}, inplace = True)
    corr = df.corr()
    corr.shape
    df.shape
    X = df.iloc[:,0:13] #independent columns
    y = df.iloc[:,-1] #target column i.e price range
    
    y = np.round(df['PRICE'])
    #Apply SelectKBest class to extract top 5 best features
    bestfeatures = SelectKBest(score_func=chi2)
    fit = bestfeatures.fit(X,y)
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(X.columns)
    # Concat two dataframes for better visualization
    featureScores = pd.concat([dfcolumns,dfscores],axis=1)
    featureScores.columns = ['SPECS','SCORE'] #naming the dataframe columns
    featureScores
    print(featureScores.nlargest(8,'SCORE')) #print 5 best features

    plt.show()

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define and train the models

# Linear Regression
model1 = LinearRegression()
model1.fit(X_train, y_train)

# Decision Tree
model2 = DecisionTreeRegressor()
model2.fit(X_train, y_train)

# Random Forest
model3 = RandomForestRegressor()
model3.fit(X_train, y_train)

# Make predictions
y_pred_lr = model1.predict(X_test)  # Predict using linear regression model
y_pred_dt = model2.predict(X_test)  # Predict using decision tree regression model
y_pred_rf = model3.predict(X_test)  # Predict using random forest regression model

# Linear Regression
y_pred_lr = model1.predict(X_test)  # Predict using linear regression model
sns.regplot(x=y_test, y=y_pred_lr, label='Actual vs Predicted (Linear Regression)')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.legend()
plt.show()

print("Training Accuracy (Linear Regression):", model1.score(X_train, y_train) * 100)
print("Testing Accuracy (Linear Regression):", model1.score(X_test, y_test) * 100)
print("Model Accuracy (Linear Regression):", r2_score(y, model1.predict(X)) * 100)

# Decision Tree
y_pred_dt = model2.predict(X_test)  # Predict using decision tree regression model
sns.regplot(x=y_test, y=y_pred_dt, label='Actual vs Predicted (Decision Tree)')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.legend()
plt.show()

print("Training Accuracy (Decision Tree):", model2.score(X_train, y_train) * 100)
print("Testing Accuracy (Decision Tree):", model2.score(X_test, y_test) * 100)
print("Model Accuracy (Decision Tree):", r2_score(y, model2.predict(X)) * 100)

# Random Forest
y_pred_rf = model3.predict(X_test)  # Predict using random forest regression model
sns.regplot(x=y_test, y=y_pred_rf, label='Actual vs Predicted (Random Forest)')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.legend()
plt.show()

print("Training Accuracy (Random Forest):", model3.score(X_train, y_train) * 100)
print("Testing Accuracy (Random Forest):", model3.score(X_test, y_test) * 100)
print("Model Accuracy (Random Forest):", r2_score(y, model3.predict(X)) * 100)
