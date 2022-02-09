import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

house=pd.read_csv("C:\\Users\\acer\\Desktop\\bengluru house\\Bengaluru_House_Data.csv")
house1=house.copy()
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
#print(house.isnull().sum())
#print(house.head())


house1.drop(columns="availability",inplace=True)

#          Working on integer values

num=house.select_dtypes(include=["int64","float64"])
#print(num.isnull().sum())

num_fill=num.fillna(num.median())

#print(num_fill.isnull().sum())

house1.update(num_fill)
#print(house1.isnull().sum())

#print(num_fill["bath"].unique())


#                Cleaning the data


# Working on Size Column

cat_soci_fill=house1["size"].fillna(house1["size"].mode()[0])
#print(cat_soci_fill.isnull().sum())
house1.update(cat_soci_fill)
house1["bhk"]=house1["size"].apply(lambda x : int(x.split(' ')[0]))
house1.drop(columns="size",inplace=True)
#print(house1.isnull().sum())


# Working on total_sqft column

#print(house1.total_sqft.unique())  # It has some values with range(1254-1236), nd with strings like meter.

def range_ave(x):
    tokens=x.split("-")
    if len(tokens)==2:
        return (float(tokens[0])+float(tokens[1]))/2
        try:
            float(x)
        except:
            return house1.drop(x)
        
total_sqft_clean=house1["total_sqft"].apply(range_ave)
#print(total_sqft_clean.loc[122])
house1.update(total_sqft_clean)
#print(house1["total_sqft"].loc[410])
lst_text=[]
for var in house1["total_sqft"]:
    try:
        if var is float(var):
            pass
    except:
        lst_text.append(var)        
#print(len(lst_text))

lst=["Meter","Perch",'Yard',"Acres","Cents","Guntha","Grounds"]

for x in lst:
    
        ind=house1[house1["total_sqft"].str.contains(x)==True].index
        #print(x,'of' ,ind)
        house1.drop(index=ind,inplace=True)

    

#print(house1.loc[1086])

#print(house1["total_sqft"].dtype)

ds=pd.to_numeric(house1["total_sqft"])
house1.drop(columns=["total_sqft"])
house1["total_sqft"]=ds
#print(house1.info())



# Creating a new column for Price Per Sqrt feet area Just for Outlier detection in next step

house1["price_per_sqft"]=house1["price"]*100000/house1["total_sqft"]
#print(house1.info())



#         Working on catagorical values


cat=house1.select_dtypes(include="object")
#print(cat.isnull().sum())



cat_soci_fill=cat["society"].fillna(cat["society"].mode()[0])
#print(cat_soci_fill.isnull().sum())
house1.update(cat_soci_fill)

#print(house1.isnull().sum())


house1.dropna(inplace=True)
#print(house1.isnull().sum())




#         Dimentionality reduction [ For One Hot Encoding]

# Working on Location column because it has too many locations and give many features on one-hot encoding.

location_stats=house1.groupby("location")["location"].agg("count").sort_values(ascending=True)   # Counting for total rows for a specific location.

#print(len(location_stats[location_stats<=10]))
location_stats_lessthen10= location_stats[location_stats<=10]
house1["location"]=house1["location"].apply(lambda x :  "other" if x in location_stats_lessthen10 else x)
#print(house1.groupby("location")["location"].agg("count").sort_values(ascending=True))  # Checking for new value other

# Working on society column because it has too many locations and give many features on one-hot encoding.

society_stats=house1.groupby("society")["society"].agg("count").sort_values(ascending=True)
house1["society"]=house1["society"].apply(lambda x: "other" if x in society_stats[society_stats<=20] else x)
#print(house1.groupby("society")["society"].agg("count").sort_values(ascending=True))  # Cheking for new value other.


# Working on area_type column because it has too many locations and give many features on one-hot encoding.


#print(house1.groupby("area_type")["area_type"].agg("count").sort_values(ascending=True))
# This column does not have too many unique values so we dont want to apply dimentionality reduction.
#print(house1.shape)

                 #  Getting Outliers and removing them

#working on bhk and total sqft feature to get outliers, generally for 1 bedroom space needed is approx 300sqft.

#print(house1[house1.total_sqft/house1.bhk<300].index)  #Getting outliers

outlier_index=[]
for i in house1[house1.total_sqft/house1.bhk<300].index:
    outlier_index.append(i)
#print(outlier_index)

for i in outlier_index:
    house1.drop(index=i,inplace=True)

#print(house1.shape)


# Working on bath column to get outliers

#print(house1[house1.bath > house1.bhk+2])  # Generally there should be bhk+2 bathrooms in it

#print(house1.shape)
bath_outlier_index=[]
for i in house1[house1.bath > house1.bhk+2].index:
    bath_outlier_index.append(i)

#print(bath_outlier_index)

house1.drop(index=bath_outlier_index,inplace=True)

#print(house1.shape)

#Working on price_per_sqft columns for outlier detection

#print(house1.head())

"""
sns.scatterplot(data=house1, x="total_sqft",y="price")
sns.show()
"""

#print(house1.price_per_sqft.describe())


# Working on location column

def remove_pps_outliers(df):
    df_out = pd.DataFrame()
    for key, subdf in df.groupby('location'):
        m = np.mean(subdf.price_per_sqft)
        st = np.std(subdf.price_per_sqft)
        reduced_df = subdf[(subdf.price_per_sqft>(m-st)) & (subdf.price_per_sqft<=(m+st))]
        df_out = pd.concat([df_out,reduced_df],ignore_index=True)
    return df_out
house1 = remove_pps_outliers(house1)



# Working on bhk columnwith respect to location column

def remove_bhk_outliers(df):
    exclude_indices = np.array([])
    for location, location_df in df.groupby('location'):
        bhk_stats = {}
        for bhk, bhk_df in location_df.groupby('bhk'):
            bhk_stats[bhk] = {
                'mean': np.mean(bhk_df.price_per_sqft),
                'std': np.std(bhk_df.price_per_sqft),
                'count': bhk_df.shape[0]
            }
        for bhk, bhk_df in location_df.groupby('bhk'):
            stats = bhk_stats.get(bhk-1)
            if stats and stats['count']>5:
                exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.price_per_sqft<(stats['mean'])].index.values)
    return df.drop(exclude_indices,axis='index')
house1 = remove_bhk_outliers(house1)


"""
plt.hist(house1.price_per_sqft,rwidth=0.8)
plt.xlabel("Price Per Square Feet")
plt.ylabel("Count")
#plt.show()
"""




               #One hot Encoding
#print(house1.head())
#Working on Location column
dumies=pd.get_dummies(house1["location"])
house1=pd.concat([house1,dumies.drop(columns="other")],axis=1)

house1.drop(columns="location",inplace=True)

#Working on area_type column

area_dict={"Plot  Area":0,"Carpet  Area":1,"Built-up  Area":2,"Super built-up  Area":3}

house1["area_type_encode"]=  house1["area_type"].map(area_dict)


house1.drop(columns="price_per_sqft",inplace=True)
house1.drop(columns="area_type",inplace=True)
house1.drop(columns="society",inplace=True)


#print(house1.tail())


           # Model training


from sklearn.model_selection import train_test_split

x=house1.drop(columns="price")
y=house1["price"]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

print("shape of x_train =",x_train.shape,
      "shape of x_test =",x_test.shape,
      "shape of y_train =",y_train.shape,
      "shape of y_test =",y_test.shape)



from sklearn.linear_model import LinearRegression

regressor=LinearRegression()
regressor.fit(x_train,y_train)

score=regressor.score(x_test, y_test)

#print(score)
predict=regressor.predict(x_test)

#print(predict)
#print(y_test.head(3))




# Use K Fold cross validation to measure accuracy of our LinearRegression model
"""

from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score

cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)

total_splits=cross_val_score(LinearRegression(), x, y, cv=cv)

print(total_splits)





# Checking for more algorithms and different parameters to get best score

from sklearn.linear_model import Lasso

ls=Lasso(alpha=1,selection="random")
ls.fit(x_test,y_test)

score=ls.score(x_test,y_test)

print(score)
"""

#print(x_test.head(2))


 #  Best score we are getting it by linear regression so we should deploy it.



# Creating a function get input and give predicted price

#print(np.where(x.columns=='8th Phase JP Nagar')[0][0])

def get_price(sqft,bath,balcony,bhk,location,area_type):
    loc_index=np.where(x.columns==location)[0][0]
    X=np.zeros(len(x.columns))
    X[0]=sqft
    X[1]=bath
    X[2]=balcony
    X[3]=bhk
    X[244]=area_type
    X[np.where(x.columns==location)[0][0]]=1
    
    return regressor.predict([X])[0]

print("Rs.",(get_price(1500,3,2,3,"Rajaji Nagar",4)*10000).round(2))



           # Visualizing the Model
"""
plt.figure(figsize=(10,10))
plt.scatter(y_test,predict,c="crimson")
plt.yscale("log")
plt.xscale("log")

p1=max(max(predict),max(y_test))
p2=min(min(predict),min(y_test))

plt.plot([p1,p2],[p1,p2],"b-")
plt.xlabel("Actual value", fontsize=15)
plt.ylabel("Predicted value", fontsize=15)
plt.axis("equal")
plt.show()
"""


      # Saving the model for Deployment


"""
import pickle
with open("bengluru.pickel","wb")as f:
    pickle.dump(regressor,f)

import json
columns={"data_columns":[col.lower()for col in x.columns]}

with open("columns.json","w")as f:
    f.write(json.dump(columns,f))
"""




