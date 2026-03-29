# EXNO2DS
# AIM:
      To perform Exploratory Data Analysis on the given data set.
      
# EXPLANATION:
  The primary aim with exploratory analysis is to examine the data for distribution, outliers and anomalies to direct specific testing of your hypothesis.
  
# ALGORITHM:
STEP 1: Import the required packages to perform Data Cleansing,Removing Outliers and Exploratory Data Analysis.

STEP 2: Replace the null value using any one of the method from mode,median and mean based on the dataset available.

STEP 3: Use boxplot method to analyze the outliers of the given dataset.

STEP 4: Remove the outliers using Inter Quantile Range method.

STEP 5: Use Countplot method to analyze in a graphical method for categorical data.

STEP 6: Use displot method to represent the univariate distribution of data.

STEP 7: Use cross tabulation method to quantitatively analyze the relationship between multiple variables.

STEP 8: Use heatmap method of representation to show relationships between two variables, one plotted on each axis.

## CODING AND OUTPUT
import pandas as pd

df=pd.read_csv(r"C:\Users\Downloads\titanic_dataset.csv")

df

<img width="1310" height="504" alt="image" src="https://github.com/user-attachments/assets/085f367a-b228-4a2a-80c2-bfdbf00f168e" />

df.shape

<img width="1209" height="33" alt="image" src="https://github.com/user-attachments/assets/13d49236-8b81-417f-af01-3264fe038840" />

df.set_index("PassengerId",inplace=True)

df

<img width="1351" height="542" alt="image" src="https://github.com/user-attachments/assets/71aa3195-81b0-420b-922b-caa2d237b1f8" />

df.nunique


<img width="1115" height="829" alt="image" src="https://github.com/user-attachments/assets/3c89c6a5-a244-46f3-94f5-fff85d97ec31" />


df['Sex'].value_counts()


<img width="1225" height="87" alt="image" src="https://github.com/user-attachments/assets/2a3b39e7-ebc6-4182-8dc4-80f6af735559" />


df.Survived.unique()


<img width="1209" height="43" alt="image" src="https://github.com/user-attachments/assets/d4d57403-5476-4eeb-8e0c-eff9609b2b82" />


df.rename(columns={"Sex":"Gender"},inplace=True)

df


<img width="1326" height="515" alt="image" src="https://github.com/user-attachments/assets/881f585d-7714-43c4-a295-6b2a9cdcd2c7" />


import seaborn as sns

sns.countplot(data=df)

<img width="1108" height="588" alt="image" src="https://github.com/user-attachments/assets/42dafb87-590d-45a7-a363-bb2f2b3768f8" />


sns.countplot(x="Survived",hue="Gender",data=df)


<img width="1110" height="572" alt="image" src="https://github.com/user-attachments/assets/017c3e3e-e15b-48c6-8310-f2cf63db16e7" />


sns.catplot(x="Survived",hue="Gender",data=df,kind="count")


<img width="1131" height="643" alt="image" src="https://github.com/user-attachments/assets/f1058ab7-4f5c-49e8-970e-22007d89ce8b" />


sns.catplot(x="Survived",hue="Gender",data=df,kind="violin")


<img width="1132" height="637" alt="image" src="https://github.com/user-attachments/assets/d343f3d1-74ce-4401-827e-af867e1c3a00" />


sns.boxplot(data=df)


<img width="1264" height="549" alt="image" src="https://github.com/user-attachments/assets/7984f513-f40e-409e-b1d8-e0d6b9da44aa" />


df.boxplot(column="Survived",by="Gender")


<img width="1275" height="594" alt="image" src="https://github.com/user-attachments/assets/9d43f10b-1709-42ef-a22e-29ed78855ad6" />


sns.scatterplot(data=df)


<img width="962" height="545" alt="image" src="https://github.com/user-attachments/assets/df308e1f-d277-4415-8c7d-f4153833ec9a" />


sns.scatterplot(x=df['Age'],y=df['Fare'])


<img width="1267" height="552" alt="image" src="https://github.com/user-attachments/assets/fb3c2e79-1439-4206-afb3-cceeac1bf1d4" />


sns.jointplot(x='Age',y='Fare',data=df)


<img width="1256" height="763" alt="image" src="https://github.com/user-attachments/assets/1a47f201-fcf9-428a-9348-c3fddb6be8da" />


sns.jointplot(x='Age',y='Fare',data=df,kind="kde")


<img width="1279" height="762" alt="image" src="https://github.com/user-attachments/assets/aeff2fe5-98ba-4310-895d-09fea9ed3e8a" />


sns.jointplot(x='Age',y='Fare',data=df,kind="hist")


<img width="1030" height="759" alt="image" src="https://github.com/user-attachments/assets/b945ff09-98e8-414d-8d38-c9c27c7d2cc8" />


sns.catplot(x='Gender',col='Survived',data=df,kind='count',color='green')


<img width="1298" height="644" alt="image" src="https://github.com/user-attachments/assets/422fc392-60e7-482b-841a-9867cddc15cc" />


sns.pairplot(data=df)


<img width="901" height="927" alt="image" src="https://github.com/user-attachments/assets/b40a83c1-e17b-4ff5-8570-464504ac143c" />


corr1=df.select_dtypes(include=["number"]).corr()

sns.heatmap(corr1,annot=True)


<img width="1066" height="632" alt="image" src="https://github.com/user-attachments/assets/d4159fc8-1eca-4b5c-806a-85b2b90bc1cc" />


sns.catplot(x='Gender',col='Survived',data=df,kind='count',hue="Pclass")

<img width="1353" height="640" alt="image" src="https://github.com/user-attachments/assets/676bf1a3-ad80-467b-ab22-d5c78091a4e0" />


import matplotlib.pyplot as plt

fig,ax1=plt.subplots(figsize=(8,5))

pt=sns.boxplot(ax=ax1,x='Pclass',y='Age',hue='Gender',data=df)


<img width="1159" height="558" alt="image" src="https://github.com/user-attachments/assets/28c7a2ea-cd25-4872-8afb-f7a98b943f9f" />



# RESULT
 Thus performing Exploratory Data Analysis on the given data set.
