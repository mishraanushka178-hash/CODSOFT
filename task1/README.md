# CODSOFT
#  Titanic Survival Prediction

This project is part of my *CodSoft Internship (Task 1)*.  
The goal is to build a *Machine Learning model* that predicts whether a passenger survived or not on the Titanic, based on features like age, sex, class, etc.

---

##  Dataset
The dataset used is the classic *Titanic dataset*, containing details of passengers such as:
- PassengerId  
- Survived (target variable)  
- Pclass (ticket class)  
- Name, Sex, Age  
- SibSp (siblings/spouses aboard)  
- Parch (parents/children aboard)  
- Ticket, Fare, Cabin, Embarked  

---

##  Steps Performed
1. *Data Preprocessing*  
   - Handled missing values  
   - Converted categorical features into numerical  

2. *Exploratory Data Analysis (EDA)*  
   - Visualized survival distribution  
   - Plotted graphs for Age, Sex, and Pclass  

3. *Model Building*  
   - Used *Logistic Regression* from scikit-learn  
   - Split data into training (80%) and testing (20%)  

4. *Model Evaluation*  
   - Accuracy Score: ~ *81%*  
   - Classification Report (Precision, Recall, F1-score)  
   - Confusion Matrix visualization  

---

##  Results
- Logistic Regression achieved *~81% accuracy*.  
- Survival chances were strongly influenced by *gender* and *class*.  

---

## ⚙ Technologies Used
- Python  
- Pandas, NumPy  
- Seaborn, Matplotlib  
- Scikit-learn