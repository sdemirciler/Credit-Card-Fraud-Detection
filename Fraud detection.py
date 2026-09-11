import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve
import tensorflow as tf
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    roc_auc_score,   # ROC-AUC
    log_loss,        # Cross-entropy / log loss
    precision_score, # Precision
    recall_score,    # Recall
    f1_score         # F1-score
)

#START COMMAND AT BOTTOM


df = pd.read_csv("C:\Fraud detection\creditcard.csv")



def display_charts():
    print(df.info())
    print(df.describe())
    print(df.head())
    print(df['Class'].value_counts())    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))  # 2x2 grid

    sns.countplot(x='Class', data=df, ax=axes[0,0])
    axes[0,0].set_title("Number of Normal vs Fraud")

    sns.histplot(df['Amount'], bins=30, ax=axes[0,1])
    axes[0,1].set_title("Transaction Amount Distribution")

    sns.histplot(df['Time'], bins=30, ax=axes[1,0])
    axes[1,0].set_title("Transaction Time Distribution")

    sns.boxplot(x='Class', y='Amount', data=df, ax=axes[1,1])
    axes[1,1].set_title("Boxplot of Amount by Class")

    plt.tight_layout() 
    plt.show() 



def load_prepare_data(choice):
    if choice == 1:
        
        x = df.drop(columns='Class')
        y = df['Class']
        global scaler,log_model,rf_model,mlp_model
        scaler = StandardScaler()

        x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,stratify=y,random_state=42)

        x_train[['Amount','Time']] = scaler.fit_transform(x_train[['Amount','Time']])
        x_test[['Amount','Time']] = scaler.transform(x_test[['Amount','Time']])
        log_model = build_logistic()
        rf_model = build_rf()
        mlp_model = build_mlp()
        log_model.fit(x_train,y_train)
        rf_model.fit(x_train,y_train)
        mlp_model.fit(x_train,y_train)
        predict_evaluate(x_train,x_test,y_train,y_test,log_model,rf_model,mlp_model)

 

def predict_sample(sample,log_model,rf_model,mlp_model):
    sample = sample.drop('Class')
    sample = sample.to_frame().T  # make 2D
    sample[['Amount','Time']] = scaler.transform(sample[['Amount','Time']])
    log_prob = log_model.predict_proba(sample)[0,1]
    rf_prob = rf_model.predict_proba(sample)[0,1]
    mlp_prob = mlp_model.predict_proba(sample)[0,1]
    fraud_prob = max(log_prob*0.6,rf_prob,mlp_prob)
    print(fraud_prob)
    if fraud_prob < 0.3:
        print("Not fraud")
    elif fraud_prob < 0.6:
        print("Flag for review")
    else:
        print("Fraud")


def build_logistic():
    log_model = LogisticRegression(
        C=0.1,
        solver='liblinear',
        max_iter=1000,
        class_weight='balanced',
        random_state=42
    )
    return log_model
    

def build_rf():
    global rf_model
    rf_model = RandomForestClassifier(
        n_estimators=250,
        class_weight='balanced',
        random_state=42
    )
    return rf_model

    
def build_mlp():
    global mlp_model
    mlp_model = MLPClassifier(
        hidden_layer_sizes=(32,),
        max_iter=100,
        activation='relu',
        solver='adam',
        random_state=42,
        verbose=True
    )
    return mlp_model



def predict_evaluate(x_train,x_test,y_train,y_test,log_model,rf_model,mlp_model):
    y_prob_log = log_model.predict_proba(x_test)[:,1]
    y_pred_log = (y_prob_log >= 0.5).astype(int) #decide threshold yourself and see metrics : 
    #y_pred_log = (y_prob_log >= threshold).astype(int)

    y_prob_rf = rf_model.predict_proba(x_test)[:,1]
    y_pred_rf = (y_prob_rf >= 0.15).astype(int)
    #maybe do same for this too.


    y_prob_mlp = mlp_model.predict_proba(x_test)[:,1]
    y_pred_mlp = mlp_model.predict(x_test) 

    print("Logistic Regression ROC-AUC:", roc_auc_score(y_test, y_prob_log))
    print("Logistic Regression Log Loss:", log_loss(y_test, y_prob_log))
    print("Precision:", precision_score(y_test, y_pred_log))
    print("Recall:", recall_score(y_test, y_pred_log))
    print("F1:", f1_score(y_test, y_pred_log))

    # Repeat for Random Forest
    print("Random Forest ROC-AUC:", roc_auc_score(y_test, y_prob_rf))
    print("Random Forest Log Loss:", log_loss(y_test, y_prob_rf))
    print("Precision:", precision_score(y_test, y_pred_rf))
    print("Recall:", recall_score(y_test, y_pred_rf))
    print("F1:", f1_score(y_test, y_pred_rf))




    plt.plot(mlp_model.loss_curve_)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("MLP Loss Curve")
    plt.show()

    plt.hist(y_prob_log, bins=50)
    plt.show()

    prec, rec, thresh = precision_recall_curve(y_test, y_prob_log)
    plt.plot(thresh, prec[:-1], label='Precision')
    plt.plot(thresh, rec[:-1], label='Recall')
    plt.xlabel('Threshold')
    plt.legend()
    plt.show()

    prec1,rec1,thresh1 = precision_recall_curve(y_test,y_prob_rf)
    plt.figure(figsize=(8,5))
    plt.plot(thresh1, prec1[:-1], label='Precision', color='blue')
    plt.plot(thresh1, rec1[:-1], label='Recall', color='orange')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Random Forest Precision-Recall vs Threshold')
    plt.legend()
    plt.show()






run = 0
while True:
    choice1 = int(input("1 = Compute (compulsory for run 1), 2. Predict with a fraud sample, 3. See graphs."))
    if choice1 == 1:
        load_prepare_data(1)
        run+=1
    elif choice1 == 2 and run >= 1:
        choice2 = int(input("Give a random index"))
        newdf = df[df['Class']==1]
        try:
            sample = newdf.iloc[choice2-1]
            predict_sample(sample,log_model,rf_model,mlp_model)
        except:
            print("No such index")
    else:
        display_charts()
