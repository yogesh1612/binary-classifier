import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix,plot_roc_curve,plot_precision_recall_curve
from sklearn.metrics import precision_score,recall_score
import time
from sklearn.metrics import plot_confusion_matrix,plot_roc_curve,plot_precision_recall_curve
from sklearn.metrics import precision_score,recall_score
import matplotlib.pyplot as plt
from pathlib import Path

st.set_option('deprecation.showfileUploaderEncoding', False)
st.set_option('deprecation.showPyplotGlobalUse', False)


def main():
 
    @st.cache(persist=True) # It will help in storing dataset and do not read again when any change is done on UI
    def load_data(file):
        data = pd.read_csv(file)
        label = LabelEncoder() # transform categorical to numaric
        for col in data.columns:
            data[col] = label.fit_transform(data[col])
        return data
    
    @st.cache(persist=True)
    def split(df,test_size,target,predictors):
        y = df[target]
        x = df[predictors]
        x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=test_size,random_state=0)
        return x_train, x_test, y_train, y_test


    def plot_metrics(metrics_list):
        if 'Confusion Matrix' in metrics_list:
            st.subheader("Confusion Matrix")
            plot_confusion_matrix(model,x_test,y_test,display_labels=class_names)
            st.pyplot()

        if 'ROC Curve' in metrics_list:
            st.subheader("ROC Curve")
            plot_roc_curve(model,x_test,y_test)
            st.pyplot()

        if 'Precision-Recall Curve' in metrics_list:
            st.subheader("Precision-Recall Curve")
            plot_precision_recall_curve(model,x_test,y_test)
            st.pyplot()    
    
     

    st.title("Binary Classification Web App")
    st.sidebar.title("Binary Classification Web App")
    st.markdown(Path('intro.md').read_text()  , unsafe_allow_html=True)
    
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", accept_multiple_files=False,type=['csv'],key='uploaded_file')
    
    if uploaded_file is not None:
        data = load_data(uploaded_file)
        if st.sidebar.checkbox("Show raw data",False):
            st.write(data)
        st.sidebar.subheader("Data Partition")
        #if st.sidebar.checkbox("Train/Test Split (default 70:30)",False,key='t_t_split') :
        tt_split = st.sidebar.beta_expander("Train/Test Split")
        target = tt_split.selectbox   ("Select Target Variable",data.columns,key="target")
        predictors = [v for v in data.columns if v!=target]
        new_predictors = tt_split.multiselect ("Select Predictors",options=predictors,default=predictors)
        test_size = tt_split.number_input("Enter Test size (proportion)",0.10,0.99,step=0.1,key="test_size",value=0.30)
        class_names = data[target].unique()

        if tt_split.button("split",key = "split"):
            x_train, x_test, y_train, y_test = split(data,test_size,target,new_predictors)
        else:
            x_train, x_test, y_train, y_test = split(data,0.30,target,new_predictors)
           # st.success(f"Data splitted successfully \n no of training examples: **{x_train.shape[0]}** \n no of test examples: **{x_test.shape[0]}**")
        #st.success(f"Selected Target variable is **{target}**")
        
        st.sidebar.subheader("Model Development")
        c_classifier =  st.sidebar.beta_expander("Choose Classifier")
        classifier = c_classifier.selectbox("Classifier",("Support Vector Machine (SVM)","Logistic Regression (LR)","Random Forest (RF)"))
        
        if classifier == 'Support Vector Machine (SVM)':
            c_classifier.subheader("Model Hyperparameters")
            C = c_classifier.number_input("C (Regularization parameter)", 0.01,10.0,step = 0.01,key='C') # regularization parameter
            kernel = c_classifier.radio("kernel",("rbf","linear"),key = 'kernel')
            gamma  = c_classifier.radio("Gamma (Kernel Coefficient",("scale","auto"),key = "gamma")

            metrics = c_classifier.multiselect("What metrics to plot?",('Confusion Matrix','ROC Curve','Precision-Recall Curve'))
            if c_classifier.button("Classify",key = "classify"):
                st.subheader("Support Vector Machine (SVM) Results")
                model = SVC(C=C,kernel=kernel,gamma=gamma)
                model.fit(x_train,y_train)
                accuracy = model.score(x_test,y_test)
                y_pred = model.predict(x_test)
                st.write("Accuracy: ",accuracy.round(2))
                st.write("Precision: ",precision_score(y_test,y_pred,labels=class_names).round(2))
                st.write("Recall: ",recall_score(y_test,y_pred,labels=class_names).round(2))
                plot_metrics(metrics)


        if classifier == 'Logistic Regression (LR)':
                c_classifier.subheader("Model Hyperparameters")
                C = c_classifier.number_input("C (Regularization parameter)", 0.01,10.0,step = 0.01,key='C_LR') # regularization parameter
                max_iter = c_classifier.slider("Maximumn number of iteration",100,5000,key ="max_iter")

                metrics = c_classifier.multiselect("What metrics to plot?",('Confusion Matrix','ROC Curve','Precision-Recall Curve'))

                if c_classifier.button("Classify",key = "classify"):
                    st.subheader("Logistic Regression (LR) Results")
                    model = LogisticRegression(C=C,max_iter=max_iter)
                    model.fit(x_train,y_train)
                    accuracy = model.score(x_test,y_test)
                    y_pred = model.predict(x_test)
                    st.write("Accuracy: ",accuracy.round(2))
                    st.write("Precision: ",precision_score(y_test,y_pred,labels=class_names).round(2))
                    st.write("Recall: ",recall_score(y_test,y_pred,labels=class_names).round(2))
                    plot_metrics(metrics)        


        if classifier == 'Random Forest (RF)':
                c_classifier.subheader("Model Hyperparameters")
                n_estimators = c_classifier.number_input("The number of trees in the forest",100,5000,step=10,key ="n_estimators")
                max_depth = c_classifier.number_input("The maximum depth of the tree",1,20,step=1,key="max_depth")
                bootstrap = c_classifier.radio("Bootstrap samples when building trees",("True","False"),key="bootstrap")
                metrics = c_classifier.multiselect("What metrics to plot?",('Confusion Matrix','ROC Curve','Precision-Recall Curve'))

                if c_classifier.button("Classify",key = "classify"):
                    st.subheader("Random Forest (RF) Results")
                    model = RandomForestClassifier(n_estimators=n_estimators,max_depth=max_depth,bootstrap=bootstrap,n_jobs=-1)
                    model.fit(x_train,y_train)
                    accuracy = model.score(x_test,y_test)
                    y_pred = model.predict(x_test)
                    st.write("Accuracy: ",accuracy.round(2))
                    st.write("Precision: ",precision_score(y_test,y_pred,labels=class_names).round(2))
                    st.write("Recall: ",recall_score(y_test,y_pred,labels=class_names).round(2))
                    plot_metrics(metrics)    



if __name__ =='__main__':
    main()