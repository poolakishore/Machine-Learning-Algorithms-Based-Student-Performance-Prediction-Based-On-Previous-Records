from ast import alias
from concurrent.futures import process
from django.shortcuts import render

# Create your views here.
from django.shortcuts import render, HttpResponse
from django.contrib import messages

import Student_performance_prediction

from .forms import UserRegistrationForm
from .models import UserRegistrationModel
from django.conf import settings
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import datetime as dt
from sklearn import preprocessing, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelBinarizer

# Create your views here.

def UserRegisterActions(request):
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            print('Data is Valid')
            form.save()
            messages.success(request, 'You have been successfully registered')
            form = UserRegistrationForm()
            return render(request, 'UserRegistrations.html', {'form': form})
        else:
            messages.success(request, 'Email or Mobile Already Existed')
            print("Invalid form")
    else:
        form = UserRegistrationForm()
    return render(request, 'UserRegistrations.html', {'form': form})

def UserLoginCheck(request):
    if request.method == "POST":
        loginid = request.POST.get('loginid')
        pswd = request.POST.get('pswd')
        print("Login ID = ", loginid, ' Password = ', pswd)
        try:
            check = UserRegistrationModel.objects.get(
                loginid=loginid, password=pswd)
            status = check.status
            print('Status is = ', status)
            if status == "activated":
                request.session['id'] = check.id
                request.session['loggeduser'] = check.name
                request.session['loginid'] = loginid
                request.session['email'] = check.email
                print("User id At", check.id, status)
                username = request.session['loggeduser']


                return render(request, 'users/UserHomePage.html', {'username':username})
            else:
                messages.success(request, 'Your Account Not at activated')
                return render(request, 'UserLogin.html')
        except Exception as e:
            print('Exception is ', str(e))
            pass
        messages.success(request, 'Invalid Login id and password')
    return render(request, 'UserLogin.html', {})


def UserHome(request):
    username = request.session['loggeduser']
    return render(request, 'users/UserHomePage.html', {'username':username})

def DatasetView(request):
    path = settings.MEDIA_ROOT + "//" + 'student_data.csv'
    df = pd.read_csv(path, nrows=100)
    df = df.to_html
    return render(request, 'users/viewdataset.html', {'data': df})

def ml(request):
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    from time import time
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.model_selection import train_test_split,GridSearchCV
    from sklearn.metrics import confusion_matrix, roc_curve, accuracy_score, f1_score, roc_auc_score, classification_report
    from astropy.table import Table # type: ignore
    from sklearn.metrics import roc_auc_score
    df = pd.read_csv(r"C:\Users\poola\OneDrive\Desktop\69.Machine_Learning_Algorithms_based_Student_Performance_Prediction_based_on_Previo\Student_performance_prediction\CODE\student\Student_performance_prediction\media\student_data.csv")
    # dfv = pd.read_csv('C:\\Users\\Admin\\Downloads\\student-data.csv')

    # mapping strings to numeric values:
    def numerical_data():
        df['school'] = df['school'].map({'GP': 0, 'MS': 1})
        df['sex'] = df['sex'].map({'M': 0, 'F': 1})
        df['address'] = df['address'].map({'U': 0, 'R': 1})
        df['famsize'] = df['famsize'].map({'LE3': 0, 'GT3': 1})
        df['Pstatus'] = df['Pstatus'].map({'T': 0, 'A': 1})
        df['Mjob'] = df['Mjob'].map({'teacher': 0, 'health': 1, 'services': 2, 'at_home': 3, 'other': 4})
        df['Fjob'] = df['Fjob'].map({'teacher': 0, 'health': 1, 'services': 2, 'at_home': 3, 'other': 4})
        df['reason'] = df['reason'].map({'home': 0, 'reputation': 1, 'course': 2, 'other': 3})
        df['guardian'] = df['guardian'].map({'mother': 0, 'father': 1, 'other': 2})
        df['schoolsup'] = df['schoolsup'].map({'no': 0, 'yes': 1})
        df['famsup'] = df['famsup'].map({'no': 0, 'yes': 1})
        df['paid'] = df['paid'].map({'no': 0, 'yes': 1})
        df['activities'] = df['activities'].map({'no': 0, 'yes': 1})
        df['nursery'] = df['nursery'].map({'no': 0, 'yes': 1})
        df['higher'] = df['higher'].map({'no': 0, 'yes': 1})
        df['internet'] = df['internet'].map({'no': 0, 'yes': 1})
        df['romantic'] = df['romantic'].map({'no': 0, 'yes' : 1})
        df['passed'] = df['passed'].map({'no': 0, 'yes': 1})
        # reorder dataframe columns :
        col = df['passed']
        del df['passed']
        df['passed'] = col

        # feature scaling will allow the algorithm to converge faster, large data will have same scal
    def feature_scaling(df):
        for i in df:
            col = df[i]
            # let's choose columns that have large values
            if(np.max(col)>6):
                Max = max(col)
                Min = min(col)
                mean = np.mean(col)
                col  = (col-mean)/(Max)
                df[i] = col
            elif(np.max(col)<6):
                col = (col-np.min(col))
                col /= np.max(col)
                df[i] = col
    numerical_data()
    df
    # Let's scal our features
    feature_scaling(df)

    # Now we are ready for models training
    df
    df.dropna().shape
    # split data train 70 % and test 30 %

    data = df.to_numpy()
    n = data.shape[1]
    x = data[:,0:n-1]
    y = data[:,n-1]
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=0)
    logisticRegr = LogisticRegression(C=1)
    logisticRegr.fit(x_train,y_train) 
    y_pred=logisticRegr.predict(x_test)
    #let's have a look at the accuracy of the model

    Sctest=logisticRegr.score(x_test,y_test)
    Sctrain=logisticRegr.score(x_train,y_train)

    print('#Accuracy test is: ',Sctest)
    print('#Accuracy train is: ',Sctrain)


    f1 = f1_score(y_test, y_pred, average='macro')
    venky=classification_report(y_test,y_pred)
    print('\n#f1 score is: ',f1)
    # return render(request,'ml.html',{'venky':venky})
    return render(request, 'users/ml.html', {'Sctest': Sctest, 'Sctrain': Sctrain, 'f1': f1, 'venky': venky})

# def predictTrustWorthy(request):
#     if request.method == 'POST':
#         import pandas as pd
#         from sklearn.model_selection import train_test_split
#         from django.conf import settings
#         #school=request.POST.get("school")  
#         sex=request.POST.get("sex")
#         #age=request.POST.get("age")
#         address=request.POST.get("address")
#         famsize=request.POST.get("famsize")
#         Pstatus=request.POST.get("Pstatus")
#         #Medu=request.POST.get("Medu")
#         #Fedu=request.POST.get("Fedu")
#         Mjob=request.POST.get("Mjob")
#         Fjob=request.POST.get("Fjob")
#         reason=request.POST.get("reason")
#         guardian=request.POST.get("guardian")
#         #traveltime=request.POST.get("traveltime")
#         #studytime=request.POST.get("studytime")
#         #failures=request.POST.get("failures") 
#         schoolsup=request.POST.get("schoolsup")    
#         famsup=request.POST.get("famsup")  
#         paid=request.POST.get("paid")
#         activities=request.POST.get("activities")
#         nursery=request.POST.get("nursery")
#         higher=request.POST.get("higher")
#         internet=request.POST.get("internet")
#         romantic=request.POST.get("romantic")
#         #famrel=request.POST.get("famrel")
#         #freetime=request.POST.get("freetime")
#         #goout=request.POST.get("goout")
#         #Dalc=request.POST.get("Dalc")
#         #Walc=request.POST.get("Walc")
#         #health=request.POST.get("health")
#         #absences=request.POST.get("absences")

#         path = settings.MEDIA_ROOT + '//' + 'student_data.csv'
#         df = pd.read_csv(path)
#         data = df
#         data = data.dropna()

#         data['passed'] = data['passed'].replace(['no','yes'],[0,1])

#         #data = data.drop(['rbc','pc','pcc','ba','bgr','bu','sc','pcv','htn','dm','cad','appet','pe','ane'],axis = 1)
#         from sklearn.ensemble import RandomForestClassifier
#         OBJ = RandomForestClassifier(n_estimators=10,criterion='entropy')
#         X = data.iloc[:, :-1]
#         y = data['passed']
#         X_train, X_test, Y_train, Y_test = train_test_split(X,y, test_size= 0.2, random_state=101)#shuffle=True
#         test_set = [sex,address,famsize,Pstatus,Mjob,Fjob,reason,guardian,schoolsup,famsup,paid,activities,nursery,higher,internet,romantic]
#         OBJ.fit(X_train.values,Y_train)
#         print(test_set)
#         y_pred = OBJ.predict([test_set])
#         print(y_pred)        
#         if y_pred == 1:
#             msg =  'yes'
#         else:
#             msg =  'no'
#         return render(request,"users/predictForm.html",{"msg":msg})
#     else:
#         return render(request,'users/predictForm.html',{})

import random

def predictTrustWorthy(request):
    if request.method == 'POST':
        # Extracting data from the POST request
        sex = request.POST.get("sex")
        address = request.POST.get("address")
        famsize = request.POST.get("famsize")
        Pstatus = request.POST.get("Pstatus")
        Mjob = request.POST.get("Mjob")
        Fjob = request.POST.get("Fjob")
        reason = request.POST.get("reason")
        guardian = request.POST.get("guardian")
        schoolsup = request.POST.get("schoolsup")
        famsup = request.POST.get("famsup")
        paid = request.POST.get("paid")
        activities = request.POST.get("activities")
        nursery = request.POST.get("nursery")
        higher = request.POST.get("higher")
        internet = request.POST.get("internet")
        romantic = request.POST.get("romantic")

        # Loading the dataset
        path = settings.MEDIA_ROOT + '/' + 'student_data.csv'
        df = pd.read_csv(path)
        data = df.dropna()

        # Check and remove unexpected values in 'passed' column
        data['passed'] = data['passed'].apply(lambda x: 1 if x == 'yes' else 0)

        # Selecting relevant columns for training
        features = ['sex', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob', 'reason', 'guardian', 'schoolsup', 'famsup',
                    'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic']

        X = pd.get_dummies(data[features])

        # Creating the test set
        test_set = {'sex': sex, 'address': address, 'famsize': famsize, 'Pstatus': Pstatus, 'Mjob': Mjob, 'Fjob': Fjob,
                    'reason': reason, 'guardian': guardian, 'schoolsup': schoolsup, 'famsup': famsup, 'paid': paid,
                    'activities': activities, 'nursery': nursery, 'higher': higher, 'internet': internet, 'romantic': romantic}

        # Creating a DataFrame for the test set
        test_df = pd.DataFrame([test_set])

        # One-hot encoding the test set
        test_df = pd.get_dummies(test_df)

        # Matching the columns between the training and test sets
        missing_cols = set(X.columns) - set(test_df.columns)
        for col in missing_cols:
            test_df[col] = 0

        # Reordering the columns to match the training set
        test_df = test_df[X.columns]

        # Preparing the data for training
        y = data['passed']
        X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=101)

        # Initializing and training the model
        OBJ = RandomForestClassifier(n_estimators=10, criterion='entropy')
        OBJ.fit(X_train, Y_train)

        # Making predictions
        y_pred = OBJ.predict(test_df.values)
        prob_pred = OBJ.predict_proba(test_df.values)[0]  # Get probabilities for pass/fail

        # Convert the prediction to 'yes' or 'no'
        msg = 'yes' if y_pred[0] == 1 else 'no'

        # Define suggestions for both outcomes
        fail_suggestions = [
            "Consider seeking help from a tutor.",
            "Try to focus on areas where you're struggling.",
            "Create a study plan and stick to it.",
            "Join group study sessions for additional support.",
            "Meet with your teachers for extra guidance."
        ]
        
        pass_suggestions = [
            "Great job! Keep up the hard work.",
            "Stay consistent with your study habits.",
            "You might want to explore advanced subjects.",
            "Maintain a balance between academics and extracurriculars.",
            "Keep challenging yourself to achieve even more."
        ]

        # Randomly pick 2 suggestions for the student based on their result
        suggestions = []
        if msg == 'no':
            suggestions = random.sample(fail_suggestions, 2)
        else:
            suggestions = random.sample(pass_suggestions, 2)

        # Calculate the pass and fail percentage (from predicted probabilities)
        pass_prob = round(prob_pred[1] * 100, 2)
        fail_prob = round(prob_pred[0] * 100, 2)

        # Prepare the context to send to the template
        context = {
            "msg": msg,
            "suggestions": suggestions,
            "pass_prob": pass_prob,
            "fail_prob": fail_prob
        }

        return render(request, "users/result.html", context)
    else:
        return render(request, 'users/predictForm.html', {})
