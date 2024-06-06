def prediction_model(pclass,sex,age,sibsp,parch,fare,embarked,Title):
    import pickle
    x=[[pclass,sex,age,sibsp,parch,fare,embarked,Title]]
    randomforest= pickle.load(open('titanic_model.sav','rb'))
    prediction= randomforest.predict(x)
    return prediction
