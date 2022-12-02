import json
import random
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, Normalizer
from sklearn.model_selection import cross_val_score, train_test_split
import matplotlib.pyplot as plt
import random
plt.rcParams.update({'font.size': 16})
def get_data():  ### Reads the json data file and converts the colour values into an array of parameters
    file = open("data_colors.json", "r") 
    data = json.loads(file.read())
    random.shuffle(data)
    
    
    inputParams = []
    results = []
    for vidInfo in data:
        params = []
        for col in vidInfo["colors"]:
            params.append(vidInfo["colors"][col])
        inputParams.append(params)

        results.append(vidInfo["topic"])
        
    return (inputParams, results)

def get_topic_pred(results, topic): ### converts the multiclass results into binary based on the input topic
    new_results = []
    for i in range(0, len(results)):
        if results[i] == topic:
            new_results.append(1)
        else:
            new_results.append(-1)

    return new_results
    
def learn(model_param, topics_array): ### generates the false postive rate, false negative rate and scores for each topic based on the input model
    
    fprs = []
    tprs = []
    scores = []
    for topic in topics_array:
        score, fpr, tpr = single_score(topic, model_param)
        scores.append(score)
        fprs.append(fpr)
        tprs.append(tpr)
    
    return scores, fprs, tprs
  
def single_score(topic, model_param): ## called by learn() to generate the fpr and tpr and accuracy score for the input topic
    params, results = get_data()
    binary_results = get_topic_pred(results,topic)
    
    norm = Normalizer().fit(params)
    params = norm.fit_transform(params)
    
    model_param.fit(params, binary_results)
    score = cross_val_score(model_param, params, binary_results, cv=5,scoring='accuracy')
    rocs = model_param.predict_proba(params)[:,1]
    fpr, tpr, _ = roc_curve(binary_results, rocs)
    
    fpr_new = []
    tpr_new = []
    for i in range(0, len(fpr)):
        fpr_new.append(fpr[i])
        tpr_new.append(tpr[i])
    
    return score.mean(), fpr_new, tpr_new
    
def poly_cross_val(model, name): ## uses cross validation to plot the accuracies of each topic for polynomials
    
    topics_array = ["food", "football match", "make up tutorial", "melting ice caps", "ocean documentary"]
    knn_test_scores = []
    for topic in topics_array:  
        knn_test_scores.append(learn_test(model, topic))
    
    x = np.arange(4)
    fig, ax = plt.subplots()
    
    for i in range(0, len(topics_array)):
        ax.bar(x + i*-0.1, knn_test_scores[i], 0.1, label=topics_array[i])
    ax.set_ylabel('Accuracy Rate')
    ax.set_xlabel('Max polynomial')
    ax.set_title('Accuracy Rates for ' + name + " Classifier for each Class")
    ax.set_xticks(x, range(1,5))
    ax.legend()
    
    
    fig.tight_layout()
    plt.show()

def learn_test(model_param, topic): ## called by poly_cross_val to plot the accuracies o
    params, results = get_data()
    binary_results = get_topic_pred(results,topic)
    
    norm = Normalizer().fit(params)
    params = norm.fit_transform(params)
    
    scores = []
    poly_list = [1, 2, 3, 4]
    for max_pol in poly_list:
        print(max_pol)
        poly_func = PolynomialFeatures(max_pol)
        poly = poly_func.fit_transform(params)
        score = cross_val_score(model_param, poly, binary_results, cv=5,scoring='accuracy')
        scores.append(score.mean())
    

    return scores

def multi_test(model, name):  ### trains the input model for a multiclass classification and plots the confusion matrices and generates the score
    params, results = get_data()
    norm = Normalizer().fit(params)
    params = norm.fit_transform(params)
    
    scores = cross_val_score(model, params, results, cv=5, scoring="accuracy")
    print(scores.mean())
    
    train, test, restrain, restest = train_test_split(params,results,test_size=0.2)

    model.fit(train, restrain)
    pred = model.predict(test)
    
    topics_array = ["food", "football match", "make up tutorial", "melting ice caps", "ocean documentary"]
    mat = confusion_matrix(pred, restest)
    disp = ConfusionMatrixDisplay(mat, display_labels=topics_array)
    disp.plot()
    plt.title("Confusion Matrix for " + name + " Classifier Predictions")
    
    plt.show()
    try:
        return model.coef_
    except:
        pass
    
    

if __name__ == "__main__":
    
    topics_array = ["food", "football match", "make up tutorial", "melting ice caps", "ocean documentary"]
    
    logistic_model = LogisticRegression(max_iter=3000)
    knn_model = KNeighborsClassifier(weights="distance")
    dummy_model = DummyClassifier(strategy="uniform")
    
    ### selecting polynomial ###
    
    # poly_cross_val(knn_model, "K Nearest Neighbour")
    # poly_cross_val(logistic_model, "Logistic Regression")
    
    
    ##### single rocs ####
    
    # dummy_scores, dummy_fprs, dummy_tprs = learn(dummy_model, topics_array)
    # knn_scores, knn_fprs, knn_tprs = learn(knn_model, topics_array)
    # logistic_scores, logistic_fprs, logistic_tprs = learn(logistic_model, topics_array)
    
    # scores = np.array([dummy_scores, knn_scores, logistic_scores]).T
    # fprs = np.array([dummy_fprs, knn_fprs, logistic_fprs], dtype=object).T
    # tprs = np.array([dummy_tprs, knn_tprs, logistic_tprs], dtype=object).T
    
    # for i in range(0, len(topics_array)):
    #     plt.plot(fprs[i][0], tprs[i][0],label="Random Dummy Classifier")
    #     plt.plot(fprs[i][1], tprs[i][1],label="K Nearest Neighbour Classifier")
    #     plt.plot(fprs[i][2], tprs[i][2],label="Logistic Regression Classifier")
    #     plt.title("ROC comparison of Classifers for Prediction of \"" + topics_array[i] + "\" Class")
    #     plt.xlabel("False Positive Rate")
    #     plt.ylabel("True Positive Rate")
    #     plt.legend()
    #     plt.show()
    #     print(scores[i])
        
    ##### multinomial #####
    
    # multi_test(dummy_model, "Random Dummy")
    # coefs = multi_test(logistic_model, "Logistic Regression")
    # multi_test(knn_model, "K Nearest Neighbours")
    
    
    
    
        

    
