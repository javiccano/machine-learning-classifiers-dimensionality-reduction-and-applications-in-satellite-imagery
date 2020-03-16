#########################################################
# Preprocessing the data
#########################################################
# Data set: AVIRIS image Indian Pine dataset:
# https://engineering.purdue.edu/~biehl/MultiSpec/hyperspectral.html

# The next code, let you:

# Load data.txt and labels.txt files
# Remove noisy bands: 104:108, 150:163, 220
# Exclude background class (class 0)
# Split data in training (20%)/test(80%) partitions (set seed to 0 for comparison purposes)

import numpy as np
from sklearn import neighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV



x = np.loadtxt('data.txt')
y = np.loadtxt('labels.txt')

# Plot the data
x_img = x[:, 0].reshape(145, 145)
y_img = y.reshape(145, 145)

plt.figure()
plt.imshow(y_img)

plt.show()

# Data: Remove noisy bands and select subscene ([103:107,149:162,119])
# select_bands=np.hstack([np.arange(103),np.arange(108,149),np.arange(163,219)])
select_bands = np.r_[np.arange(103), np.arange(108, 149), np.arange(163, 219)]
x = x[:, select_bands]


# Exclude the background (class 0).
class_no0 = np.nonzero(y != 0)
x = x[class_no0]
y = y[class_no0]

# Checking: x.shape ((10366, 200))

np.random.seed(0)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=.8)


medias_X_train = np.mean(X_train, axis = 0)   
desvs_X_train = np.std(X_train, axis = 0)  
X_train = (X_train - medias_X_train) / desvs_X_train
X_test = (X_test - medias_X_train) / desvs_X_train 

###################################### KNN #########################################

# Parameters
K_max = 20
rang_K = np.arange(1, K_max+1)
nfold = 10
vect_tr=[]
vect_test=[]
# Define a dictionary with the name of the parameters to explore as a key and the ranges to explores as value
tuned_parameters = [{'n_neighbors': rang_K}]


# Cross validation proccess 
clf_base = neighbors.KNeighborsClassifier( )
# Define the classfifier with the CV process (use GridSearchCV here!!!)
clf = GridSearchCV(clf_base, tuned_parameters, cv=nfold)
# Train it (this executes the CV)
clf.fit(X_train, y_train)

print 'CV process sucessfully finished'

#Printing results
print("Cross validation results:")

paramsFolds = clf.cv_results_['params']
meanScoreFolds = clf.cv_results_['mean_test_score']
stdScoreFolds = clf.cv_results_['std_test_score']

for fold in range(nfold):
    params = paramsFolds[fold]
    mean_score = meanScoreFolds[fold]
    std_score = stdScoreFolds[fold]
    print("For K = %d, validation accuracy is %2.2f (+/-%1.3f)%%" 
          % (params['n_neighbors'], 100*mean_score, 100*std_score / 2))

# Selecting validation error (mean values)
vect_val=meanScoreFolds

# Ploting results
"""plt.figure()
plt.plot(rang_K,vect_tr,'b', label='Training accuracy')
plt.plot(rang_K,vect_test,'r', label='Test accuracy')
plt.plot(rang_K,vect_val,'g', label='Validation accuracy')
plt.legend()
plt.xlabel('K value')
plt.ylabel('Accuracy')
plt.title('Evolution of K-NN accuracy (including validation result)')
plt.show()"""
print clf.best_estimator_
print clf.best_params_

#Assign to K_opt the value of K selected by CV
K_opt = clf.best_params_['n_neighbors']
print("The value optimum of K is %d" %(K_opt))

# Select the final classifier  and compute its test error
KNN_acc_test = clf.best_score_
print("The validation test accuracy is %2.2f" %(100*KNN_acc_test))

KNN_acc_test2 = clf.score(X_test, y_test)
print("The test accuracy is %2.2f" %(100*KNN_acc_test2))


"""def plot_boundary(clf, X, Y, plt):
    Plot the classification regions for a given classifier.

    Args:
        clf: scikit-learn classifier object.
        X (numpy dnarray): training or test data to be plotted (number data x number dimensions). Only frist two 
                            dimensions are ploted
        Y (numpy dnarray): labels of the training or test data to be plotted (number data x 1).
        plt: graphic object where you wish to plot                                             
   
    

    plot_colors = "brymc"
    plot_step = 0.02
    n_classes = np.unique(Y).shape[0]
    # Plot the decision regions
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                        np.arange(y_min, y_max, plot_step))

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    cs = plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)

    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.axis("tight")

    # Plot the training points
    for i, color in zip(range(n_classes), plot_colors):
        idx = np.where(Y == i)
        plt.scatter(X[idx, 0], X[idx, 1], c=color, cmap=plt.cm.Paired)

    plt.axis("tight")
    
plt.figure(figsize=(8, 4))
# Plot classification boundary over training data
plt.subplot(1,2,1)
plot_boundary(clf, X_train, y_train, plt)

# Plot classification boundary over test data
plt.subplot(1,2,2)
plot_boundary(clf, X_test, y_test, plt)

plt.show()"""

###################################### Binary tree #########################################


from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
clf_tree = DecisionTreeClassifier()
clf_tree.fit(X_train,y_train)
acc_tree= clf_tree.score(X_test,y_test)

print("The test accuracy of the decision tree is %2.2f" %(100*acc_tree))


###################################### Random Forest #########################################

from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
np.random.seed(0)


rang_n_trees=np.arange(1,10)
tuned_parameters = [{'n_estimators': rang_n_trees}]
nfold = 10
print 'This can take a some minutes, be patient'
# Create RF classifier object with CV
clf_RF  = RandomForestClassifier()
clf_RF = GridSearchCV(clf_RF, tuned_parameters, cv=nfold)

acc_RF_vector=[]
n_trees_vector=[]
for run in np.arange(50):
    # For each run, train it, compute its accuracy and examine the number of optimal trees
    clf_RF.fit(X_train,y_train)
    acc = clf_RF.score(X_test,y_test)
    acc_RF_vector.append(acc)
    n_trees = clf_RF.best_params_['n_estimators']
    n_trees_vector.append(n_trees)
    
# Compute averaged accuracies and number of used trees
mean_acc_RF = np.mean(acc_RF_vector) 
std_acc_RF = np.std(acc_RF_vector) 

mean_n_trees = np.mean(n_trees_vector)
std_n_trees = np.std(n_trees_vector) 

# Print the results
print('Averaged accuracy for RF classifier is %2.2f +/- %2.2f '%(100*mean_acc_RF, 100*std_acc_RF))
print('Averaged number of selected trees is %2.2f +/- %2.2f '%(mean_n_trees, std_n_trees))




############################# Bagging ##################


from sklearn.ensemble import BaggingClassifier
from sklearn import tree
np.random.seed(0)
base_learner = tree.DecisionTreeClassifier(max_depth=1)

acc_test_evol = []
rang_n_learners = range(1,50,2)
for n_learners in rang_n_learners:
    acc_test_run=[]
    for run in range(50):
        bagging = BaggingClassifier(base_learner, max_samples=0.5, max_features=0.5, n_estimators=n_learners)
        bagging.fit(X_train,y_train)
        acc = bagging.score(X_test,y_test)
        acc_test_run.append(acc)
    acc_test_evol.append(np.mean(acc_test_run))

# Ploting results
plt.figure()
plt.plot(rang_n_learners,acc_test_evol)
plt.xlabel('Number of learners')
plt.ylabel('Accuracy')
plt.title('Evolution of the bagged ensemble accuracy with the number of learners ')
plt.show()


############### Boosting ##############


###########################################################
# TODO: Replace <FILL IN> with appropriate code
###########################################################
# Initialize the random generator seed to test results
np.random.seed(0)

from sklearn.ensemble import AdaBoostClassifier

base_learner = tree.DecisionTreeClassifier(max_depth=15)

# Train a discrete Adaboost classifier and obtain its accuracy
AB_D = AdaBoostClassifier(base_learner,n_estimators=200,algorithm='SAMME')
AB_D.fit(X_train,y_train)
acc_AB_D = AB_D.score(X_test,y_test)


# Train a real Adaboost classifier and obtain its accuracy
AB_R = AdaBoostClassifier(base_learner,n_estimators=200,algorithm='SAMME.R')
AB_R.fit(X_train,y_train)
acc_AB_R = AB_R.score(X_test,y_test)

print('Accuracy of discrete adaboost ensemble is %2.2f '%(100*acc_AB_D))
print('Accuracy of real adaboost ensemble is %2.2f '%(100*acc_AB_R))


acc_AB_D_evol=[acc for acc in AB_D.staged_score(X_test, y_test)]
acc_AB_R_evol=[acc for acc in AB_R.staged_score(X_test, y_test)]


# Ploting results
rang_n_learners=np.arange(50)+1
plt.figure()
plt.subplot(211)
plt.plot(rang_n_learners,acc_AB_D_evol)
plt.xlabel('Number of learners')
plt.ylabel('Accuracy')
plt.title('Discrete AB accuracy')
plt.subplot(212)
plt.plot(rang_n_learners,acc_AB_R_evol)
plt.xlabel('Number of learners')
plt.ylabel('Accuracy')
plt.title('Real AB accuracy')
plt.show()




########################### Lineales ##################




############## SVM #########################


##### SVM kernel lineal 

from sklearn import svm
from sklearn.model_selection import GridSearchCV
rang_C = np.logspace(-3, 3, 10)
tuned_parameters = [{'C': rang_C}]

nfold = 10

# Train a liner SVM and adjust by CV the parameter C
clf_base = svm.SVC(kernel='linear')
lin_svc  = GridSearchCV(clf_base, tuned_parameters, cv=nfold)
lin_svc.fit(X_train, y_train)

# Save the value of C selected and compute the final accuracy
C_opt = lin_svc.best_params_['C']
acc_lin_svc = lin_svc.best_score_ 

print "The C value selected is " + str(C_opt)
print("The validation accuracy of the linear SVM is %2.2f" %(100*acc_lin_svc))

acc_rbf_svc = lin_svc.score(X_test, y_test)
print("The test accuracy of the linear SVM is %2.2f" %(100*acc_rbf_svc))


##### SVM kernel gaussiano

n_dim=X_train.shape[1]
rang_g=np.array([0.125, 0.25, 0.5, 1, 2, 4, 8])/(np.sqrt(n_dim))
tuned_parameters = [{'C': rang_C, 'gamma': rang_g}]

# Train an SVM with gaussian kernel and adjust by CV the parameter C
clf_base = svm.SVC(kernel='rbf')
rbf_svc  = GridSearchCV(clf_base, tuned_parameters, cv=nfold)
rbf_svc.fit(X_train, y_train)
# Save the values of C and gamma selected and compute the final accuracy
C_opt = rbf_svc.best_params_['C']
g_opt = rbf_svc.best_params_['gamma']


print "The C value selected is " + str(C_opt)
print "The gamma value selected is " + str(g_opt)
acc_rbf_svc = rbf_svc.score(X_test, y_test)
print("The test accuracy of the RBF SVM is %2.2f" %(100*acc_rbf_svc))


############ SVM polinomyal kernel 


rang_d=np.arange(1,5)
tuned_parameters = [{'C': rang_C, 'degree': rang_d}]

# Train an SVM with polynomial kernel and adjust by CV the parameter C
clf_base =  svm.SVC(kernel='poly')
poly_svc  = GridSearchCV(clf_base, tuned_parameters, cv=nfold) 
poly_svc.fit(X_train, y_train)

# Save the values of C and degree selected and compute the final accuracy
C_opt = poly_svc.best_params_['C'] 
d_opt = poly_svc.best_params_['degree'] 


print "The C value selected is " + str(C_opt)
print "The degree value selected is " + str(d_opt)
acc_poly_svc = poly_svc.score(X_test, y_test)
print("The test accuracy of the polynomial SVM is %2.2f" %(100*acc_poly_svc))






######3 Logistic regression ######


from sklearn.linear_model import LogisticRegression

rang_C = np.logspace(-3, 3, 10)
tuned_parameters = [{'C': rang_C}]
nfold = 10

# Train a LR model and adjust by CV the parameter C
clf_LR  = GridSearchCV(LogisticRegression(),
                   tuned_parameters, cv=nfold)
clf_LR.fit(X_train, y_train)
acc_test_LR=clf_LR.score(X_test, y_test) 


##### LDA  ######

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
clf_LDA = LDA()
clf_LDA.fit(X_train, y_train)
acc_test_LDA=clf_LDA.score(X_test, y_test) 

print("The test accuracy of LR is %2.2f" %(100*acc_test_LR))
print("The test accuracy of LDA is %2.2f" %(100*acc_test_LDA))