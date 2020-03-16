from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.preprocessing import StandardScaler

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
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=.8)
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.333)


#
#medias_X_train = np.mean(X_train, axis = 0)   
#desvs_X_train = np.std(X_train, axis = 0)  
#X_train = (X_train - medias_X_train) / desvs_X_train
#X_val = (X_val - medias_X_train) / desvs_X_train 
#X_test = (X_test - medias_X_train) / desvs_X_train 
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_val = scaler.transform(X_val)

n_classes = np.unique(y).shape[0]
set_classes = np.unique(y)
Y_train_bin = label_binarize(Y_train, classes=set_classes)
Y_test_bin = label_binarize(Y_test, classes=set_classes)




from sklearn import svm
from sklearn.model_selection import GridSearchCV

C = 215.44;
gamma = 8.84e-3;



#rang_C = np.logspace(-3, 3, 10)
#tuned_parameters = [{'C': rang_C}]
#
#
#n_dim=X_train.shape[1]
#rang_g=np.array([0.125, 0.25, 0.5, 1, 2, 4, 8])/(np.sqrt(n_dim))
#tuned_parameters = [{'C': rang_C, 'gamma': rang_g}]
#
## Train an SVM with gaussian kernel and adjust by CV the parameter C
rbf_svc = svm.SVC(kernel='rbf', C=C, gamma=gamma)
#rbf_svc  = GridSearchCV(clf_base, tuned_parameters, cv=nfold)
rbf_svc.fit(X_train, Y_train)
## Save the values of C and gamma selected and compute the final accuracy
#C_opt = rbf_svc.best_params_['C']
#g_opt = rbf_svc.best_params_['gamma']
#
#
#print "The C value selected is " + str(C_opt)
#print "The gamma value selected is " + str(g_opt)
acc_rbf_svc = rbf_svc.score(X_test, Y_test)
print("The test accuracy of the RBF SVM is %2.2f" %(100*acc_rbf_svc))

#########################################################################
###########           LINEALES             ##############################
#########################################################################


############################### PCA #####################################
from sklearn.decomposition import PCA

N_feat_max=100
# Define and train pca object
pca = PCA(n_components=N_feat_max)
pca.fit(X_train, Y_train)

# Project the training, validation and test data
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
X_val_pca = pca.transform(X_val)

# Compute new dimensions
dim_train = np.shape(X_train_pca)
dim_val = np.shape(X_test_pca)
dim_test = np.shape(X_val_pca)

print 'Dimensions of training data are: ' + str(dim_train)
print 'Dimensions of validation data are: ' + str(dim_val)
print 'Dimensions of test data are: ' + str(dim_test)
print 'Dimensions of train y are: ' + str(np.shape(X_train_pca[:,:10]))


from sklearn import svm
from sklearn.model_selection import GridSearchCV

def SVM_accuracy_evolution(X_train_t, Y_train, X_val_t, Y_val, X_test_t, Y_test, rang_feat, C2, gamma2):
    """Compute the accuracy of training, validation and test data for different the number of features given
        in rang_feat.

    Args:
        X_train_t (numpy dnarray): training data projected in the new feature space (number data x number dimensions).
        Y_train (numpy dnarray): labels of the training data (number data x 1).
        X_val_t (numpy dnarray): validation data projected in the new feature space (number data x number dimensions).
        Y_val (numpy dnarray): labels of the validation data (number data x 1).
        X_test_t (numpy dnarray): test data projected in the new feature space (number data x number dimensions).
        Y_test (numpy dnarray): labels of the test data (number data x 1).
        rang_feat: range with different number of features to be evaluated                                           
   
    """
    
    
   
    clf = svm.SVC(kernel='rbf', C=C2, gamma=gamma2)
    acc_tr = []
    acc_val = []
    acc_test = []
    for i in rang_feat:
        # Train SVM classifier
        #rang_C = np.logspace(-3, 3, 10)
        #tuned_parameters = [{'C': rang_C}]
        #nfold = 10
        #lin_svc  = GridSearchCV(clf, tuned_parameters, cv=nfold)
        clf.fit(X_train_t[:,:i], Y_train)
        
        # Compute accuracies
        acc_tr.append(clf.score(X_train_t[:,:i], Y_train))
        acc_val.append(clf.score(X_val_t[:,:i], Y_val))
        acc_test.append(clf.score(X_test_t[:,:i], Y_test))

    return np.array(acc_tr), np.array(acc_val), np.array(acc_test)
                    
# Run the function with the pca extracted features                    
rang_feat = np.arange(5, N_feat_max, 10) # To speed up the execution, we use steps of 10 features.
[acc_tr, acc_val, acc_test] = SVM_accuracy_evolution(X_train_pca, Y_train, X_val_pca, Y_val, X_test_pca, Y_test, rang_feat, C, gamma)


import matplotlib.pyplot as plt

def plot_accuracy_evolution(rang_feat, acc_tr, acc_val, acc_test):
    """Plot the accuracy evolution for training, validation and test data sets.

    Args:
        rang_feat: range with different number of features where the accuracy has been evaluated   
        acc_tr: numpy vector with the training accuracies
        acc_val: numpy vector with the validation accuracies
        acc_test: numpy vector with the test accuracies                                          
   
    """
    plt.plot(rang_feat, acc_tr, "b", label="train")
    plt.plot(rang_feat, acc_val, "g", label="validation")
    plt.plot(rang_feat, acc_test, "r", label="test")
    plt.xlabel("Number of features")
    plt.ylabel("Accuracy")
    plt.title('Accuracy evolution')
    plt.legend(['Training', 'Validation', 'Test'], loc = 4)
    

# Plot it!
plt.figure()
plot_accuracy_evolution(rang_feat, acc_tr, acc_val, acc_test)
plt.show()

pos_max = np.argmax(acc_val)
num_opt_feat = rang_feat[pos_max]
test_acc_opt = acc_test[pos_max]
print 'Number optimum of features: ' + str(num_opt_feat)
print("The optimum test accuracy is  %2.2f%%" %(100*test_acc_opt))


########################### PLS ##################################3

from sklearn.cross_decomposition import PLSSVD

N_feat_max = n_classes # As many new features as classes minus 1
# 1. Obtain PLS projections
pls = PLSSVD(n_components=N_feat_max)
pls.fit(X_train, Y_train_bin)
X_train_pls = pls.transform(X_train)
X_val_pls = pls.transform(X_val)
X_test_pls = pls.transform(X_test)

# 2. Compute and plot accuracy evolution
rang_feat = np.arange(1, N_feat_max, 1) 
[acc_tr, acc_val, acc_test] = SVM_accuracy_evolution(X_train_pls, Y_train, X_val_pls, Y_val, X_test_pls, Y_test, rang_feat, C, gamma)
plt.figure()
plot_accuracy_evolution(rang_feat, acc_tr, acc_val, acc_test)
plt.show()

# 3. Find the optimum number of features
pos_max = np.argmax(acc_val)
num_opt_feat = rang_feat[pos_max]
test_acc_opt = acc_test[pos_max]

print 'Number optimum of features: ' + str(num_opt_feat)
print("The optimum test accuracy is  %2.2f%%" %(100*test_acc_opt))


######################### CCA ########################################

from lib.mva import mva
N_feat_max = n_classes -1 # As many new features as classes minus 1
# 1. Obtain CCA projections
CCA = mva ('CCA', N_feat_max)
CCA.fit(X_train, Y_train_bin, reg=1e-2)  # Here, set reg = 1e-2
X_train_cca = CCA.transform(X_train)
X_val_cca = CCA.transform(X_val)
X_test_cca = CCA.transform(X_test)

# 2. Compute and plot accuracy evolution
rang_feat = np.arange(1, N_feat_max, 1)
[acc_tr, acc_val, acc_test] = SVM_accuracy_evolution(X_train_cca, Y_train, X_val_cca, Y_val, X_test_cca, Y_test, rang_feat, C, gamma)
plt.figure()
plot_accuracy_evolution(rang_feat, acc_tr, acc_val, acc_test)
plt.show()

# 3. Find the optimum number of features
pos_max = np.argmax(acc_test)
num_opt_feat = rang_feat[pos_max]
test_acc_opt = acc_test[pos_max]

print 'Number optimum of features: ' + str(num_opt_feat)
print("The optimum test accuracy is  %2.2f%%" %(100*test_acc_opt))

############################# LDA ###################################

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

N_feat_max = n_classes  -1# As many new features as classes minus 1
# 1. Obtain LDA or CCA projections
cca = LinearDiscriminantAnalysis(n_components=N_feat_max)
cca.fit(X_train, Y_train)
X_train_cca = cca.transform(X_train)
X_val_cca = cca.transform(X_val)
X_test_cca = cca.transform(X_test)

# 2. Compute and plot accuracy evolution
rang_feat = np.arange(1, N_feat_max, 1)
[acc_tr, acc_val, acc_test] = SVM_accuracy_evolution(X_train_cca, Y_train, X_val_cca, Y_val, X_test_cca, Y_test, rang_feat, C, gamma)
plt.figure()
plot_accuracy_evolution(rang_feat, acc_tr, acc_val, acc_test)
plt.show()

# 3. Find the optimum number of features
pos_max = np.argmax(acc_test)
num_opt_feat = rang_feat[pos_max]
test_acc_opt = acc_test[pos_max]

print 'Number optimum of features: ' + str(num_opt_feat)
print("The optimum test accuracy is  %2.2f%%" %(100*test_acc_opt))


#########################################################################
###########         NO  LINEALES             ############################
#########################################################################

###################### K-PCA ###################################

from sklearn.decomposition import PCA, KernelPCA
#
#N_feat_max=100
#
## linear PCA
#pca = PCA(n_components=N_feat_max)
#pca.fit(X_train, Y_train)
#P_train = pca.transform(X_train)
#P_test = pca.transform(X_test)
#
## KPCA
#pca_K = KernelPCA(n_components=N_feat_max, kernel="rbf", gamma=0.01)
#pca_K.fit(X_train, Y_train)
#P_train_k = pca_K.transform(X_train)
#P_test_k =pca_K.transform(X_test)
#
#print 'PCA and KPCA projections sucessfully computed'
#
## Define SVM classifier
#from sklearn import svm
#clf = svm.SVC(kernel='rbf', C=C, gamma=gamma)
#
## Train it using linear PCA projections and evaluate it
#clf.fit(P_train, Y_train)
#acc_test_lin = clf.score(P_test, Y_test)
#
## Train it using KPCA projections and evaluate it
#clf.fit(P_train_k, Y_train)
#acc_test_kernel = clf.score(P_test_k, Y_test)
#
#print("The test accuracy using linear PCA projections is  %2.2f%%" %(100*acc_test_lin))
#print("The test accuracy using KPCA projections is  %2.2f%%" %(100*acc_test_kernel))
#
#
#def plot_projected_data(data, label):
#    
#    """Plot the desired sample data assigning differenet colors according to their categories.
#    Only two first dimensions of data ar plot and only three different categories are considered.
#
#    Args:
#        data: data set to be plot (number data x dimensions). 
#        labes: target vector indicating the category of each data.
#    """
#    
#    reds = label == 0
#    blues = label == 1
#    green = label == 2
#
#    plt.plot(data[reds, 0], data[reds, 1], "ro")
#    plt.plot(data[blues, 0], data[blues, 1], "bo")
#    plt.plot(data[green, 0], data[green, 1], "go")
#    plt.xlabel("$x_1$")
#    plt.ylabel("$x_2$")
#    
#    
#plt.figure(figsize=(8, 8))
#plt.subplot(2,2,1)
#plt.title("Projected space of linear PCA for training data")
#plot_projected_data(P_train, Y_train)
#
#plt.subplot(2,2,2)
#plt.title("Projected space of KPCA for training data")
#plot_projected_data(P_train_k, Y_train)
#
#plt.subplot(2,2,3)
#plt.title("Projected space of linear PCA for test data")
#plot_projected_data(P_test, Y_test)
#
#plt.subplot(2,2,4)
#plt.title("Projected space of KPCA for test data")
#plot_projected_data(P_test_k, Y_test)
#
#plt.show()
#
#
####Validacion gamma
#
#X_train2, X_val, Y_train2, Y_val = train_test_split(X_train, Y_train, test_size=0.33)
#
## Normalizing the data
#scaler = StandardScaler()
#X_train2 = scaler.fit_transform(X_train2)
#X_val = scaler.transform(X_val)
#X_test = scaler.transform(X_test)
#
## Binarize the training labels for supervised feature extraction methods
#set_classes = np.unique(y)
#Y_train_bin2 = label_binarize(Y_train2, classes=set_classes)
#
#from sklearn.decomposition import KernelPCA
#from sklearn import svm
#
#np.random.seed(0)
#
## Defining parameters
#N_feat_max = 100
#rang_g = np.logspace(-3,0,10)#[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5 , 10, 50, 1]
#
## Variables to save validation and test accuracies
#acc_val = []
#acc_test = []
#
## Bucle to explore gamma values
#for g_value in rang_g:
#    print 'Evaluting with gamma ' + str(g_value)
#    
#    # 1. Train KPCA and project the data
#    pca_K = KernelPCA(n_components=N_feat_max, kernel="rbf", gamma=g_value)
#    pca_K.fit(X_train2, Y_train2)
#    P_train_k = pca_K.transform(X_train2)
#    P_val_k = pca_K.transform(X_val)
#    P_test_k = pca_K.transform(X_test)
#        
#    # 2. Evaluate the projection performance
#    clf = svm.SVC(kernel='rbf', C=C, gamma=gamma)
#    clf.fit(P_train_k, Y_train2)
#    acc_val.append(clf.score(P_val_k, Y_val))
#    acc_test.append(clf.score(P_test_k, Y_test))
#
## Find the optimum value of gamma and its corresponging test accuracy
#pos_max = np.argmax(acc_test)
#g_opt = rang_g[pos_max]
#acc_test_opt = acc_test[pos_max]
#
#print 'Optimum of value of gamma: ' + str(g_opt)
#print 'Test accuracy: ' + str(acc_test_opt)
#
## Train KPCA and project the data
#pca_K = KernelPCA(n_components=N_feat_max, kernel="rbf", gamma=g_opt)
#pca_K.fit(X_train2)
#P_train_k = pca_K.transform(X_train2)
#P_val_k = pca_K.transform(X_val)
#P_test_k = pca_K.transform(X_test)
#   
## Plot the projected data
#plt.figure(figsize=(15, 5))
#plt.subplot(1,3,1)
#plt.title("Projected space of KPCA: train data")
#plot_projected_data(P_train_k, Y_train2)
#
#plt.subplot(1,3,2)
#plt.title("Projected space of KPCA: validation data")
#plot_projected_data(P_val_k, Y_val)
#
#plt.subplot(1,3,3)
#plt.title("Projected space of KPCA: test data")
#plot_projected_data(P_test_k, Y_test)
#
#plt.show()




###Alternative way
def rbf_kernel_sig(X1, X2, sig=0):
    size1=X1.shape[0];
    size2=X2.shape[0];
    if X1.ndim==1:
        X1=X1[:,np.newaxis]
        X2=X2[:,np.newaxis]
    G=(X1* X1).sum(axis=1)
    H=(X2* X2).sum(axis=1)
    Q = np.tile(G, [size2,1]).T
    R = np.tile(H, [size1,1])
    KK=np.dot(X1,X2.T)
    dist=(Q + R - 2*KK)
    if sig==0:  # Then, we estimate its value
        aux=dist-np.tril(dist)
        aux=aux.reshape(size1**2,1)
        sig=np.sqrt(0.5*np.mean(aux[np.where(aux>0)]))             
    K = np.exp(-dist/sig**2);
    return K, sig
# Computing the kernel matrix
from sklearn.metrics.pairwise import rbf_kernel

g_value = 0.1

# Compute the kernel matrix (use the X_train matrix, before dividing it in validation and training data)
K_tr, gopt = rbf_kernel_sig(X_train, X_train)
K_test, gopt  = rbf_kernel_sig(X_test, X_train, sig=gopt)

def center_K(K):
    """Center a kernel matrix K, i.e., removes the data mean in the feature space.

    Args:
        K: kernel matrix                                        
    """
    size_1,size_2 = K.shape;
    D1 = K.sum(axis=0)/size_1 
    D2 = K.sum(axis=1)/size_2
    E = D2.sum(axis=0)/size_1

    K_n = K + np.tile(E,[size_1,size_2]) - np.tile(D1,[size_1,1]) - np.tile(D2,[size_2,1]).T
    return K_n
    
# Center the kernel matrix
K_tr_c = center_K(K_tr)
K_test_c = center_K(K_test)

from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn import svm

# Defining parameters
N_feat_max = 100


## PCA method (to complete)
# 1. Train PCA with the kernel matrix and project the data
pca_K2 = PCA(n_components=N_feat_max)
pca_K2.fit(K_tr_c, Y_train) 
P_train_k2 = pca_K2.transform(K_tr_c) 
P_test_k2 = pca_K2.transform(K_test_c)
        
# 2. Evaluate the projection performance
clf = svm.SVC(kernel='linear')
clf.fit(P_train_k2, Y_train)
print 'Test accuracy with PCA with a kenel matrix as input: '+ str(clf.score(P_test_k2, Y_test))

## KPCA method (for comparison purposes)
# 1. Train KPCA and project the data
# Fixing gamma to 0.5 here, it is equivalent to gamma=1 in rbf function
pca_K = KernelPCA(n_components=N_feat_max, kernel="rbf", gamma=0.01) 
pca_K.fit(X_train)
P_train_k = pca_K.transform(X_train)
P_test_k = pca_K.transform(X_test)
        
# 2. Evaluate the projection performance
clf = svm.SVC(kernel='rbf', C=C, gamma=gamma)
clf.fit(P_train_k, Y_train)
print 'Test accuracy with KPCA: '+ str(clf.score(P_test_k, Y_test))




###################### KCCA ##########################
from lib.mva import mva

# Defining parameters
N_feat_max = n_classes


## PCA method (to complete)
# 1. Train PCA with the kernel matrix and project the data
CCA = mva('CCA', N_feat_max)
CCA.fit(K_tr_c, Y_train_bin, reg=1e-2) 
P_train_k2 = CCA.transform(K_tr_c) 
P_test_k2 = CCA.transform(K_test_c)
        
# 2. Evaluate the projection performance


nfold=10

rang_C = np.logspace(-3, 3, 10)
tuned_parameters = [{'C': rang_C}]
#
#
n_dim=X_train.shape[1]
rang_g=np.array([0.125, 0.25, 0.5, 1, 2, 4, 8])/(np.sqrt(n_dim))
tuned_parameters = [{'C': rang_C, 'gamma': rang_g}]
#
## Train an SVM with gaussian kernel and adjust by CV the parameter C
clf_base = svm.SVC(kernel='rbf')
rbf_svc  = GridSearchCV(clf_base, tuned_parameters, cv=nfold)
rbf_svc.fit(P_train_k2, Y_train)


#clf = svm.SVC(kernel='rbf', C=C, gamma=gamma/10)
#clf.fit(P_train_k2, Y_train)
print 'Test accuracy with CCA with a kenel matrix as input: '+ str(rbf_svc.score(P_test_k2, Y_test))




###################### KPLS ############################

from sklearn.cross_decomposition import PLSSVD
# Defining parameters
N_feat_max = n_classes


## PCA method (to complete)
# 1. Train PCA with the kernel matrix and project the data
pls = PLSSVD(n_components=N_feat_max)
pls.fit(K_tr_c, Y_train_bin) 
P_train_k2 = pls.transform(K_tr_c) 
P_test_k2 = pls.transform(K_test_c)
        
# 2. Evaluate the projection performance
#clf = svm.SVC(kernel='rbf', C=C, gamma=gamma/32)
#clf.fit(P_train_k2, Y_train)
nfold=10

rang_C = np.logspace(-3, 3, 10)
tuned_parameters = [{'C': rang_C}]
#
#
n_dim=X_train.shape[1]
rang_g=np.array([0.125, 0.25, 0.5, 1, 2, 4, 8])/(np.sqrt(n_dim))
tuned_parameters = [{'C': rang_C, 'gamma': rang_g}]
#
## Train an SVM with gaussian kernel and adjust by CV the parameter C
clf_base = svm.SVC(kernel='rbf')
rbf_svc  = GridSearchCV(clf_base, tuned_parameters, cv=nfold)
rbf_svc.fit(P_train_k2, Y_train)
print 'Test accuracy with PLS with a kenel matrix as input: '+ str(rbf_svc.score(P_test_k2, Y_test))