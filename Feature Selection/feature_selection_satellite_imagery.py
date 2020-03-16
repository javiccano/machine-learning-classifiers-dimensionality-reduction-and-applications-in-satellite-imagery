import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.preprocessing import StandardScaler
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif
# from sklearn.metrics.pairwise import rbf_kernel
# from sklearn.feature_selection import f_classif
# from sklearn.model_selection import train_test_split

# from sklearn.preprocessing import label_binarize
from sklearn.feature_selection import RFE
from sklearn.svm import SVC
from sklearn.svm import LinearSVC




###############################################################################
#################################### P1 #######################################
###############################################################################



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




x = np.loadtxt('data.txt')
y = np.loadtxt('labels.txt')

# Plot the data
x_img = x[:, 0].reshape(145, 145)
y_img = y.reshape(145, 145)

plt.figure()
plt.imshow(y_img)

plt.show()


# No por ahora
#==============================================================================
# # Data: Remove noisy bands and select subscene ([103:107,149:162,119])
# # select_bands=np.hstack([np.arange(103),np.arange(108,149),np.arange(163,219)])
# select_bands = np.r_[np.arange(103), np.arange(108, 149), np.arange(163, 219)]
# x = x[:, select_bands]
# 
# 
#==============================================================================



# Exclude the background (class 0).
class_no0 = np.nonzero(y != 0)
x = x[class_no0]
y = y[class_no0]




n_classes = np.unique(y).shape[0]



n_features = x.shape[1]
n_classes = np.unique(y).shape[0]

print("Dataset size information:")
print("n_features: %d" % n_features)
print("n_classes: %d" % n_classes)


###############################################################################
# Preparing the data

# Initialize the random generator seed to compare results
np.random.seed(1)

# Split into a training set and a test set using a stratified k fold

# split into a training and testing set
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.25)

# split into a training and validation set
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.333)

# Normalizing the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_val = scaler.transform(X_val)

# Binarize the labels for some feature selection methods
set_classes = np.unique(y)
Y_train_bin = label_binarize(Y_train, classes=set_classes)

print("Number of training samples: %d" % X_train.shape[0])
print("Number of validation samples: %d" % X_val.shape[0])
print("Number of test samples: %d" % X_test.shape[0])

################################################################################
#################### FILTERING METHODS ##############################
################################################################################


################################# F-score ##############################

F, p = f_classif(X_train, Y_train)  # Returns F-score and their associated p values

# sort in descending order
ind_rel_feat = np.argsort(F)[::-1]
    
# Print the feature ranking
print("Feature ranking:")

for f in range(10):
    print("%d. feature %d (%f)" % (f + 1, ind_rel_feat[f], F[ind_rel_feat[f]]))
    
    


def SVM_accuracy_evolution(X_train_s, Y_train, X_val_s, Y_val, X_test_s, Y_test, rang_feat):
    """Compute the accuracy of training, validation and test data for different the number of features given
        in rang_feat.

    Args:
        X_train_s (numpy dnarray): training data sorted by relevance (more relevant are first) (number data x number dimensions).
        Y_train (numpy dnarray): labels of the training data (number data x 1).
        X_val_s (numpy dnarray): validation data sorted by relevance (more relevant are first) (number data x number dimensions).
        Y_val (numpy dnarray): labels of the validation data (number data x 1).
        X_test_s (numpy dnarray): test data sorted by relevance (more relevant are first) (number data x number dimensions).
        Y_test (numpy dnarray): labels of the test data (number data x 1).
        rang_feat: range with different number of features to be evaluated                                           
   
    """
    
    # Define the model to train a liner SVM and adjust by CV the parameter C
    clf = svm.SVC(kernel='linear')
    acc_tr = []
    acc_val = []
    acc_test = []
    for i in rang_feat:
        # Train SVM classifier
        clf.fit(X_train_s[:, :i], Y_train)
        # Compute accuracies
        acc_tr.append(clf.score(X_train_s[:, :i], Y_train))
        acc_val.append(clf.score(X_val_s[:, :i], Y_val))
        acc_test.append(clf.score(X_test_s[:, :i], Y_test))

    return np.array(acc_tr), np.array(acc_val), np.array(acc_test)




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
    
    
    
# Define the range of features to explore
# (explore the 1000 most relevant ones, starting with 5 and usings steps of 10 features)
rang_feat = np.arange(1, 220, 5) 
X_train_s = X_train[:,ind_rel_feat]
X_val_s = X_val[:,ind_rel_feat]
X_test_s = X_test[:,ind_rel_feat]
[acc_tr, acc_val, acc_test] = SVM_accuracy_evolution(X_train_s, Y_train, X_val_s, Y_val, X_test_s, Y_test, rang_feat)

# Plot the results
plt.figure()
plot_accuracy_evolution(rang_feat, acc_tr, acc_val, acc_test)
plt.show()

# Find the optimum number of features
pos_max = np.argmax(acc_val)
num_opt_feat = rang_feat[pos_max]
test_acc_opt = acc_test[pos_max]

print 'Number optimum of features: ' + str(num_opt_feat)
print("The optimum test accuracy is  %2.2f%%" %(100*test_acc_opt))


def plot_image(image, h, w):
    """Helper function to plot a face image
    Args:
        image: numpy vector with the image to plot (of dimensions h*w)
        h: height of the image (in number of pixels)
        w: width of the image (in number of pixels)  """  
    
    plt.imshow(image.reshape((h, w)), cmap=plt.cm.gray)
    plt.xticks(())
    plt.yticks(())
    plt.colorbar()
    

#==============================================================================
# # Example to plot an image
# plt.figure(figsize=(6,6))
# plot_image(F, h, w)
# plt.show()
#==============================================================================



# Create the mask of selected features
# Several lines to <FILL IN>

mask = np.zeros(len(F))
mask[ind_rel_feat[0:num_opt_feat]] = 1

#==============================================================================
# # Plot it!
# plt.figure(figsize=(6,6))
# plot_image(mask, h, w)
# plt.show()
#==============================================================================







########################## Mutual information ########################

# Obtain MI values
MI = mutual_info_classif(X_train, Y_train, random_state =0)  # Returns MI values

# sort in descending order
ind_rel_feat = np.argsort(MI)[::-1]
    
# Print the feature ranking
print("Feature ranking:")

for f in range(10):
    print("%d. feature %d (%f)" % (f + 1, ind_rel_feat[f], MI[ind_rel_feat[f]]))
    
    
    
# Define the range of features to explore
# (explore the 1000 most relevant ones, starting with 5 and usings steps of 10 features)
rang_feat = np.arange(1, 220, 5) 
X_train_s = X_train[:,ind_rel_feat]
X_val_s = X_val[:,ind_rel_feat]
X_test_s = X_test[:,ind_rel_feat]
[acc_tr, acc_val, acc_test] = SVM_accuracy_evolution(X_train_s, Y_train, X_val_s, Y_val, X_test_s, Y_test, rang_feat)


# Plot the results
plt.figure()
plot_accuracy_evolution(rang_feat, acc_tr, acc_val, acc_test)
plt.show()

# Find the optimum number of features
pos_max = np.argmax(acc_val)
num_opt_feat = rang_feat[pos_max]
test_acc_opt = acc_test[pos_max]

print 'Number optimum of features: ' + str(num_opt_feat)
print("The optimum test accuracy is  %2.2f%%" %(100*test_acc_opt))
    
#==============================================================================
# # Create and plot the mask
# mask = np.zeros(len(F))
# mask[ind_rel_feat[0:num_opt_feat]] = 1
# plt.figure()
# plot_image(mask, h, w)
# plt.show()
#==============================================================================



######################## Random Forest ##########################3

np.random.seed(1)
# Build a forest and obtain the feature importances
forest = RandomForestClassifier(n_estimators=250)
forest.fit(X_train, Y_train)
importances = forest.feature_importances_

# Obtain the positions of the sorted features (the most relevant first)
ind_rel_feat = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(10):
    print("%d. feature %d (%f)" % (f + 1, ind_rel_feat[f], importances[ind_rel_feat[f]]))
    
    

# Define the range of features to explore
# (explore the 1000 most relevant ones, starting with 5 and usings steps of 10 features)
rang_feat = np.arange(1, 220, 5) # To speed up the execution, we use steps of 10 features.
X_train_s = X_train[:,ind_rel_feat]
X_val_s = X_val[:,ind_rel_feat]
X_test_s = X_test[:,ind_rel_feat]
[acc_tr, acc_val, acc_test] = SVM_accuracy_evolution(X_train_s, Y_train, X_val_s, Y_val, X_test_s, Y_test, rang_feat)

# Plot it!
plt.figure()
plot_accuracy_evolution(rang_feat, acc_tr, acc_val, acc_test)
plt.show()

# Find the optimum number of features
pos_max = np.argmax(acc_val)
num_opt_feat = rang_feat[pos_max]
test_acc_opt = acc_test[pos_max]

print 'Number optimum of features: ' + str(num_opt_feat)
print("The optimum test accuracy is  %2.2f%%" %(100*test_acc_opt))



#==============================================================================
# # Create and plot the mask
# mask = np.zeros(len(F))
# mask[ind_rel_feat[0:num_opt_feat]] = 1
# plt.figure()
# plot_image(mask, h, w)
# plt.show()
# 
#==============================================================================





#######################################################################
###################### HSIC relevance measurement ######################
#######################################################################


def estimate_gamma(X):
    """Estimate an appropiate valie of the gamma parameter to be used the build a RBF kernel with the data X
    Args:
        X: input data                             
    """
    size=X.shape[0];
    if X.ndim==1:
        X=X[:,np.newaxis]
        
    G=(X* X).sum(axis=1)
    KK=np.dot(X,X.T)
    
    R = np.tile(G, [size,1])
    Q = R.T
    
    dist=(Q + R - 2*KK)
   
    aux=dist-np.tril(dist)
    aux=aux.reshape(size**2,1)
    sig=np.sqrt(0.5*np.mean(aux[np.where(aux>0)]))  
    gamma = 1./sig**2  
    return gamma


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




def HSIC_rbf(X, Y):
    """Compute HSIC value between input and output data using a RBF kernel

    Args:
        X: input data
        Y: output data
    """
    if X.ndim==1:
        X=X[:,np.newaxis]
    if Y.ndim==1:
        Y=Y[:,np.newaxis]
    # 1. Estimate gamma value for X and Y
#==============================================================================
#     gamma_x = estimate_gamma(X)
#     gamma_y = estimate_gamma(Y)
#==============================================================================
    
    # 2. Compute kernel matrices
    K_x,g = rbf_kernel_sig(X,X)
    K_y,g = rbf_kernel_sig(Y,Y)
    
    # 3. Center kernel matrices
    K_xc = center_K(K_x)
    K_yc = center_K(K_y)
    
    # 4. Compute HSIC value
    N = len(Y_train)
    HSIC= N**(-2)*np.trace(np.dot(K_xc,K_yc))
    return HSIC

# Test HSIC function
i_test = 90
HSIC=HSIC_rbf(X_train[:,i_test], Y_train_bin)
print HSIC



# Compute HSIC relevances
importances=np.zeros(X_train.shape[1])
for i in range(X_train.shape[1]):
    importances[i]= HSIC_rbf(X_train[:,i], Y_train_bin)
    
# Obtain the positions of the sorted features (the most relevant first)
ind_rel_feat = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(10):
    print("%d. feature %d (%f)" % (f + 1, ind_rel_feat[f], importances[ind_rel_feat[f]]))



# Define the range of features to explore
# (explore the 1000 most relevant ones, starting with 5 and usings steps of 10 features)
rang_feat = np.arange(1, 220, 5) # To speed up the execution, we use steps of 10 features.


X_train_s = X_train[:,ind_rel_feat]
X_val_s = X_val[:,ind_rel_feat]
X_test_s = X_test[:,ind_rel_feat]
[acc_tr, acc_val, acc_test] = SVM_accuracy_evolution(X_train_s, Y_train, X_val_s, Y_val, X_test_s, Y_test, rang_feat)



# Plot it!
plt.figure()
plot_accuracy_evolution(rang_feat, acc_tr, acc_val, acc_test)
plt.show()

# Find the optimum number of features
pos_max = np.argmax(acc_val)
num_opt_feat = rang_feat[pos_max]
test_acc_opt = acc_test[pos_max]

print 'Number optimum of features: ' + str(num_opt_feat)
print("The optimum test accuracy is  %2.2f%%" %(100*test_acc_opt))

#==============================================================================
# # Create and plot the mask
# mask = np.zeros(len(F))
# mask[ind_rel_feat[0:num_opt_feat]] = 1
# plt.figure()
# plot_image(mask, h, w)
# plt.show()
#==============================================================================




#######################################################################
################### MRmr ##############################################
#######################################################################


# Variable initialization
n_var = X_train.shape[1]
var_sel = np.empty(0,dtype=int) # subset of selected features
var_cand = np.arange(n_var) # subset of candidate features

# Precompute relevances (f_classif value of all input variables wiht output variable)
relevances,p = f_classif(X_train, Y_train)

# Precomupute redundancies (correlation among all input variables, it is a matrix of n_var x n_var)
#redundancies = np.dot(X_train.T, X_train)
redundancies = np.corrcoef(np.transpose(X_train))

# Select the most relevant feature (the one with maximum relevance)
sel = np.argmax(relevances)
print sel
# Add it to the subset of selected features
var_sel = np.hstack((var_sel, sel))
# Remove it from the subset of candidate features
var_cand = np.delete(var_cand, sel)
print np.shape(var_cand)
print var_sel
print var_cand


# Iteratively select variables
for i in range(n_var-1):
    # Get relevance values of the var_cand variables
    relevances_cand= relevances[var_cand]
    
    # Compute redundancies with selected features:
    # from the redundancies matrix select the rows of var_sel and the columns of var_cand
    redundancy_sel = redundancies[var_sel[:, np.newaxis],var_cand]   
    # Average the redundancy values over the selected features (rows) 
    # to get a redundancy value for each candidate variables   
    redundancy_cand=np.mean(redundancy_sel,axis=0)
    
    # Compute MRmr = relevances_cand - redundancy_cand
    MRmr=relevances_cand - redundancy_cand
    
    # Select the new feature as the one with the maximum MRmr value
    sel=np.argmax(MRmr)
    # Add it to the subset of selected features
    var_sel=np.hstack((var_sel, var_cand[sel]))
    # Remove it from the subset of candidate features
    var_cand= np.delete(var_cand, sel)

ind_rel_feat = var_sel

# Define the range of features to explore
# (explore the 1000 most relevant ones, starting with 5 and usings steps of 10 features)
rang_feat = np.arange(1, 220, 5) # To speed up the execution, we use steps of 10 features.
X_train_s = X_train[:,ind_rel_feat]
X_val_s = X_val[:,ind_rel_feat]
X_test_s = X_test[:,ind_rel_feat]
[acc_tr, acc_val, acc_test] = SVM_accuracy_evolution(X_train_s, Y_train, X_val_s, Y_val, X_test_s, Y_test, rang_feat)

# Plot it!
plt.figure()
plot_accuracy_evolution(rang_feat, acc_tr, acc_val, acc_test)
plt.show()

# Find the optimum number of features
pos_max = np.argmax(acc_val)
num_opt_feat = rang_feat[pos_max]
test_acc_opt = acc_test[pos_max]


print 'Number optimum of features: ' + str(num_opt_feat)
print("The optimum test accuracy is  %2.2f%%" %(100*test_acc_opt))

#==============================================================================
# # Create and plot the mask
# mask = np.zeros(len(F))
# mask[ind_rel_feat[0:num_opt_feat]] = 1
# plt.figure()
# plot_image(mask, h, w)
# plt.show()
#==============================================================================



###############################################################################
#################################### P2 #######################################
###############################################################################





def SVM_accuracy_evolution(X_train_s, Y_train, X_val_s, Y_val, X_test_s, Y_test, rang_feat):
    """Compute the accuracy of training, validation and test data for different the number of features given
        in rang_feat.

    Args:
        X_train_s (numpy dnarray): training data sorted by relevance (more relevant are first) (number data x number dimensions).
        Y_train (numpy dnarray): labels of the training data (number data x 1).
        X_val_s (numpy dnarray): validation data sorted by relevance (more relevant are first) (number data x number dimensions).
        Y_val (numpy dnarray): labels of the validation data (number data x 1).
        X_test_s (numpy dnarray): test data sorted by relevance (more relevant are first) (number data x number dimensions).
        Y_test (numpy dnarray): labels of the test data (number data x 1).
        rang_feat: range with different number of features to be evaluated                                           
   
    """
    
    # Define the model to train a liner SVM and adjust by CV the parameter C
    clf = svm.SVC(kernel='linear')
    acc_tr = []
    acc_val = []
    acc_test = []
    for i in rang_feat:
        # Train SVM classifier
        clf.fit(X_train_s[:, :i], Y_train)
        # Compute accuracies
        acc_tr.append(clf.score(X_train_s[:, :i], Y_train))
        acc_val.append(clf.score(X_val_s[:, :i], Y_val))
        acc_test.append(clf.score(X_test_s[:, :i], Y_test))

    return np.array(acc_tr), np.array(acc_val), np.array(acc_test)




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


#==============================================================================
# def plot_image(image, h, w):
#     """Helper function to plot a face image
#     Args:
#         image: numpy vector with the image to plot (of dimensions h*w)
#         h: height of the image (in number of pixels)
#         w: width of the image (in number of pixels)  """  
#     
#     plt.imshow(image.reshape((h, w)), cmap=plt.cm.gray)
#     plt.xticks(())
#     plt.yticks(())
#     plt.colorbar()
#     
# 
#==============================================================================
#########################################################################
###################### Wrappers #########################################
#########################################################################


########################## RFE ########################################



print 'The training of this method can take some minutes, be patient...'
# Define the classifier, the RFE method and train it
estimator = SVC(kernel="linear")
RFE_selector = RFE(estimator,n_features_to_select=10, step=10)
RFE_selector.fit(X_train, Y_train)

# Obtain the positions of the sorted features (the most relevant first)
ind_rel_feat = np.argsort(RFE_selector.ranking_)

# Print the feature ranking
print("Feature ranking:")

for f in range(10):
    print("%d. feature %d" % (f + 1, ind_rel_feat[f]))

# Define the range of features to explore
# (explore the 1000 most relevant ones, starting with 5 and usings steps of 10 features)
rang_feat = np.arange(1, 220, 5) # To speed up the execution, we use steps of 10 features.
X_train_s = X_train[:,ind_rel_feat]
X_val_s = X_val[:,ind_rel_feat]
X_test_s = X_test[:,ind_rel_feat]
[acc_tr, acc_val, acc_test] = SVM_accuracy_evolution(X_train_s, Y_train, X_val_s, Y_val, X_test_s, Y_test, rang_feat)

# Plot it!
plt.figure()
plot_accuracy_evolution(rang_feat, acc_tr, acc_val, acc_test)
plt.show()

# Find the optimum number of features
pos_max = np.argmax(acc_val)
num_opt_feat = rang_feat[pos_max]
test_acc_opt = acc_test[pos_max]

print 'Number optimum of features: ' + str(num_opt_feat)
print("The optimum test accuracy is  %2.2f%%" %(100*test_acc_opt))

#==============================================================================
# # Create and plot the mask
# mask = np.zeros([h*w])
# mask[ind_rel_feat[0:num_opt_feat]] = 1
# plt.figure()
# plot_image(mask, h, w)
# plt.show()
#==============================================================================

#################################################################
##################### Embedded methods ##########################
#################################################################


########################  L1-SVM ################################


np.random.seed(1)
# Defining some useful variables to save results
acc_tr = []
acc_val = []
acc_test = []
sparsity_rate = []

# Defining the range of C values to explore
rang_C = 2*np.logspace(-2, 3, 20)

print 'The training of this method can take some minutes, be patient...'
for i, C in enumerate(rang_C):
    # Define and train SVM classifier
    svm_l1 = LinearSVC(C=C, penalty="l1", dual=False)
    svm_l1.fit(X_train, Y_train)
    
    # Compute the sparsity rate (coef_l1 contains zeros due to the
    # L1 sparsity inducing norm)
    aux = np.sum(svm_l1.coef_.ravel()==0)
    sparsity_rate.append(float(aux)/(svm_l1.coef_.shape[0]*svm_l1.coef_.shape[1]))
    
    # Compute accuracies
    acc_tr.append(svm_l1.score(X_train, Y_train))
    acc_val.append(svm_l1.score(X_val, Y_val))
    acc_test.append(svm_l1.score(X_test, Y_test))

    
# Plot the accuracy curves
plt.figure()
plot_accuracy_evolution(sparsity_rate, acc_tr, acc_val, acc_test)
plt.xlabel("Sparsity rate")
plt.show()

# Find the optimum value of C
index = np.argmax(acc_val)
C_opt = rang_C[index]
test_acc_opt = acc_test[index]

print 'Optimum value of C: ' + str(C_opt)
print("The optimum test accuracy is  %2.2f%%" %(100*test_acc_opt))

# Train the linear SVM with the optimum value of C
svm_l1 = LinearSVC(C=C_opt, penalty="l1", dual=False)
svm_l1.fit(X_train, Y_train)
coef_l1 = svm_l1.coef_
coef_l1.shape

#### Useful function for display purposes ##########
import matplotlib.pyplot as plt
#==============================================================================
# def plot_gallery(images, titles, h, w, n_row=4, n_col=10):
#     """Helper function to plot a gallery of portraits"""
#     plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
#     plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
#     for i in range(images.shape[0]):
#         plt.subplot(n_row, n_col, i + 1)
#         plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
#         plt.title(titles[i], size=12)
#         plt.xticks(())
#         plt.yticks(())
# ##################################################### 
# 
#==============================================================================
#==============================================================================
# # Create a mask per class
mask_class = np.zeros(coef_l1.shape)
mask_class [np.where(coef_l1>0)] = 1
# 
# # Plot all the masks
# titles = ['class '+str(c) for c in set_classes]
# plot_gallery(mask_class, titles, h, w)
# plt.show()
#==============================================================================



# Finding the selected fatures
ind_var_sel = np.where(sum(mask_class)>0) # <FILL IN>

# Save the number of selected features
num_var_sel = np.shape(ind_var_sel)# <FILL IN>

# Evaluating performance (for comparison purposes, let's use SVC classifier)
clf = svm.SVC(kernel='linear')
clf.fit(X_train, Y_train)
acc_test = clf.score(X_test, Y_test)# <FILL IN>

print 'Number optimum of features: ' + str(num_var_sel )
print("The test accuracy with real feature selection is  %2.2f%%" %(100*acc_test))

#==============================================================================
# # Obtain and plot the mask
# mask = np.zeros([h*w])
# mask[ind_rel_feat[0:num_opt_feat]] = 1
# #mask = # <FILL IN>
# plot_image(mask, h, w)
# plt.show()
#     
#==============================================================================
#Plot a relevance mask
# <FILL IN>
















from sklearn.linear_model import LogisticRegression
np.random.seed(1)
# Defining some useful variables to save results
acc_tr = []
acc_val = []
acc_test = []
sparsity_rate = []



# Defining the range of C values to explore
rang_C = np.logspace(-3, 3, 10)

print 'The training of this method can take some minutes, be patient...'
for i, C in enumerate(rang_C):
    
    
    
    
    # Define and train SVM classifier
    
    lr_l1 = LogisticRegression(C=C, penalty="l1", dual=False)
    lr_l1.fit(X_train, Y_train)
    
    # Compute the sparsity rate (coef_l1 contains zeros due to the
    # L1 sparsity inducing norm)
    aux = np.sum(lr_l1.coef_.ravel()==0)
    sparsity_rate.append(float(aux)/(lr_l1.coef_.shape[0]*lr_l1.coef_.shape[1]))
    
    # Compute accuracies
    acc_tr.append(lr_l1.score(X_train, Y_train))
    acc_val.append(lr_l1.score(X_val, Y_val))
    acc_test.append(lr_l1.score(X_test, Y_test))

    
# Plot the accuracy curves
plt.figure()
plot_accuracy_evolution(sparsity_rate, acc_tr, acc_val, acc_test)
plt.xlabel("Sparsity rate")
plt.show()

# Find the optimum value of C
index = np.argmax(acc_val)
C_opt = rang_C[index]
test_acc_opt = acc_test[index]

print 'Optimum value of C: ' + str(C_opt)
print("The optimum test accuracy is  %2.2f%%" %(100*test_acc_opt))






# Train the linear SVM with the optimum value of C
lr_l1 = LogisticRegression(C=C_opt, penalty="l1", dual=False)
lr_l1.fit(X_train, Y_train)
coef_l1 = lr_l1.coef_
coef_l1.shape

#==============================================================================
# #### Useful function for display purposes ##########
# 
# def plot_gallery(images, titles, h, w, n_row=4, n_col=10):
#     """Helper function to plot a gallery of portraits"""
#     plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
#     plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
#     for i in range(images.shape[0]):
#         plt.subplot(n_row, n_col, i + 1)
#         plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
#         plt.title(titles[i], size=12)
#         plt.xticks(())
#         plt.yticks(())
# ##################################################### 
#==============================================================================

#==============================================================================
# # Create a mask per class
mask_class = np.zeros(coef_l1.shape)
mask_class [np.where(coef_l1>0)] = 1
# 
# # Plot all the masks
# titles = ['class '+str(c) for c in set_classes]
# plot_gallery(mask_class, titles, h, w)
# plt.show()
# 
# 
# 
#==============================================================================


# Finding the selected fatures
ind_var_sel = np.where(sum(mask_class)>0) # <FILL IN>

# Save the number of selected features
num_var_sel = np.shape(ind_var_sel)# <FILL IN>

# Evaluating performance (for comparison purposes, let's use SVC classifier)
# clf = svm.SVC(kernel='linear')
# clf.fit(X_train, Y_train)
# acc_test = clf.score(X_test, Y_test)# <FILL IN>

print 'Number optimum of features: ' + str(num_var_sel )
print("The test accuracy with real feature selection is  %2.2f%%" %(100*acc_test))

#==============================================================================
# # Obtain and plot the mask
# mask = np.zeros([h*w])
# mask[ind_rel_feat[0:num_opt_feat]] = 1
# 
# plot_image(mask, h, w)
# plt.show()
#     
#==============================================================================
#Plot a relevance mask
# <FILL IN>





###################### l1-logistic regression #######################

from sklearn.linear_model import LogisticRegression
np.random.seed(1)
# Defining some useful variables to save results
acc_tr = []
acc_val = []
acc_test = []
sparsity_rate = []



# Defining the range of C values to explore
rang_C = np.logspace(-3, 3, 10)

print 'The training of this method can take some minutes, be patient...'
for i, C in enumerate(rang_C):
    
    
    
    
    # Define and train SVM classifier
    
    lr_l1 = LogisticRegression(C=C, penalty="l1", dual=False)
    lr_l1.fit(X_train, Y_train)
    
    # Compute the sparsity rate (coef_l1 contains zeros due to the
    # L1 sparsity inducing norm)
    aux = np.sum(lr_l1.coef_.ravel()==0)
    sparsity_rate.append(float(aux)/(lr_l1.coef_.shape[0]*lr_l1.coef_.shape[1]))
    
    # Compute accuracies
    acc_tr.append(lr_l1.score(X_train, Y_train))
    acc_val.append(lr_l1.score(X_val, Y_val))
    acc_test.append(lr_l1.score(X_test, Y_test))

    
# Plot the accuracy curves
plt.figure()
plot_accuracy_evolution(sparsity_rate, acc_tr, acc_val, acc_test)
plt.xlabel("Sparsity rate")
plt.show()

# Find the optimum value of C
index = np.argmax(acc_val)
C_opt = rang_C[index]
test_acc_opt = acc_test[index]

print 'Optimum value of C: ' + str(C_opt)
print("The optimum test accuracy is  %2.2f%%" %(100*test_acc_opt))






# Train the linear SVM with the optimum value of C
lr_l1 = LogisticRegression(C=C_opt, penalty="l1", dual=False)
lr_l1.fit(X_train, Y_train)
coef_l1 = lr_l1.coef_
coef_l1.shape

#### Useful function for display purposes ##########
import matplotlib.pyplot as plt
#==============================================================================
# def plot_gallery(images, titles, h, w, n_row=4, n_col=10):
#     """Helper function to plot a gallery of portraits"""
#     plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
#     plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
#     for i in range(images.shape[0]):
#         plt.subplot(n_row, n_col, i + 1)
#         plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
#         plt.title(titles[i], size=12)
#         plt.xticks(())
#         plt.yticks(())
# ##################################################### 
#==============================================================================

#==============================================================================
# # Create a mask per class
# mask_class = np.zeros(coef_l1.shape)
# mask_class [np.where(coef_l1>0)] = 1
# 
# # Plot all the masks
# titles = ['class '+str(c) for c in set_classes]
# plot_gallery(mask_class, titles, h, w)
# plt.show()
#==============================================================================




# Finding the selected fatures
ind_var_sel = np.where(sum(mask_class)>0) # <FILL IN>

# Save the number of selected features
num_var_sel = np.shape(ind_var_sel)# <FILL IN>

# Evaluating performance (for comparison purposes, let's use SVC classifier)
clf = svm.SVC(kernel='linear')
clf.fit(X_train, Y_train)
acc_test = clf.score(X_test, Y_test)# <FILL IN>

print 'Number optimum of features: ' + str(num_var_sel )
print("The test accuracy with real feature selection is  %2.2f%%" %(100*acc_test))

#==============================================================================
# # Obtain and plot the mask
# mask = np.zeros([h*w])
# mask[ind_rel_feat[0:num_opt_feat]] = 1
# 
# plot_image(mask, h, w)
# plt.show()
#==============================================================================
    
#Plot a relevance mask
# <FILL IN>

