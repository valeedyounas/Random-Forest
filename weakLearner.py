# -*- coding: utf-8 -*-
"""weakLearner.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Aun6Mdx-IIyVWk6xzbA-AsMIhnL9yqIM
"""

#---------------------------------------------#
#-------| Written By: Sibt ul Hussain |-------#
#---------------------------------------------#

#---------------Instructions------------------#

# You will be writing a super class named WeakLearner
# and then will be implmenting its sub classes
# RandomWeakLearner and LinearWeakLearner. Remember
# all the overridded functions in Python are by default
# virtual functions and every child classes inherits all the
# properties and attributes of parent class.

# Your task is to  override the train and evaluate functions
# of superclass WeakLearner in each of its base classes. 
# For this purpose you might have to write the auxiliary functions as well.

#--------------------------------------------------#

# Now, go and look for the missing code sections and fill them.
#-------------------------------------------#
import numpy as np
import scipy.stats as stats
np.random.seed(seed=99)

''' MY FUNCTIONS '''

def random_float(min=0,max=1,size=1,round=1):
    while(True):
        s= (max-min) * np.random.random(size) + min
        if len(s) == len(set(s)):
            break 
    if size == 1:
        return float(np.round(s,round))
    return np.round(s,round)

def getpairs(iterable):
    from tools import powerset
    y = set(powerset(iterable=iterable))
    return [e for e in y if len(e)==2]

def entropy(Y):
    target = Y
    elements, counts = np.unique(target,return_counts=True)
    return np.sum( [ (-counts[i] / np.sum(counts)) * np.log2(counts[i] / np.sum(counts)) for i in range(len(elements)) ] )

class WeakLearner: # A simple weaklearner you used in Decision Trees...
    """ A Super class to implement different forms of weak learners...


    """
    def __init__(self):
        """
        Input:
            

        """

    def train(self,X, Y):
        '''
            Trains a weak learner from all numerical attribute for all possible split points for
            possible feature selection
            
            Input:
            ---------
            X: features [m x d]
            Y: labels
            
            Returns:
            ----------
            v: splitting threshold
            score: splitting score
            Xlidx: Index of examples belonging to left child node
            Xridx: Index of examples belonging to right child node
            
        '''
        nexamples,nfeatures=X.shape

        #-----------------------TODO-----------------------#
        #--------Write Your Code Here ---------------------#
        best_score = np.NINF
        self._bestSplitValue = 0.0
        lidx = []
        ridx = []
        self._index = 0
        X_t = X.T
        for i in range(nfeatures):
            splitvalue, score, Xlidx, Xridx = self.evaluate_numerical_attribute(X_t[i],Y)
            if score > best_score:
                best_score = score
                self._bestSplitValue = splitvalue
                lidx = Xlidx
                ridx = Xridx
                self._index = i
        
        return self._bestSplitValue,best_score,lidx,ridx,self._index
        
        
        #---------End of Your Code-------------------------#
        return v,score, Xlidx,Xridx
    def evaluate(self,X):
        """
        Evalute the trained weak learner  on the given example...
        """ 
        #-----------------------TODO-----------------------#
        #--------Write Your Code Here ---------------------#
        
        return X.T[self._index]<self._bestSplitValue    
        
        #---------End of Your Code-------------------------#
    def evaluate_numerical_attribute(self,feat, Y):
        '''
            Evaluates the numerical attribute for all possible split points for
            possible feature selection
            
            Input:
            ---------
            feat: a contiuous feature
            Y: labels
            
            Returns:
            ----------
            v: splitting threshold
            score: splitting score
            Xlidx: Index of examples belonging to left child node
            Xridx: Index of examples belonging to right child node
            
        '''
        
        classes=np.unique(Y)
        #-----------------------TODO-----------------------#
        #--------Write Your Code Here ---------------------#
        # Same code as you written in DT assignment...
        nclasses=len(classes)
        m = len(Y)
    
        total_entorpy = entropy(Y)
        feat2 = feat[1:len(feat)-1]
        feature_values = np.unique(feat2)
        weighted_entropy = np.Inf
        bestSplitPoint = 0.0
        xlidx = []
        xridx = []
        for value in feature_values: # looping through sorted values of feature   
            lidx = np.nonzero(feat<value)
            ridx = np.nonzero(feat>=value)
            lidx=np.array(lidx)
            lidx = lidx.reshape((-1,))
            ridx=np.array(ridx)
            ridx = ridx.reshape((-1,))
            # Calculating entropy for left and right nodes
            mleft = len(lidx)
            mright = len(ridx)
            ent_left = entropy(Y[lidx])
            ent_right = entropy(Y[ridx])  
            
            # Calculating weighted entropy using ent_left and ent_right
            
            _entropy = (mleft/m * ent_left) + (mright/m * ent_right)

            #Entropy should be minimum: storing min entropy in weighted_entropy
            if(_entropy < weighted_entropy):
                weighted_entropy = _entropy
                bestSplitPoint = value
                xlidx = lidx
                xridx = ridx

        
        return bestSplitPoint, total_entorpy - weighted_entropy, xlidx, xridx

                
        #correction in bestSplitPoint
        unique_f = np.unique(f)
        ind = np.argwhere(unique_f == bestSplitPoint)[0]
#         print "hyn: ", Hyn
#         print "a, bestSplitPoint, unique_f[a],unique_f[a+1], unique_f", ind, bestSplitPoint, unique_f[ind],unique_f[ind+1], unique_f
        return (bestSplitPoint + unique_f[ind - 1]) / 2.0, total_entorpy - Hyn, xlidX, xridx
            
        
        #---------End of Your Code-------------------------#
            
        return split,mingain,Xlidx,Xridx

class RandomWeakLearner(WeakLearner):  # Axis Aligned weak learner....
    """ An Inherited class to implement Axis-Aligned weak learner using 
        a random set of features from the given set of features...


    """
    def __init__(self, nsplits=+np.inf, nrandfeat=None):
        """
        Input:
            nsplits = How many nsplits to use for each random feature, (if +inf, check all possible splits)
            nrandfeat = number of random features to test for each node (if None, nrandfeat= sqrt(nfeatures) )
        """
        WeakLearner.__init__(self) # calling base class constructor...        
        self.nsplits=nsplits
        self.nrandfeat=nrandfeat

    def train(self,X, Y):
        '''
            Trains a weak learner from all numerical attribute for all possible split points for
            possible feature selection
            
            Input:
            ---------
            X: a [m x d]  features matrix
            Y: a [m x 1] labels matrix
            
            Returns:
            ----------
            v: splitting threshold
            score: splitting score
            Xlidx: Index of examples belonging to left child node
            Xridx: Index of examples belonging to right child node
            
            Since we are going to find entropy of nrandfeat e.g nrandfeat = 2; one of the feature would
            have highest entropy, that feature will become the root node, I think we also need to return
            the idx of that column/feature i.e selected as root node 
        '''
        nexamples,nfeatures=X.shape

        
        if(not self.nrandfeat):
            self.nrandfeat=int(np.round(np.sqrt(nfeatures)))

        #-----------------------TODO-----------------------#
        #--------Write Your Code Here ---------------------#
        fidxs = np.arange(0,nfeatures)
        frandidxs=np.random.choice(fidxs,self.nrandfeat,replace=False)

        best_score = np.NINF
        self._bestSplitValue = 0.0
        lidx = []
        ridx = []
        self._index = 0
        X_t = X.T
        for i in frandidxs:
            splitvalue, score, Xlidx, Xridx = self.findBestRandomSplit(X_t[i],Y)
            if score > best_score:
                best_score = score
                self._bestSplitValue = splitvalue
                lidx = Xlidx
                ridx = Xridx
                self._index = i
        
        return self._bestSplitValue,best_score,lidx,ridx,self._index
        
        #---------End of Your Code-------------------------#
        return minscore, bXl,bXr

    def evaluate(self,X):
        """
        Evalute the trained weak learner  on the given example...
        """ 
        #-----------------------TODO-----------------------#
        #--------Write Your Code Here ---------------------#
        
        return X.T[self._index]<self._bestSplitValue

    def findBestRandomSplit(self,feat,Y):
        """
            
            Find the best random split by randomly sampling "nsplits"
            splits from the feature range...

            Input:
            ----------
            feat: [n X 1] nexamples with a single feature
            Y: [n X 1] label vector...

        """
        frange=np.max(feat)-np.min(feat)

        #import pdb;         pdb.set_trace()
        #-----------------------TODO-----------------------#
        #--------Write Your Code Here ---------------------#
        classes = set(Y)
        nclasses=len(classes)
        m = len(Y)

        total_entorpy = entropy(Y)
        feat2 = feat[1:len(feat)-1]
        if self.nsplits is np.Inf:
            feature_values = np.unique(feat2)
        else:
            _nsplits = self.nsplits
            while _nsplits>len(feat2):
                _nsplits-=1
            feature_values = np.random.choice(feat2,_nsplits,replace=False)
        
        weighted_entropy = np.Inf
        bestSplitPoint = 0.0
        xlidx = []
        xridx = []
        for value in feature_values: # looping through sorted values of feature   
            lidx = np.nonzero(feat<value)
            ridx = np.nonzero(feat>=value)
            lidx=np.array(lidx)
            lidx = lidx.reshape((-1,))
            ridx=np.array(ridx)
            ridx = ridx.reshape((-1,))

            # Calculating entropy for left and right nodes
            mleft = len(lidx)
            mright = len(ridx)
            ent_left = entropy(Y[lidx])
            ent_right = entropy(Y[ridx])  

            # Calculating weighted entropy using ent_left and ent_right
            
            _entropy = (mleft/m * ent_left) + (mright/m * ent_right)

            #Entropy should be minimum: storing min entropy in weighted_entropy
            if(_entropy < weighted_entropy):
                weighted_entropy = _entropy
                bestSplitPoint = value
                xlidx = lidx
                xridx = ridx
        
        

            
        return bestSplitPoint, total_entorpy - weighted_entropy, xlidx, xridx
        #---------End of Your Code-------------------------#
        return splitvalue, minscore, Xlidx, Xridx


    def calculateEntropy(self,Y, mship):
        """
            calculates the split entropy using Y and mship (logical array) telling which 
            child the examples are being split into...

            Input:
            ---------
                Y: a label array
                mship: (logical array) telling which child the examples are being split into, whether
                        each example is assigned to left split or the right one..
            Returns:
            ---------
                entropy: split entropy of the split
        """

        lexam=Y[mship]
        rexam=Y[np.logical_not(mship)]

        pleft= len(lexam) / float(len(Y))
        pright= 1-pleft

        pl= stats.itemfreq(lexam)[:,1] / float(len(lexam)) + np.spacing(1)
        pr= stats.itemfreq(rexam)[:,1] / float(len(rexam)) + np.spacing(1)

        hl= -np.sum(pl*np.log2(pl)) 
        hr= -np.sum(pr*np.log2(pr)) 

        sentropy = pleft * hl + pright * hr

        return sentropy

# build a classifier ax+by+c=0
class LinearWeakLearner(RandomWeakLearner):  # A 2-dimensional linear weak learner....
    """ An Inherited class to implement 2D line based weak learner using 
        a random set of features from the given set of features...


    """
    def __init__(self, nsplits=10):
        """
        Input:
            nsplits = How many splits to use for each choosen line set of parameters...
            
        """
        RandomWeakLearner.__init__(self,nsplits)
        

    def train(self,X, Y):
        '''
            Trains a weak learner from all numerical attribute for all possible 
            
            Input:
            ---------
            X: a [m x d] data matrix ...
            Y: labels
            
            Returns:
            ----------
            v: splitting threshold
            score: splitting score
            Xlidx: Index of examples belonging to left child node
            Xridx: Index of examples belonging to right child node
            index: pair of features decided as root
            
        '''
        nexamples,nfeatures=X.shape

        #-----------------------TODO-----------------------#
        #--------Write Your Code Here ---------------------#
        
        fidx = [ i for i in range(nfeatures) ]
        pairs = getpairs(fidx)
        best_score = np.NINF
        self.__bestSplitPoint = [] # Theeta vector 
        lidx = []
        ridx = []
        self.__index = tuple() 
        X_t = X.T
        for p in pairs:
            i = p[0]
            c1 = X_t[i]
            i = p[1]
            c2 = X_t[i]
            Xpair = np.column_stack((c1,c2)) # (m x 2)
            splitPoint, score, xlidx, xridx = self.find_best_line(Xpair,Y)
            if score > best_score:
                best_score = score
                self.__bestSplitPoint = splitPoint
                lidx = xlidx
                ridx = xridx
                self.__index = p

        return self.__bestSplitPoint,best_score,lidx,ridx,self.__index



        
        #---------End of Your Code-------------------------#

        return minscore, bXl, bXr


    def find_best_line(self,Xpair,Y):
        '''
            finds the best line and return the idx to use for left and
            right child
            Input: 
                Xpair: pair of two features of shape [m x 2]
                Y: labels of shape [m x 1]
            Output:
                BestSplitPoint: Theeta i.e values of a, b, c
                score: splitting score
                Xlidx: Index of examples belonging to left child node
                Xridx: Index of examples belonging to right child node 
        '''

        m = len(Y)
        total_entorpy = entropy(Y)
        o = np.ones((m,1))
        feat = Xpair.T[0]
        j = feat.reshape((-1,1))
        feat = Xpair.T[1]
        k = feat.reshape((-1,1))
        feature_vector = np.column_stack((j,k,o)) # (m x 3)

        weighted_entropy = np.Inf
        bestSplitPoint = 0
        xlidx = []
        xridx = []
        for _ in range(self.nsplits):
            lidx=[]
            ridx=[]
            while len(lidx)==0 or len(ridx)==0:
                Theeta = np.array([0.1,0.1,0.1])
                while (len(np.unique(np.sign(Theeta)))==1):
                    Theeta = np.random.normal(size=3)   # (3 x 1)
        
                res = np.dot(feature_vector,Theeta) # (m x 1) array of values
                lidx = [i for i in range(m) if res[i] < 0]
                ridx = [i for i in range(m) if res[i] >=0]

            res = np.dot(feature_vector,Theeta) # (m x 1) array of values
            lidx = [i for i in range(m) if res[i] < 0]
            ridx = [i for i in range(m) if res[i] >=0]
        
            # Calculating entropy for left and right nodes
            mleft = len(lidx)
            mright = len(ridx)
            ent_left = entropy(Y[lidx])
            ent_right = entropy(Y[ridx])  

            # Calculating weighted entropy using ent_left and ent_right
            
            _entropy = (mleft/m * ent_left) + (mright/m * ent_right)

            #Entropy should be minimum: storing min entropy in weighted_entropy
            if(_entropy < weighted_entropy):
                weighted_entropy = _entropy
                bestSplitPoint = Theeta
                xlidx = lidx
                xridx = ridx
        return bestSplitPoint, total_entorpy - weighted_entropy, xlidx, xridx
                

    def evaluate(self,X):
        """
        Evalute the trained weak learner  on the given example...
        """ 
        #-----------------------TODO-----------------------#
        #--------Write Your Code Here ---------------------#
        
        f1 = self.__index[0]
        f2 = self.__index[1]
        v1 = X.T[f1]
        v2 = X.T[f2]
        p=np.array([v1,v2,1])
        feature_vector=p.reshape((1,-1))
        Theeta = self.__bestSplitPoint
        res = np.dot(feature_vector,Theeta)
        return res < 0
        
        #---------End of Your Code-------------------------#

class ConicWeakLearner(RandomWeakLearner):
    """ An Inherited class to implement 2D Conic based weak learner using 
        a random set of features from the given set of features...


    """
    def __init__(self, nsplits=10):
        """
        Input:
            nsplits = How many splits to use for each choosen line set of parameters...
            
        """
        RandomWeakLearner.__init__(self,nsplits)
        

    def train(self,X, Y):
        '''
            Trains a weak learner from all numerical attribute for all possible 
            
            Input:
            ---------
            X: a [m x d] data matrix ...
            Y: labels
            
            Returns:
            ----------
            v: splitting threshold
            score: splitting score
            Xlidx: Index of examples belonging to left child node
            Xridx: Index of examples belonging to right child node
            index: pair of features decided as root
            
        '''
        nexamples,nfeatures=X.shape

        #-----------------------TODO-----------------------#
        #--------Write Your Code Here ---------------------#
        
        fidx = [ i for i in range(nfeatures) ]
        pairs = getpairs(fidx)
        best_score = np.NINF
        self.__bestSplitPoint = [] # Theeta vector 
        lidx = []
        ridx = []
        self.__index = tuple() 
        X_t = X.T
        for p in pairs:
            i = p[0]
            c1 = X_t[i]
            i = p[1]
            c2 = X_t[i]
            Xpair = np.column_stack((c1,c2)) # (m x 2)
            splitPoint, score, xlidx, xridx = self.find_best_conic(Xpair,Y)
            if score > best_score:
                best_score = score
                self.__bestSplitPoint = splitPoint
                lidx = xlidx
                ridx = xridx
                self.__index = p

        return self.__bestSplitPoint,best_score,lidx,ridx,self.__index



        
        #---------End of Your Code-------------------------#

        return minscore, bXl, bXr


    def find_best_conic(self,Xpair,Y):
        '''
            finds the best line and return the idx to use for left and
            right child
            Input: 
                Xpair: pair of two features of shape [m x 2]
                Y: labels of shape [m x 1]
            Output:
                BestSplitPoint: Theeta i.e values of a, b, c, d, e, f
                score: splitting score
                Xlidx: Index of examples belonging to left child node
                Xridx: Index of examples belonging to right child node 
        '''

        m = len(Y)
        total_entorpy = entropy(Y)
        o = np.ones((m,1))
        feat = Xpair.T[0]
        j = feat.reshape((-1,1))
        j_2 = j**2
        feat = Xpair.T[1]
        k = feat.reshape((-1,1))
        jk = j*k
        k_2 = k**2
        feature_vector = np.column_stack((j_2,jk,k_2,j,k,o)) # (m x 6)

        weighted_entropy = np.Inf
        bestSplitPoint = 0
        xlidx = []
        xridx = []
        for _ in range(self.nsplits):
            lidx=[]
            ridx=[]
            while len(lidx)==0 or len(ridx)==0:
                Theeta = np.array([0.1,0.1,0.1,0.1,0.1,0.1])
                while (len(np.unique(np.sign(Theeta)))==1):
                    Theeta = np.random.normal(size=6)   # (6 x 1)
                res = np.dot(feature_vector,Theeta) # (m x 1) array of values
                lidx = [i for i in range(m) if res[i] < 0]
                ridx = [i for i in range(m) if res[i] >=0]
            
        
            # Calculating entropy for left and right nodes
            mleft = len(lidx)
            mright = len(ridx)
            ent_left = entropy(Y[lidx])
            ent_right = entropy(Y[ridx])  

            # Calculating weighted entropy using ent_left and ent_right
            
            _entropy = (mleft/m * ent_left) + (mright/m * ent_right)

            #Entropy should be minimum: storing min entropy in weighted_entropy
            if(_entropy < weighted_entropy):
                weighted_entropy = _entropy
                bestSplitPoint = Theeta
                xlidx = lidx
                xridx = ridx
        return bestSplitPoint, total_entorpy - weighted_entropy, xlidx, xridx
                

    def evaluate(self,X):
        """
        Evalute the trained weak learner  on the given example...
        """ 
        #-----------------------TODO-----------------------#
        #--------Write Your Code Here ---------------------#
        
        f1 = self.__index[0]
        f2 = self.__index[1]
        x = X.T[f1]
        y = X.T[f2]
        v1 = x**2
        v2 = x*y
        v3 = y**2
        v4 = x
        v5 = y
        p=np.array([v1,v2,v3,v4,v5,1])
        feature_vector=p.reshape((1,-1))
        Theeta = self.__bestSplitPoint
        res = np.dot(feature_vector,Theeta)
        return res < 0
        
        #---------End of Your Code-------------------------#