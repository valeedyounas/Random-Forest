pdata = pd.read_csv('iris.data')
pdata.columns = ['petal-length','petal-width','sepal-length','sepal-width','labels']
df = pdata.copy()
from sklearn.utils import shuffle
df = shuffle(df)
data = np.array(df)
X = data[:,:4]
Y = data[:,-1]
import tools as t
Xtrain,Ytrain,Xtest,Ytest = t.split_data(X,Y)

dt = DecisionTree(weaklearner='Axis-Aligned')
dt.train(Xtrain,Ytrain)
print(dt)
res = dt.test(Xtest)
num = len(Ytest[res==Ytest])
den = len(Ytest)
acc = num/den
acc