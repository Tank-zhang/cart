# -*- coding:utf-8 -*-
from numpy import *
def loadDataSet(filename):
    dataMat = []
    fr = open(filename)
    for line in fr.readlines():
        curLine = line.strip('\n').split('\t')
        fltLine = map(float, curLine)
        dataMat.append(fltLine)
    fr.close()
    return dataMat
def regLeaf(dataSet):
    return mean(dataSet[:,-1])
def regErr(dataSet):
    a = var(dataSet[:,-1])*shape(dataSet)[0]
    d = dataSet[:,-1]
    b =  var(dataSet[:,-1])    #计算方差
    c = shape(dataSet)[0]
    return var(dataSet[:,-1])*shape(dataSet)[0]
def regTreeEval(model, inDat):
    return float(model)
def linearSolve(dataSet):
    m,n=shape(dataSet)
    X = mat(ones((m,n)))
    Y = mat(ones((m,1)))
    X[:,1:n]=dataSet[:,0:n-1]
    Y=dataSet[:,-1]
    xTx = X.T*X
    if linalg.det(xTx)==0.0:
        raise NameError('This matrix is singular, cannot do inverse, \
               try increasing the second value of ops')
    ws = xTx.T*(X.T*Y)
    return ws, X, Y
def modelLeaf(dataSet):
    ws, X, Y = linearSolve(dataSet)
    return ws
def modelErr(dataSet):
    ws,X,Y = linearSolve(dataSet)
    yHat = X*ws
    return sum(power(Y-yHat,2))
def modelTreeEval(model, inDat):
    n=shape(inDat)[1]
    X = mat(ones((1,n+1)))
    X[:,1:n+1]=inDat
    return float(X*model)
def chooseBestSplit(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):
    tolS = ops[0]
    tolN = ops[1]
    if len(set(dataSet[:,-1].T.tolist()[0])) == 1:
        return None, leafType(dataSet)
    m,n=shape(dataSet)
    S = errType(dataSet)
    bestS = inf
    bestIndex = 0
    bestValue = 0
    a = set(dataSet[:, 0].flatten().A[0])
    for featIndex in range(n-1):  #寻找方差最小的分裂点和分裂值
        for splitVal in set(dataSet[:,featIndex].flatten().A[0]):
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)   #寻找最佳分裂点  mt0为大于splitVal
            count = shape(mat0)[0]
            if(shape(mat0)[0]<tolN) or (shape(mat1)[0]<tolN):
                continue
            newS = errType(mat0)+errType(mat1)    #计算两个矩阵的方差
            if newS < bestS:
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
    if (S-bestS)<tolS:
        return None, leafType(dataSet)
    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)   #按最佳分裂点和分裂值分割数据
    if(shape(mat0)[0]<tolN) or (shape(mat1)[0]<tolN):
        print "Not enough nums"
        return None, leafType(dataSet)
    return bestIndex, bestValue
def binSplitDataSet(dataSet, feature, value):
    a = nonzero(dataSet[:, feature] > value) #返回大于value的下表
    b = nonzero(dataSet[:, feature]>value)[0]
    mat0 = dataSet[nonzero(dataSet[:, feature]>value)[0],:]
    mat1 = dataSet[nonzero(dataSet[:, feature]<=value)[0],:]
    return mat0, mat1
def createTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)
    if feat == None:
        return val
    retTree={}
    retTree['spInd'] = feat
    retTree['spVal'] = val
    lSet, rSet = binSplitDataSet(dataSet, feat, val)
    retTree['left']=createTree(lSet, leafType, errType, ops)
    retTree['right']=createTree(rSet, leafType, errType, ops)
    return retTree
def isTree(obj):
    return (type(obj).__name__=='dict')
def getMean(tree):
    if isTree(tree['right']):
        tree['right'] = getMean(tree['right'])
    if isTree(tree['left']):
        tree['left'] = getMean(tree['left'])
    return (tree['left']+tree['right'])/2.0
def prune(tree, testData):
    if shape(testData)[0] == 0:
        return getMean(tree)
    if(isTree(tree['right']) or isTree(tree['left'])):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'],tree['spVal'])
    if isTree(tree['left']):
        tree['left']=prune(tree['left'],lSet)
    if isTree(tree['right']):
        tree['right']=prune(tree['right'],rSet)
    if not isTree(tree['right']) and not isTree(tree['left']):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
        errorNoMerge = sum(power(lSet[:,-1]-tree['left'],2))+\
                       sum(power(rSet[:,-1]-tree['right'],2))
        treeMean = (tree['left']+tree['right'])/2.0
        errorMerge = sum(power(testData[:,-1]-treeMean,2))
        if errorMerge < errorNoMerge:
            print "Merging"
            return treeMean
        else:
            return tree
    else:
        return tree
def treeForeCast(tree, inData, modelEval=regTreeEval):
    if not isTree(tree):
        return modelEval(tree, inData)
    if inData[tree['spInd']]>tree['spVal']:
        if isTree(tree['left']):
            return treeForeCast(tree['left'], inData, modelEval)
        else:
            return modelEval(tree['left'],inData)
    else:
        if isTree(tree['right']):
            return treeForeCast(tree['right'], inData, modelEval)
        else:
            return modelEval(tree['right'], inData)
def createForeCast(tree, testData, modelEval=regTreeEval):
    m=len(testData)
    yHat = mat(zeros((m,1)))
    for i in range(m):
        yHat[i,0]=treeForeCast(tree, mat(testData[i]), modelEval)
    return yHat
'''
myData2 = loadDataSet(r"ex2.txt")
myMat2 = mat(myData2)
tree2 = createTree(myMat2, ops=(0,1))
print tree2
myData2Test = loadDataSet(r"ex2test.txt")
myMat2Test = mat(myData2Test)
print prune(tree2, myMat2Test)
'''
trainMat = mat(loadDataSet('bikeSpeedVsIq_train.txt'))
testMat = mat(loadDataSet('bikeSpeedVsIq_test.txt'))
myregTree=createTree(trainMat, ops=(1,20))
mymodTree=createTree(trainMat, modelLeaf, modelErr, (1,20))
yregHat=createForeCast(myregTree, testMat[:,0])
print yregHat
ymodHat=createForeCast(mymodTree, testMat[:,0], modelTreeEval)
regCo = corrcoef(yregHat, testMat[:,1], rowvar=0)[0,1]
modCo = corrcoef(ymodHat, testMat[:,1], rowvar=0)[0,1]
print "reg", regCo
print "model", modCo
