import numpy as np
import math
from sklearn import neighbors, datasets, metrics

n_neighbors = 12  # K in KNN classifier
path = "C:\\Users\\Ariel\\Downloads\\FaceDatabase"
faces = datasets.load_files(container_path=path, shuffle=False, encoding='utf=8')

alldata = faces.data
for x in range(len(alldata)):  # reformatting face data
    alldata[x] = alldata[x].split('\r\n')
    for y in range(len(alldata[x])):
        alldata[x][y] = alldata[x][y].split(' ')
        for z in range(len(alldata[x][y])):
            alldata[x][y][z] = float(alldata[x][y][z])
# alldata[40 samples][22 attributes][2 values]
alldata = np.array(alldata).reshape(40, 22, 2)
alltarget = faces.target


def dist(p1, p2):
    euclidean = ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 1/2
    return euclidean


exctractedData = []
for x in range(len(alldata)):
    featureData = []
    e1 = dist(alldata[x][9], alldata[x][10])
    e2 = dist(alldata[x][11], alldata[x][12])
    e3 = dist(alldata[x][0], alldata[x][1])
    e4 = dist(alldata[x][8], alldata[x][13])
    n1 = dist(alldata[x][15], alldata[x][16])
    n2 = dist(alldata[x][20], alldata[x][21])
    l1 = dist(alldata[x][2], alldata[x][3])
    l2 = dist(alldata[x][17], alldata[x][19])
    b1 = dist(alldata[x][4], alldata[x][5])
    b2 = dist(alldata[x][6], alldata[x][7])
    a1 = dist(alldata[x][10], alldata[x][19])

    featureData.append(max(e1, e2) / e4)    # eyeL
    featureData.append(e3 / e4)             # eyeD
    featureData.append(n1 / n2)             # noseR
    featureData.append(l1 / l2)             # lipS
    featureData.append(l1 / n2)             # lipL
    featureData.append(max(b1, b2) / e4)    # browL
    featureData.append(a1 / n2)             # agresR
    exctractedData.append(featureData)

print(np.array(exctractedData))
# extractedData[40 samples][7 distances]
trainData = []  		# will have all train data of the classes
trainTarget = []  		# will have all class labels of the train data

testData = []  			# will have all test data for prediction test
testClasses = []		# will have the corresponding classes
setSize = 13	   		# dictates size of training set, max set size is 20 (max count of each class)
numClasses = 2

# Class 0 data separation
for i in range(0, setSize):
    trainData.append(exctractedData[i])
    trainTarget.append(alltarget[i])
for i in range(setSize, 20):
    testData.append(exctractedData[i])
    testClasses.append(alltarget[i])

# Class 1 data separation
for i in range(20, 20 + setSize):
    trainData.append(exctractedData[i])
    trainTarget.append(alltarget[i])
for i in range(20 + setSize, 40):
    testData.append(exctractedData[i])
    testClasses.append(alltarget[i])

'''
print("all the given data used for training:")
for j in range(len(trainData)):
    print(trainData[j])
'''

'''
print("correct classes for the data used in training:")
j = 0
for x in range(numClasses):
    print(trainTarget[j:j+setSize])
    j += setSize
'''

'''
print("all the data used for prediction test:")
for z in range(len(testData)):
    print(testData[z])
'''

''''''
nn = neighbors.KNeighborsClassifier(n_neighbors)
nn.fit(trainData, trainTarget)  		# Training is done
predictions = nn.predict(testData)  	# testing
print("--------results----------")
print("correct classes for prediction test set: \n", testClasses)
print("class predictions: \n", list(predictions))
print("---------metrics---------")
print("confusion matrix: \n", metrics.confusion_matrix(testClasses, predictions))
print("accuracy score:", metrics.accuracy_score(testClasses, predictions))
print("precision score:", metrics.precision_score(testClasses, predictions))
print("recall score:", metrics.recall_score(testClasses, predictions))

''''''
print("---------stats---------")
print("knn =", n_neighbors)
print(setSize/20 * 100, "% of the data set was used for training")
correct = 0
predictions = list(predictions)			# checking for errors
for x in range(len(testClasses)):
    if predictions[x] == testClasses[x]:
        correct += 1
print((correct / len(testClasses)) * 100, "% accuracy on test set")
