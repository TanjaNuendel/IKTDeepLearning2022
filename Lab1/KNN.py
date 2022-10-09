import numpy
# fix random seed for reproducibility
numpy.random.seed(7)

# load pima indians dataset
dataset = numpy.loadtxt("pima-indians-diabetes.data.csv", delimiter=",")
numpy.random.shuffle(dataset)
splitratio = 0.8
K_neighbors = 15
#tuple (distance, label)

# split into input (X) and output (Y) variables
#take first 8 values of each data entry (no label)
#x: data entries, y: labels 
#jeweils aufgeteilt in training und testing data 
X_train = dataset[:int(len(dataset)*splitratio),0:8]
X_val = dataset[int(len(dataset)*splitratio):,0:8]
Y_train = dataset[:int(len(dataset)*splitratio),8]
Y_val = dataset[int(len(dataset)*splitratio):,8]
#print(X_train)
#print(Y_train)
#print(len(Y_train))
#print(len(Y_val))

def distance(one,two):
    return numpy.linalg.norm(one-two)

def shortestDistance(x,x_rest,y_rest):
    #init shortest distance to distance of x to first training value 
    shortest = distance(x,x_rest[0])
    predicted = y_rest[0]
    for i in range(len(x_rest)):
        if distance(x,x_rest[i])<=shortest:
            shortest = distance(x,x_rest[i])
            #assign label of current nearest neighbor as prediction label
            predicted = y_rest[i]
    return predicted,shortest

def knn_distances(x,x_rest,y_rest):
    distances = []
    for i in range(len(x_rest)):
        distances.append((distance(x,x_rest[i]),y_rest[i]))
        #print((distance(x,x_rest[i]),y_rest[i]))
    
    distances.sort(key=lambda x: x[0])
    #count number of 0/1 (labels)
    a = 0
    for i in range(K_neighbors):
        a += distances[i][1]
    if(a<=int(K_neighbors/2)):
        return 0,distances[0][1]
    return 1,distances[0][1]

#true/false positive/negative 
TP = 0
TN = 0
FP = 0
FN = 0
for i in range(len(X_val)):
    x = X_val[i]
    y = Y_val[i]
    #pred,shortest = shortestDistance(x,X_train,Y_train)
    pred,shortest = knn_distances(x,X_train,Y_train)
    #print("Y:",pred,"Y hat:",y,"Distance:",shortest)
    #print("Y: ", )


    if(y==1 and pred ==1):
        TP += 1

    if(y==0 and pred ==0):
        TN += 1

    if(y==1 and pred ==0):
        FN += 1

    if(y==0 and pred ==1):
        FP += 1

print("Accuracy:",(TP+TN)/(TP+TN+FP+FN))
print("Recall",TP/(TP+FN))
print("Precision",TP/(TP+FP))
print("F1",(2*TP)/(2*TP+FP+FN))

#print(len(dataset))

