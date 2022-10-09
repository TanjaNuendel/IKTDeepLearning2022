
# Import the modules needed


from numpy import vstack
from pandas import read_csv
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, average_precision_score
from sklearn.metrics import confusion_matrix, recall_score, f1_score
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torch
from torch import Tensor
from torch.nn import Linear
from torch.nn import ReLU
from torch.nn import Sigmoid
from torch.nn import Module
from torch.optim import SGD
from torch.nn import BCELoss
from torch.optim import lr_scheduler
from torch.nn.init import kaiming_uniform_
from torch.nn.init import xavier_uniform_
import time
import math

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Create the custom data loader
# Create a custom CSVDataset loader

class ecoliDataset(Dataset):
    #Constructor for initially loading
    def __init__(self,path):
        df = read_csv(path, header=None)
        # Store the inputs and outputs
        self.X = df.values[:, :-1]
        self.y = df.values[:, -1] #Assuming your outcome variable is in the first column
        self.X = self.X.astype('float32')
        # Label encode the target as values 1 and 0 or sick and not sick
        self.y = LabelEncoder().fit_transform(self.y)
        self.y = self.y.astype('float32')
        self.y = self.y.reshape((len(self.y), 1))

    # Get the number of rows in the dataset
    def __len__(self):
        return len(self.X)
    # Get a row at an index
    def __getitem__(self,idx):
        return [self.X[idx], self.y[idx]]

    # Create custom class method - instead of dunder methods
    def split_data(self, split_ratio=0.2):
        test_size = round(split_ratio * len(self.X))
        train_size = len(self.X) - test_size
        return random_split(self, [train_size, test_size])

# Create the Class for training
# Create model
class ecoliModel(Module):
    def __init__(self, n_inputs):
        super(ecoliModel, self).__init__()
        # First hidden layer
        self.hidden1 = Linear(n_inputs, 20)
        kaiming_uniform_(self.hidden1.weight, nonlinearity='relu')
        self.act1 = ReLU()
        # Second hidden layer
        self.hidden2 = Linear(20, 10)
        kaiming_uniform_(self.hidden2.weight, nonlinearity='relu')
        self.act2 = ReLU()
        # Third hidden layer
        self.hidden3 = Linear(10,1)
        xavier_uniform_(self.hidden3.weight)
        self.act3 = Sigmoid()

    def forward(self, X):
        #Input to the first hidden layer
        X = self.hidden1(X)
        X = self.act1(X)
        # Second hidden layer
        X = self.hidden2(X)
        X = self.act2(X)
        # Third hidden layer
        X = self.hidden3(X)
        X = self.act3(X)
        return X

# Create the train model function
# Create training loop based off our custom class
def train_model(train_dl, model, epochs=100, lr=0.01, momentum=0.9, save_path='evoli_best_model.pth'):
    # Define your optimisation function for reducing loss when weights are calculated 
    # and propogated through the network
    start = time.time()
    criterion = BCELoss()
    optimizer = SGD(model.parameters(), lr=lr, momentum=momentum)
    loss = 0.0

    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch+1, epochs))
        print('-' * 10)
        model.train()
        # Iterate through training data loader
        for i, (inputs, targets) in enumerate(train_dl):
            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs.data,1) #Get the class labels
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        torch.save(model, save_path)
    time_delta = time.time() - start
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_delta // 60, time_delta % 60
    ))
    
    return model

# Create the evaluation function

def evaluate_model(test_dl, model, beta=1.0):
    preds = []
    actuals = []

    for (i, (inputs, targets)) in enumerate(test_dl):
        #Evaluate the model on the test set
        yhat = model(inputs)
        #Retrieve a numpy weights array
        yhat = yhat.detach().numpy()
        # Extract the weights using detach to get the numerical values in an ndarray, instead of tensor
        actual = targets.numpy()
        actual = actual.reshape((len(actual), 1))
        # Round to get the class value i.e. sick vs not sick
        yhat = yhat.round()
        # Store the predictions in the empty lists initialised at the start of the class
        preds.append(yhat)
        actuals.append(actual)
    
    # Stack the predictions and actual arrays vertically
    preds, actuals = vstack(preds), vstack(actuals)
    #Calculate metrics
    cm = confusion_matrix(actuals, preds)
    # Get descriptions of tp, tn, fp, fn
    tn, fp, fn, tp = cm.ravel()
    total = sum(cm.ravel())
    
    metrics = {
        'accuracy': accuracy_score(actuals, preds),
        'AU_ROC': roc_auc_score(actuals, preds),
        'f1_score': f1_score(actuals, preds),
        'average_precision_score': average_precision_score(actuals, preds),
        'f_beta': ((1+beta**2) * precision_score(actuals, preds) * recall_score(actuals, preds)) / (beta**2 * precision_score(actuals, preds) + recall_score(actuals, preds)),
        'matthews_correlation_coefficient': (tp*tn - fp*fn) / math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)),
        'precision': precision_score(actuals, preds),
        'recall': recall_score(actuals, preds),
        'true_positive_rate_TPR':recall_score(actuals, preds),
        'false_positive_rate_FPR':fp / (fp + tn) ,
        'false_discovery_rate': fp / (fp +tp),
        'false_negative_rate': fn / (fn + tp) ,
        'negative_predictive_value': tn / (tn+fn),
        'misclassification_error_rate': (fp+fn)/total ,
        'sensitivity': tp / (tp + fn),
        'specificity': tn / (tn + fp),
        #'confusion_matrix': confusion_matrix(actuals, preds), 
        'TP': tp,
        'FP': fp, 
        'FN': fn, 
        'TN': tn
    }
    return metrics, preds, actuals
        
# Create prediction routine
def predict(row, model):
    row = Tensor([row])
    yhat = model(row)
    # Get numpy array
    yhat = yhat.detach().numpy()
    return yhat


# Create the prepare thyroid dataset function
def prepare_dataset(path):
    dataset = ecoliDataset(path)
    train, test = dataset.split_data(split_ratio=0.3)
    # Prepare data loaders
    train_dl = DataLoader(train, batch_size=32, shuffle=True)
    test_dl = DataLoader(test, batch_size=1024, shuffle=False)
    return train_dl, test_dl



# Use the model
if __name__=='__main__':
    print('[EXPERIMENT] Thyroid Binary classification\n', '-'*80)
    #---------------Thyroid Example 1 ---------------------
    # Get the data for Thyroid example
    #train_dl, test_dl = prepare_thyroid_dataset('https://raw.githubusercontent.com/StatsGary/Data/main/thyroid_raw.csv')
    train_dl, test_dl = prepare_dataset('Lab2\.data\shuffled_ecoli2.csv')
    print(f"The length of the training set is: {len(train_dl.dataset)}")
    print(f"The length of the testing set is {len(test_dl.dataset)}\n", '-'*80)
    # Train the model
    # Specify the number of input dimensions
    print('Starting model training...\n', '-'*80)
    model = ecoliModel(7)
    # Train the model
    train_model(train_dl, model, 
                save_path='eco_model.pth',
                epochs=100, 
                lr=0.01)
    # Evaluate the model
    print('Starting model evaluation...\n', '-'*80)
    results = evaluate_model(test_dl, model, beta=1)
    model_metrics = results[0]
    metrics_df = pd.DataFrame.from_dict(model_metrics, orient='index', columns=['metric'])
    metrics_df.index.name = 'metric_type'
    metrics_df.reset_index(inplace=True)
    metrics_df.to_csv('confusion_matrix_thyroid.csv', index=False)
    print(metrics_df)

    """# Make prediction against model
    print('Running prediction against model...\n', '-'*80,)
    # Make prediction against model
    row = [0.83,0.37,0.48,0.50,0.61,0.71,0.74]
    #row = [1,0,1,-0.18829,0.93035,-0.36156,-0.10868,-0.93597,1,-0.04549,0.50874,-0.67743,0.34432,-0.69707,-0.51685,-0.97515,0.05499,-0.62237,0.33109,-1,-0.13151,-0.45300,-0.18056,-0.35734,-0.20332,-0.26569,-0.20468,-0.18401,-0.19040,-0.11593,-0.16626,-0.06288,-0.13738,-0.02447]
    yhat = predict(row, model)
    print('Predicted: %.3f (class=%d)' % (yhat, yhat.round()))
"""


