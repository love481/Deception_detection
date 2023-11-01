import torch
import torch
import torch.nn as nn
import torch.nn
from dataset_lstm import JSONDataset,collate_fn
from torch.utils.data import DataLoader
import json,random
import numpy
import matplotlib.pyplot as plt
## for sample.json with peak accuracy 79.12% on Real_life_trail_dataset
random.seed(0)
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
def create_dataset(testing_percentage=0.2):
    with open('sample.json', 'r') as f:
        data_real_life = json.load(f)
    data_real_life=list(data_real_life.values())
    random.shuffle(data_real_life)
    data_len=len(data_real_life)
    test_data=data_real_life[:int(data_len*testing_percentage)]
    train_data=data_real_life[int(data_len*testing_percentage):]
    return train_data,test_data

def running_mean(x, N):
    cumsum = numpy.cumsum(numpy.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)
def plot_data(train_loss,test_loss,accuracies):
    test_loss=running_mean(test_loss,10)
    train_loss=running_mean(train_loss,10)
    accuracy=running_mean(accuracies,10)
    fig = plt.figure(figsize=(30, 10))
    plt.subplot(1, 2, 1)
    plt.plot(list(range(1,len(train_loss)+1)), train_loss, 'g', label='Training loss')
    plt.plot(list(range(1,len(test_loss)+1)), test_loss, 'b', label='testing loss')
    plt.title('Training and testing loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.subplot(1, 2, 2)
    plt.plot(list(range(1,len(accuracy)+1)),accuracy,'g', label='Testing accuracy')
    plt.title('Testing accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('accuracy')
    plt.savefig('img.png')



class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.hidden_dim = 512
        self.layer_dim = 1
        self.emb_dim=41
        self.fc1 = nn.LSTM(self.emb_dim,self.hidden_dim,self.layer_dim)
        self.fc2 = nn.Linear(self.hidden_dim,1)
        self.activation2=nn.Sigmoid()

    def forward(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(1), self.hidden_dim,device=x.device).requires_grad_()
        c0 = torch.zeros(self.layer_dim, x.size(1), self.hidden_dim,device=x.device).requires_grad_()
        out, (_,_) = self.fc1(x, (h0.detach(), c0.detach()))
        out = self.fc2(out[:, -1, :]) 
        x=self.activation2(out)
        return x



model = Model()
model=model.to(device)
loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.0001)

# Use the DataLoader to load data in parallel
params = {'batch_size': 4,
          'shuffle': True,
          'num_workers': 6}
max_epochs = 500
train_data,test_data=create_dataset()
training_dataset = JSONDataset(train_data)
training_dataloader = DataLoader(training_dataset,collate_fn=collate_fn,**params)
testing_dataset = JSONDataset(test_data)
testing_dataloader = DataLoader(testing_dataset,collate_fn=collate_fn,**params)
# Loop over epochs
training_loss=[]
testing_loss=[]
accuracies=[]
for epoch in range(max_epochs):
    # Training
    model.train()
    total_correct = 0.0
    total_samples = 0.0
    train_bce=0.0
    test_bce=0.0
    for i,(local_batch, local_labels) in enumerate(training_dataloader):
        # Model computations
        local_batch, local_labels=local_batch.to(device), local_labels.to(device)
        y_pred = model(local_batch)
        loss = loss_fn(y_pred, local_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_bce+=loss.cpu().detach().numpy()
    train_bce=train_bce/(i+1)
    model.eval()
    # test
    with torch.set_grad_enabled(False):
        for i,(local_batch, local_labels) in enumerate(testing_dataloader):
            local_batch, local_labels=local_batch.to(device), local_labels.to(device)
            y_pred = model(local_batch)
            loss = loss_fn(y_pred, local_labels)
            test_bce+=loss.cpu().detach().numpy()
            total_correct += ((y_pred>0.5) == local_labels).sum().item()
            total_samples += local_labels.size(0)
        accuracy = 100 * total_correct / total_samples
        test_bce=test_bce/(i+1)
    print(f'Epoch {epoch+1}: Accuracy = {accuracy:.2f}%')
    print("Epoch %d: train bce %.4f, test bce %.4f" % (epoch+1, train_bce, test_bce))
    training_loss.append(train_bce)
    testing_loss.append(test_bce)
    accuracies.append(accuracy)
    if epoch>10 and epoch%21==0:
        plot_data(training_loss,testing_loss,accuracies)
