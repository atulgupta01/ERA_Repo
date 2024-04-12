import torchvision
import torch
import torch.optim as optim
from tqdm import tqdm
from ERA_Repo.util import get_learning_rate
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR


train_losses = []
test_losses = []
train_acc = []
test_acc = []
lrs = []

def train(model, device, train_loader, optimizer, scheduler, criterion):
  model.train()
  pbar = tqdm(train_loader)
  correct = 0
  processed = 0
  for batch_idx, (data, target) in enumerate(pbar):
    # get samples
    data, target = data.to(device), target.to(device)

    # Init
    optimizer.zero_grad()
    # In PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes.
    # Because of this, when you start your training loop, ideally you should zero out the gradients so that you do the parameter update correctly.

    # Predict
    y_pred = model(data)

    # Calculate loss
    loss = criterion(y_pred, target)
    train_losses.append(loss)
    lrs.append(get_learning_rate(optimizer))

    # Backpropagation
    loss.backward()
    optimizer.step()
    scheduler.step()

    # Update pbar-tqdm

    pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    correct += pred.eq(target.view_as(pred)).sum().item()
    processed += len(data)

    pbar.set_description(desc= f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f} LR={get_learning_rate(optimizer)}')
    train_acc.append(100*correct/processed)

def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    test_acc.append(100. * correct / len(test_loader.dataset))

def train_val_seq(model, device, train_loader, test_loader, optimizer, scheduler, criterion, EPOCHS):
    for epoch in range(EPOCHS):
        print("EPOCH:", epoch)
        train(model, device, train_loader, optimizer, scheduler, criterion)
        test(model, device, test_loader, criterion)
    
    return train_losses, test_losses, train_acc, test_acc, lrs

class Cifar10Dataset(torchvision.datasets.CIFAR10):
    def __init__(self, root="~/data/cifar10", train=True, download=True, transform=None):
        super().__init__(root=root, train=train, download=download, transform=transform)

    def __getitem__(self, index):
        image, label = self.data[index], self.targets[index]

        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]

        return image, label

class args():

  def __init__(self,device = 'cpu' ,use_cuda = False) -> None:
    self.batch_size = 512
    self.device = device
    self.use_cuda = use_cuda
    self.kwargs = {'num_workers': 1, 'pin_memory': True} if self.use_cuda else {}

def get_cifar_data(train_transforms, test_transforms):
    trainset = Cifar10Dataset(root='./data', train=True, download=True, transform=train_transforms)
    testset = Cifar10Dataset(root='./data', train=False, download=True, transform=test_transforms)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args().batch_size,
                                          shuffle=True, **args().kwargs)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args().batch_size,
                                          shuffle=True, **args().kwargs)
    
    return train_loader, test_loader

def get_optimizer(v_name, model, lr, decay):
  if v_name == "ADAM":
    optimizer = optim.Adam(model.parameters(), lr = lr, weight_decay = decay)

return optimizer

def get_scheduler(v_name, MAX_LR, steps_per_epoch, anneal_strategy = 'linear', v_epochs = 20):
  if v_name == "OneCycle":
    scheduler = OneCycleLR(optimizer, 
                           max_lr=MAX_LR, 
                           steps_per_epoch= steps_per_epoch,
                           anneal_strategy = anneal_strategy, 
                           epochs=get_epochs(v_epochs), 
                           pct_start=5/get_epochs(v_epochs),
                           div_factor=100, 
                           three_phase=False, 
                           final_div_factor=100)

  return scheduler

def get_epochs(v_epochs = 20):
  return v_epochs
  
