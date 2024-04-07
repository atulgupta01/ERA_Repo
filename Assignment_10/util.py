import cv2
import torchvision
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from ERA_Repo.Assignment_10.LR_Finder import LRFinder
import torch.optim as optim
from tqdm import tqdm

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


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

train_transforms = A.Compose([
    
    A.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
    A.PadIfNeeded(min_height=40, min_width=40, always_apply=True),
    A.HorizontalFlip(p=0.5),
    A.RandomCrop(height=32, width=32, always_apply=True),
    A.CoarseDropout(max_holes = 1, max_height=8, max_width=8, min_holes = 1, min_height=8, min_width=8,
                    fill_value=([0.4914, 0.4822, 0.4465]), mask_fill_value = None),
    ToTensorV2(),
])

test_transforms = A.Compose([
    A.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
    ToTensorV2(),
])

def get_learning_rate(optimizer):
    
    for param in optimizer.param_groups:
        return param['lr']
        
def lr_range_checker(model, train_loader, optimizer, criterion):
  lr_finder = LRFinder(model, optimizer, criterion, device="cuda")
  lr_finder.range_test(train_loader, end_lr=10, num_iter=200, step_mode="exp")
  lr_finder.plot() # to inspect the loss-learning rate graph
  lr_finder.reset() # to reset the model and optimizer to their initial state

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
    
  return train_losses, train_acc, lrs

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
    
    return test_losses, test_acc
