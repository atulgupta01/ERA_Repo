import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from ERA_Repo.Assignment_10.LR_Finder import LRFinder
from torchsummary import summary
import matplotlib.pyplot as plt

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

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

def get_model_summary(model, device, input_size):
  model = model().to(device)
  summary(model, input_size= input_size)

  return model

def accuracy_plot(train_losses, test_losses, train_acc, test_acc):
  fig, axs = plt.subplots(2,2,figsize=(15,10))
  axs[0, 0].plot(train_losses)
  axs[0, 0].set_title("Training Loss")
  axs[1, 0].plot(train_acc)
  axs[1, 0].set_title("Training Accuracy")
  axs[0, 1].plot(test_losses)
  axs[0, 1].set_title("Test Loss")
  axs[1, 1].plot(test_acc)
  axs[1, 1].set_title("Test Accuracy")

