import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from ERA_Repo.Assignment_10.LR_Finder import LRFinder
from torchsummary import summary
import matplotlib.pyplot as plt
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

CIFAR_classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

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

def inverse_normalize(tensor):
    inv_normalize = transforms.Normalize(
                    mean= [-m/s for m, s in zip([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])],
                std= [1/s for s in [0.2023, 0.1994, 0.2010]]
                )
    return inv_normalize(tensor)

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

def get_error_images(model, test_loader, device, img_count):
  error_images = []
  error_target = []
  error_predicted = []
  count = 0

  for data, target in test_loader:
    data, target = data.to(device), target.to(device)
    output = model(data)
    pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    for i in range (0, 500):
      if (pred[i].cpu().numpy()[0] != target[i].cpu().numpy()):
        error_images.append(data[i])
        error_target.append(target[i].cpu().numpy())
        error_predicted.append(pred[i].cpu().numpy()[0])

        count = count + 1

        if count >= img_count:
          break
    return error_images, error_target, error_predicted

def plot_error(error_images, error_target, error_predicted, row_count):
  
  figure = plt.figure(figsize=(5, 8))
  total_img = int(len(error_target))
  
  for index in range(0, total_img):
    plt.subplot(row_count, int(total_img/row_count), index+1)
    plt.axis('off')
    img = error_images[index].cpu().numpy()
    plt.imshow(np.transpose(img, (1, 2, 0)))
    v_label = "Predicted: " + CIFAR_classes[error_predicted[index].item()] + \
            "\nActual: " + CIFAR_classes[error_target[index].item()]
    plt.title(label=v_label, fontsize=8, color="blue")
  plt.show()

def gradcam_plot(model, img, target):

  target_layers = model.layer3
  input_tensor = img.unsqueeze(0)
  cam = GradCAM(model=model, target_layers=target_layers)
  targets = [ClassifierOutputTarget(target)]
  grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
  grayscale_cam = grayscale_cam[0, :]
  rgb_img = inverse_normalize(img).permute(1, 2, 0).cpu().numpy()
  visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
  model_output = cam.outputs

  return visualization, model_output

def plot_gradcam_images(model, error_images, error_predicted, error_target, row_count):
  cam_vis_list = []
  cam_out_list = []
  
  total_img = len(error_images)
  row_count = 5

  for i in range(0, total_img):
    vis, out = gradcam_plot(model, error_images[i], error_target[i])

    cam_vis_list.append(vis)
    cam_out_list.append(out)

  figure = plt.figure(figsize=(5, 8))

  for index in range(0, total_img):
      plt.subplot(row_count, int(total_img/row_count), index+1)
      plt.axis('off')
      img = cam_vis_list[index]
      plt.imshow(img)
      v_label = "Predicted: " + CIFAR_classes[error_predicted[index].item()] + \
              "\nActual: " + CIFAR_classes[error_target[index].item()]
      plt.title(label=v_label, fontsize=8, color="blue")

  plt.show()
