import torchvision
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from gradcam import visualize
import cv2 as cv
import torch
from Apatch_pretrained_models import pretrainedmodels
from Inpainting import inpainting
# %matplotlib inline

# Utilities
def imshow(img):
    # img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def imsave(img,addr='./bar.png'):
    # img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    # plt.show()
    plt.savefig(name)

def normalize_0_1(inp):
  return (inp-inp.min())/(inp.max()-inp.min())




#Variables
netClassifier = 'alexnet'

print("=> creating model ")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
netClassifier = pretrainedmodels.__dict__[netClassifier](num_classes=1000, pretrained='imagenet')
netClassifier.to(device)





# Main
if __name__ == "__main__":
  patch_samples_dir = './Images/'
  #2998,
  orig_name = '2992_563_original.png' #U ONLY HAVE TO CHANGE THIS
  a1,b1,_ = orig_name.split('_')
  class_orig = int(b1)
  class_adv = 859

  adv_name = a1 + '_' +  str(class_adv) + '_adversarial.png'
  mask_name = a1  + '_mask.png'

  # Loading data
  img_orig = Image.open(patch_samples_dir+orig_name)
  img_adv = Image.open(patch_samples_dir+adv_name)
  img_mask = Image.open(patch_samples_dir+mask_name)

  # Preprocessing
  x_orig = np.array(img_orig)/255.
  x_adv = np.array(img_adv)/255.
  x_mask = np.array(img_mask)/255.

  grad_cam_input_orig = torch.tensor(x_orig.transpose(2,0,1), dtype=torch.float, device='cuda:0').unsqueeze(0)
  grad_cam_input_adv = torch.tensor(x_adv.transpose(2,0,1), dtype=torch.float, device='cuda:0').unsqueeze(0)

  # Calculating the gradcam visualizations for both real image and the one with adversarial patch
  cam_orig = visualize(grad_cam_input_orig, class_orig, netClassifier, save_img=True, file_name='./original_gradcam.png')
  cam_adv = visualize(grad_cam_input_adv, class_adv, netClassifier, save_img=True, file_name='./original_gradcam.png')

  cam_orig_alone = np.tile(np.expand_dims(cam_orig,0),[3,1,1])

  cam_x_image_orig = x_orig.transpose(2,0,1)*(np.tile(np.expand_dims(cam_orig,0),[3,1,1]))
  cam_x_image_orig = normalize_0_1(cam_x_image_orig)

  gradient_orig = cv.Laplacian(cam_x_image_orig,cv.CV_64F)
  gradient_orig = normalize_0_1(gradient_orig)
  blur_orig = cv.GaussianBlur(gradient_orig,(5,5),0)

  myblur_orig = np.zeros_like(blur_orig)
  myblur_orig[blur_orig>=blur_orig.mean()+(blur_orig.mean()-blur_orig.min())/3.9] = 1.

  temp_visual1 = torch.tensor([x_orig.transpose(2,0,1),cam_orig_alone,cam_x_image_orig,gradient_orig,blur_orig,myblur_orig])
  # imshow(torchvision.utils.make_grid(temp_visual1))  #for  further visualizations, uncomment this


  #adversarial image
  cam_adv_alone = np.tile(np.expand_dims(cam_adv,0),[3,1,1])

  cam_x_image_adv = x_adv.transpose(2,0,1)*(np.tile(np.expand_dims(cam_adv,0),[3,1,1]))
  cam_x_image_adv = normalize_0_1(cam_x_image_adv)

  gradient_adv = cv.Laplacian(cam_x_image_adv,cv.CV_64F)
  gradient_adv = normalize_0_1(gradient_adv)

  # Adding GaussianBlur for better localization
  blur_adv = cv.GaussianBlur(gradient_adv,(5,5),0)
  my_mask = np.zeros_like(blur_adv)
  # This is a made up threshold, the 3.9 is a hyperparameter and seems to work well.
  # The following line is masking everything else than the most import part of the image in the eyes of the GRADCAM
  my_mask[blur_adv>=blur_adv.mean()+(blur_adv.mean()-blur_adv.min())/3.9] = 1.

  temp_visual = torch.tensor([x_adv.transpose(2,0,1),cam_adv_alone,cam_x_image_adv,gradient_adv,blur_adv,my_mask])
  # imshow(torchvision.utils.make_grid(temp_visual))  #for  further visualizations, uncomment this

  # imshow(torchvision.utils.make_grid(torch.tensor(my_mask)))  #for  further visualizations, uncomment this






  # As the first attempt to localize the adversarial patch, we used Convex Hull. 
  # Even though accurate, if there is even a single outlier, the adv. patch would be localized poorly
  from scipy.spatial import ConvexHull, convex_hull_plot_2d
  temp = (my_mask*blur_adv).sum(0).clip(0.,1.)
  # print(np.count_nonzero(temp))
  convex_input = np.zeros([np.count_nonzero(temp),2])
  k=0
  for i in range(temp.shape[0]):
    for j in range(temp.shape[1]):
      if temp[i,j]>0.:
        # pass
        convex_input[k][0] = i
        convex_input[k][1] = j
        k+=1

  # print(convex_input.shape,np.count_nonzero(convex_input))
  hull = ConvexHull(convex_input,qhull_options='QG4')
  plt.plot(convex_input[:,0], convex_input[:,1], 'o')
  for simplex in hull.simplices:
    # print(simplex)
    plt.plot(convex_input[simplex, 0], convex_input[simplex, 1], 'k-')
  # plt.show() #for  further visualizations, uncomment this
  xs = convex_input[hull.vertices,0]
  ys = convex_input[hull.vertices,1]
  plt.plot(convex_input[hull.vertices,0], convex_input[hull.vertices,1], 'r--', lw=2)
  plt.plot(convex_input[hull.vertices[0],0], convex_input[hull.vertices[0],1], 'ro')
  # plt.show() #for  further visualizations, uncomment this
  fig = plt.figure()
  ax = fig.add_subplot(1,1,1)
  for visible_facet in hull.simplices[hull.good]:
    ax.plot(hull.points[visible_facet, 0],
            hull.points[visible_facet, 1],
            color='violet',lw=6)
  convex_hull_plot_2d(hull, ax=ax)
  # plt.show() #for  further visualizations, uncomment this
  plt.savefig('./Results/Convex_Hull_%d.png'%a1)





  # Since we know our patches are square shaped, Convex Hull accuracy is useless (since it would only complicate things)
  # Hence, we used 2 greedy approached:
  # 1. a) Calculate the min and max along X and Y where there is any active region in the detected mask (my_mask)
  #    b) Then activate the whole region inside the 4 points of part (a) and deactivate the rest
  #
  # 2. a) In each column and row, first check if there are plenty of active points in the detected mask (my_mask)
  #       Calculate the min and max along X and Y in the region that have plenty of active points
  #    b) The same as part 1
  #
  # We could also do this using Mathematical Morphology Algorithms. Since our greedy algorithm (2) seems
  # to work well, we didn't use those algorithms

  threshold = 4
  xs_min = np.Infinity
  xs_max = -np.Infinity
  ys_min = np.Infinity
  ys_max = -np.Infinity
  temp = np.sum(my_mask,0)
  a,b = temp.shape
  for x in range(a):
    if np.count_nonzero(temp[x,:]) > threshold:
      if x < xs_min:
        xs_min = x
      if x > xs_max:
        xs_max = x

  for y in range(b):
    if np.count_nonzero(temp[:,y]) > threshold:
      if y < ys_min:
        ys_min = y
      if y > ys_max:
        ys_max = y

  
  # Patch_mask performs greedy algorithm (1) 
  # Patch_mask1 performs greedy algorithm (2)
  patch_mask = np.zeros_like(my_mask)
  patch_mask1 = np.zeros_like(my_mask)
  patch_mask [:,int(xs.min()):int(xs.max()),int(ys.min()):int(ys.max())] = 1.
  patch_mask1 [:,int(xs_min):int(xs_max),int(ys_min):int(ys_max)] = 1.

  # Saliency Maps visualized upon the image
  attention_map_patch_mask = x_adv.transpose(2,0,1) * patch_mask
  attention_map_patch_mask1 = x_adv.transpose(2,0,1) * patch_mask1

  # Here we visualize the calculated masks on the Image and Alone (First on image the in comparision with true mask)
  # Left: True Image/Mask, Middle: (2) Greedy Algorithm, Right: (1) Greedy Algorithm

  temp_visual = torch.tensor([x_adv.transpose(2,0,1),nwot,nwot1])
  # imshow(torchvision.utils.make_grid(temp_visual,padding=4,pad_value=0.8)) # for  further visualizations, uncomment this
  imsave(torchvision.utils.make_grid(temp_visual,padding=4,pad_value=0.8),'./Results/bar_image_%s.png'%a1)

  temp_visual = torch.tensor([x_mask.transpose(2,0,1),patch_mask1,patch_mask])
  # imshow(torchvision.utils.make_grid(temp_visual,padding=4,pad_value=0.9)) # for  further visualizations, uncomment this
  imsave(torchvision.utils.make_grid(temp_visual,padding=4,pad_value=0.8),'./Results/bar_mask_%s.png'%a1)
 

  # To further show the difference between, we show the 4 points of the square calculated by our
  # greedy algorithm (2) and that of the true mask
  temp = np.sum(x_mask,-1)

  xs_min_true = np.Infinity
  xs_max_true = -np.Infinity
  ys_min_true = np.Infinity
  ys_max_true = -np.Infinity
  for x in range(a):
    if np.count_nonzero(temp[x,:]) :
      if x < xs_min_true:
        xs_min_true = x
      if x > xs_max_true:
        xs_max_true = x

  for y in range(b):
    if np.count_nonzero(temp[:,y]) :
      if y < ys_min_true:
        ys_min_true = y
      if y > ys_max_true:
        ys_max_true = y

  print("For Image %s, coordination of the four points of the square of the True Mask:\n (%d,%d) , (%d,%d) , (%d,%d) , (%d,%d)"%(a1,xs_min_true,ys_min_true , xs_max_true,ys_min_true,
                                                                                                xs_min_true,ys_max_true , xs_max_true,ys_max_true))
                                      
  print("For Image %s, coordination of the four points of the square of the Calculated Mask:\n (%d,%d) , (%d,%d) , (%d,%d) , (%d,%d)"%(a1,xs_min,ys_min , xs_max,ys_min,
                                                                                                xs_min,ys_max , xs_max,ys_max))

  #the output_image is the resulting image with its adv. patch removed.
  output_image = inpainting(img_adv, patch_mask1)
  imageio.imwrite('./output.png', upcast(output_image.permute(1, 2, 0).detach().cpu().numpy()))

  
  # For Image 2992, coordination of the four points of the square of the True Mask:
  # (74,148) , (139,148) , (74,213) , (139,213)
  # For Image 2992, coordination of the four points of the square of the Calculated Mask:
  # (72,148) , (140,148) , (72,213) , (140,213)
  
  # For Image 2968, coordination of the four points of the square of the True Mask:
  # (90,69) , (155,69) , (90,134) , (155,134)
  # For Image 2968, coordination of the four points of the square of the Calculated Mask:
  # (90,69) , (155,69) , (90,134) , (155,134)

  # For Image 2954, coordination of the four points of the square of the True Mask:
  # (139,145) , (204,145) , (139,210) , (204,210)
  # For Image 2954, coordination of the four points of the square of the Calculated Mask:
  # (138,145) , (206,145) , (138,210) , (206,210)

  # On average, the masks were mostly fit or wider than the ground truth which is absolutely desirable
  # since using Image Inpainting can reconstruct image parts. However, it cannot alter anything
  # outsie the provided mask. So if the mask doesn't cover the whole Adversarial Patch, the Patch
  # would not be removed completely.