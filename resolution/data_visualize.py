import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np

image =sitk.ReadImage("train/normal dose/14717961.mhd")
image = sitk.GetArrayFromImage(image)
image = np.squeeze(image[150, ...])  # if the image is 3d, the slice is integer
plt.imshow(image, cmap='gray')
plt.axis('off')
plt.show()