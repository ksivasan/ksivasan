
## I. Implementation of Otsu's algorithm

Otsu's algorithm is a non-parametric unsupervised image segmentation method that works using a discriminative criterion based on an optimal threshold selected to maximize the separability of the resulting classes. It uses zeroth and first order moments of the gray-level histogram. 

Given an image, we can count the number of pixels that correspond to a particular pixel value. Say, the image has pixel values ranging from 0 to L, we first count the number of pixels corresponding to each of these values. If $n_i$ represents the number of pixels in $i^{th}$ level, then the probability of having $i^{th}$ pixel value is $p_i =\frac{ n_i}{N}$ where N is the total no. of pixels in the image. In Otsu's algorithm, we want to find an optimal threshold value $k$ that separates the image into two classes $C_0$ and $C_1$ (say) which corresponds to pixel values in range $[1,2,\dots,k]$ and $[k+1,\dots, L]$ respectively. The optimality criterion is decided based on the value of k that maximizes the between-class variances which is defined as follows:

Calculation of Between-class variance:

$$\omega_0 = P(C_0) = \sum^k_{i=1} p_i = \omega(k)$$
$$\omega_1 = P(C_1) = \sum^L_{i=K+1} p_i = 1-\omega(k)$$
$$\mu_0 = \sum^k_{i=1}iP(i|C_0) =  \frac{\sum^k_{i=1}ip_i}{\omega_0}$$
$$\mu_1 = \sum^L_{i=k+1}iP(i|C_1) =  \frac{\sum^L_{i=k+1}ip_i}{\omega_1}$$
$$\sigma_B^2 = \omega_0\omega_1(\mu_0-\mu_1)^2$$

Since calculation of Between-class variance depends of k, we calculate for all possible values of k and find the maximum value of $\sigma_B^2$ and set the corresponding k as our optimal threshold. All pixels having pixel values less than k form class 0 and rest belong to class 1. Since this method works in 2D image, we can apply this algorithm along each channels and get a segmentation mask which is a matrix of 0 and 1 specifying each pixel of it's class. Thus, we take a logical 'AND' operator on three such segmentation outputs. Finally, we can apply this mask over original image to get our image segmentation output. Additionally, we can iterate this process by applying Otsu's algorithm over the segmented image again. This was not found useful in current set of images. 

## II. Texture-based feature extraction

Instead of applying over RGB channels, we can create features based on textures as input to Otsu's algorithm. First, the RGB image is converted to grayscale image using any of the preferred methods. Here, we rastor scan a window of $N*N$ over each pixel and set it's value as equal to the variance of the pixel values in the pixels of this window (appropriate padding is necessary). By setting three different values for N, we get three channels similar to the previous step. Following above procedure from here on, we can get image segmentation based on texture-based feature extraction. 

## III. Contour extraction algorithm

Once we have a segmentation mask, we can choose to make it better by applying a filtering step using dilation and/ erosion. We then apply our Contour extraction algorithm. In this work, a nearest neighbour based contour extraction algorithm is implemented for simplicity. The algorithm involves rastor scan of a window $N*N$ over each pixel and set the value of pixel as 1 if the value of pixel is 1 and also the sum of pixel values over the window is less than $N*N$. In all the examples here, I have used N=3 which means 8-neighbours are considered. 

## IV. Observations

* We can see the edges in Otsu's segmentation output for RGB image features are thinner than the texture based feature extracted images. Hence, the results are better for texture based Otsu's image segmentation.

* Contour extraction algorithm works fine in most cases. Otsu's RGB image based segmentation has better contour extraction than the texture based implementation. Since the edges are thick in Texture based implementation, there are two layered contours extracted which is not looking good. 

* When Otsu's algorithm was applied for more than one iteration, the resulting output was not convincing. It blacked out most of the regions. 

* Few things to consider and will be of future interest: Increasing number of classes in Otsu's algorithm, applying AND operator over result of RGB and Texture based Segmentation mask, finding better contour extraction algorithms. There are few manual choices to be made and hence, finding an automatic segmentation algorithm would be larger long term goal. 

## V. Results

The general workflow consists of getting segmentation mask for each channel using Otsu's algorithm and apply logical AND operation to get a final Segmentation mask. I chose to apply a dilation followed by erosion procedure to filter noise in mask and then, applied contour extraction algorithm. Finally, this is multiplied with image (element wise) and normalized to get the segmentation image. I have plotted the partial outputs after Otsu's segmentation and shown segmentation masks for reference. 
The filename specifies the type of operation performed: it goes as "Contour or none" + "Otsu (RGB channel based) or Texture (feature)" + Seg + " Mask or Image" + file(1,2,3).


```python
from IPython.display import Image
```


```python
Image(filename='baby.jpg',width=500, height=500)
```




![jpeg](output_6_0.jpeg)




```python
Image(filename='Otsu_Seg_Mask1.png',width=500, height=500)
```




![png](output_7_0.png)




```python
Image(filename='Otsu_Seg_Image1.png',width=500, height=500)
```




![png](output_8_0.png)




```python
Image(filename='Otsu_Contour_Seg_Mask1.png',width=500, height=500)
```




![png](output_9_0.png)




```python
Image(filename='Otsu_Contour_Seg_Image1.png',width=500, height=500)
```




![png](output_10_0.png)




```python
Image(filename='Texture_Seg_Mask1.png',width=500, height=500)
```




![png](output_11_0.png)




```python
Image(filename='Texture_Seg_Image1.png', width=500, height=500)
```




![png](output_12_0.png)




```python
Image(filename='Texture_Contour_Seg_Mask1.png',width=500, height=500)
```




![png](output_13_0.png)




```python
Image(filename='Texture_Contour_Seg_Image1.png',width=500, height=500)
```




![png](output_14_0.png)




```python
Image(filename='lighthouse.jpg',width=300, height=300)
```




![jpeg](output_15_0.jpeg)




```python
Image(filename='Otsu_Seg_Mask2.png',width=300, height=300)
```




![png](output_16_0.png)




```python
Image(filename='Otsu_Seg_Image2.png',width=300, height=300)
```




![png](output_17_0.png)




```python
Image(filename='Otsu_Contour_Seg_Mask2.png',width=300, height=300)
```




![png](output_18_0.png)




```python
Image(filename='Otsu_Contour_Seg_Image2.png',width=300, height=300)
```




![png](output_19_0.png)




```python
Image(filename='Texture_Seg_Mask2.png',width=300, height=300)
```




![png](output_20_0.png)




```python
Image(filename='Texture_Seg_Image2.png',width=300, height=300)
```




![png](output_21_0.png)




```python
Image(filename='Texture_Contour_Seg_Mask2.png',width=300, height=300)
```




![png](output_22_0.png)




```python
Image(filename='Texture_Contour_Seg_Image2.png',width=300, height=300)
```




![png](output_23_0.png)




```python
Image(filename='ski.jpg',width=300, height=300)
```




![jpeg](output_24_0.jpeg)




```python
Image(filename='Otsu_Seg_Mask3.png',width=300, height=300)
```




![png](output_25_0.png)




```python
Image(filename='Otsu_Seg_Image3.png',width=300, height=300)
```




![png](output_26_0.png)




```python
Image(filename='Otsu_Contour_Seg_Mask3.png',width=300, height=300)
```




![png](output_27_0.png)




```python
Image(filename='Otsu_Contour_Seg_Image3.png',width=300, height=300)
```




![png](output_28_0.png)




```python
Image(filename='Texture_Seg_Mask3.png',width=300, height=300)
```




![png](output_29_0.png)




```python
Image(filename='Texture_Seg_Image3.png',width=300, height=300)
```




![png](output_30_0.png)




```python
Image(filename='Texture_Contour_Seg_Mask3.png',width=300, height=300)
```




![png](output_31_0.png)




```python
Image(filename='Texture_Contour_Seg_Image3.png',width=300, height=300)
```




![png](output_32_0.png)



## VI. Source Code


```python
import cv2
import numpy as np
cv2.destroyAllWindows()

# function to convert Seg mask to Image
def Seg2Image(Seg):
    # input:  2D Seg mask
    # output: Image
    [a,b] = np.shape(Seg)
    Image = np.zeros((a,b,3), dtype=float)
    Image[:,:,0] = Seg
    Image[:,:,1] = Seg
    Image[:,:,2] = Seg
    return Image
    
# function to get Segmentation mask using Otsu's algorithm
def Otsu(Image, Otsu_Iter):
	for iter in range(0, Otsu_Iter):    
		pixel = np.zeros((256,), dtype=int)
    # count the no. of pixels per pixel value
		for pix in range(1,256):
			pixel[pix] = np.size(Image[Image==pix])
    # find probability
		pix_prob = pixel/np.sum(pixel)
		sigma_b = np.zeros((255,), dtype=int)
    # calculate between class variance for all k
		for k in range(0,255):
			w_0 = np.sum(pix_prob[0:k])
			if w_0 > 0 and w_0 < 1:
				w_1 = 1-w_0
				mu_k = np.sum(np.multiply(pix_prob[0:k], np.arange(0,k)))
				mu_total = np.sum(np.multiply(pix_prob, np.arange(0,256)))
				mu_0 = mu_k/w_0
				mu_1 = (mu_total - mu_k)/w_1
				sigma_b[k] = w_0*w_1*(mu_0-mu_1)**2
    # Set threshold as max of sigma
		threshold = np.argmax(sigma_b)
    # Set segmenation mask based on threshold
		Seg = Image < threshold -1
    # Save Seg as uint8 for plotting
		Seg = Seg.astype(np.uint8)
		Image = np.multiply(Image, Seg)
	return Seg
	
def Texture(Image, N):
  # find the window size 
	mark = int((N-1)/2)
	[a,b] = np.shape(Image)
	Seg = np.zeros(Image.shape, dtype=float)
  # pad the image
	Image = np.pad(Image, [mark, mark], 'constant')
  # obtain the patch and find it's variance. Save it as pixel value
	for xiter in range(0,a):
		for yiter in range(0,b):
			patch = Image[xiter:2*mark+xiter+1, yiter:2*mark+yiter+1]
			Seg[xiter, yiter] = np.var(patch)
	return Seg
	
def Contour_Extraction(Image, N):
  # find window size
	mark = int((N-1)/2)
	# print(np.shape(Image))
	[a,b] = np.shape(Image)
	Seg = np.zeros(Image.shape, dtype=float)
  # pad image
	Image = np.pad(Image, [mark, mark], 'constant')
  # obtain patch and find if it is valide
	for xiter in range(0,a):
		for yiter in range(0,b):
			if Image[xiter+mark, yiter+mark] !=0:
				patch = Image[xiter:2*mark+xiter+1, yiter:2*mark+yiter+1]
				if np.sum(patch)!= N*N:
					Seg[xiter,yiter] = 1
	return Seg

# helper function to save Seg as Image
def Saving_Output(Image):
    Image = 255*Image
    Image = Image.astype(np.uint8)
    return Image

# helper function to call Otsu's alg and Contour for RGB image intensity based segmentation
def Otsu_Segmentation(RGBImage, fileName, Otsu_Iter=1):
	SegR = Otsu(RGBImage[:,:,0], Otsu_Iter)
	SegG = Otsu(RGBImage[:,:,1], Otsu_Iter)
	SegB = Otsu(RGBImage[:,:,2], Otsu_Iter)
	Seg = np.multiply(SegR, SegB)
	Seg = np.multiply(Seg, SegG)
	Output = Seg2Image(Seg)
	#Output = np.repeat(Seg[:,:,np.newaxis],3, axis=2)
	cv2.imshow("Otsu_Seg_Mask",Output)
	cv2.imwrite('Otsu_Seg_Mask'+str(fileName)+'.png',Saving_Output(Output))
	Smooth = np.multiply(RGBImage,Output)/255
	cv2.imshow("Otsu_Seg_Image", Smooth)
	cv2.imwrite("Otsu_Seg_Image"+str(fileName)+".png", Saving_Output(Smooth))
	Contour = Contour_Extraction(Seg, 3)
	Contour = np.repeat(Contour[:,:,np.newaxis],3, axis=2)
	cv2.imshow("Otsu_Contour_Seg_Mask", Contour)
	cv2.imwrite("Otsu_Contour_Seg_Mask"+str(fileName)+".png",Saving_Output(Contour))
	Contour_smooth = np.multiply(RGBImage, Contour)/255
	cv2.imshow("Otsu_Contour_Seg_Image", Contour_smooth)
	cv2.imwrite("Otsu_Contour_Seg_Image"+str(fileName)+".png", Saving_Output(Contour_smooth))
	return     

# helper function to call Otsu's algorithm and Contour extraction for Texture feature based segmentation
def Texture_Segmentation(RGBImage, fileName, Otsu_Iter=1):
	Gray_Image = cv2.cvtColor(RGBImage, cv2.COLOR_BGR2GRAY)
	T_channelR = Texture(Gray_Image, 3)
	T_channelG = Texture(Gray_Image, 5)
	T_channelB = Texture(Gray_Image, 7)
	T_SegR = Otsu(T_channelR, Otsu_Iter)
	T_SegG = Otsu(T_channelG, Otsu_Iter)
	T_SegB = Otsu(T_channelB, Otsu_Iter)
	TSeg = np.multiply(T_SegR, T_SegG)
	TSeg = np.multiply(TSeg, T_SegB)
	#Toutput = np.repeat(TSeg[:,:,np.newaxis],3, axis=2)
	Toutput = Seg2Image(TSeg)
	cv2.imshow("Texture_Seg_Mask", Toutput)
	cv2.imwrite("Texture_Seg_Mask"+str(fileName)+".png",Saving_Output(Toutput))
	Tsmooth = np.multiply(RGBImage,Toutput)/255
	cv2.imshow("Texture_Seg_Image", Tsmooth)
	cv2.imwrite("Texture_Seg_Image"+str(fileName)+".png", Saving_Output(Tsmooth))
	TContour = Contour_Extraction(TSeg, 3)
	TContour = np.repeat(TContour[:,:,np.newaxis],3, axis=2)
	cv2.imshow("Texture_Contour_Seg_Mask", TContour)
	cv2.imwrite("Texture_Contour_Seg_Mask"+str(fileName)+".png",Saving_Output(TContour))
	TContour_smooth = np.multiply(RGBImage, TContour)/255
	cv2.imshow("Texture_Contour_Seg_Image", TContour_smooth)
	cv2.imwrite("Texture_Contour_Seg_Image"+str(fileName)+".png",Saving_Output(TContour_smooth))
	return

	
img1 = cv2.imread("baby.jpg")
img2 = cv2.imread("lighthouse.jpg")
img3 = cv2.imread("ski.jpg")
img4 = cv2.imread('brain.jpg')
img5 = cv2.imread('leopard.jpg')
img6 = cv2.imread('lake.jpg')

Otsu_Segmentation(img1, 1, 1)
Texture_Segmentation(img1, 1, 1)
Otsu_Segmentation(img2, 2, 1)
Texture_Segmentation(img2, 2, 1)
Otsu_Segmentation(img3, 3, 1)
Texture_Segmentation(img3, 3, 1)
Otsu_Segmentation(img4, 4, 1)
Texture_Segmentation(img4, 4, 1)
Otsu_Segmentation(img5, 5, 1)
Texture_Segmentation(img5, 5, 1)
Otsu_Segmentation(img6, 6, 1)
Texture_Segmentation(img6, 6, 1)
```

    (600, 800)
    (600, 800)
    (768, 512)
    (768, 512)
    (480, 319)
    (480, 319)
    (766, 741)
    (766, 741)
    (321, 481)
    (321, 481)
    (734, 979)
    (734, 979)
    


```python

```
