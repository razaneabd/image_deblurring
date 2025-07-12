from blur_detector import detect_blur
from models.SRCNN import SRCNN
from torchvision.transforms import transforms
import matplotlib.pyplot as plt
import torch.optim as optim
import torch
import numpy as np
import argparse
import imutils
import cv2


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, required=True)
args = vars(ap.parse_args())

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(device)
model = SRCNN().to(device)
model.load_state_dict(torch.load('./outputs/model.pth'))
model.eval()
print(model)
transform = transforms.Compose([
                                transforms.ToPILImage(), 
                                transforms.Resize((224, 224)), 
                                transforms.ToTensor()
                                ])

orig = cv2.imread(args["image"])
orig = imutils.resize(orig, width = 500)
#Converts it to grayscale(for blur detection)
gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)

#Uses the detect_blur function to decide if img is blurry.
(mean, blurry) = detect_blur(gray, size = 60, thresh = 20)
image = np.dstack([gray] * 3)


if blurry:
    color = (0, 0, 255)
    text = "Blurry ({:.4f})"
    text = text.format(mean)
    cv2.putText(image, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    print("[INFO] {}".format(text))
    cv2.imshow("Output", image)
    cv2.waitKey(0)

    orig = cv2.imread(args["image"])
    orig = transform(orig)
    blur_image = orig.unsqueeze(0)#adds a batch dimension.

    with torch.no_grad():
        sharp = model(blur_image.to(device))
    sharp = sharp.squeeze(0).cpu()
    #Predicts the deblurred img
    sharp_image = np.array(sharp.permute(1, 2, 0))
    orig = np.array(sharp.permute(1, 2, 0))
    #Rearranges tensor dimensions to image format (H, W, C).
    vis = np.concatenate((orig, sharp_image), axis=1)
    cv2.putText(image, "Deblurred Image", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.imshow("Output:Deblurred Image", sharp_image)
    cv2.waitKey(0)
    print("Deblurred Image")   
else:
    color = (0, 255, 0)
    text = "Not Blurry ({:.4f})"
    text = text.format(mean)
    cv2.putText(image, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    print("[INFO] {}".format(text))
    cv2.imshow("Output", image)
    cv2.waitKey(0)
