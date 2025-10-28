import cv2
import numpy as np

input_image_name = 'ai.png'
output_image_name = 'sari_gri_bulanik.png'


img = cv2.imread(input_image_name)
if img is None:
    print(f"HATA: '{input_image_name}' dosyasi bulunaamdi")
else:
    resized_img = cv2.resize(img, (400, 300))


x,y,w,h = 50,50,200,150

roi = resized_img[y:y+h, x:x+w]

roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

roi_gray_blur = cv2.GaussianBlur(roi_gray, (15, 15), 0)

roi_bgr_blur = cv2.cvtColor(roi_gray_blur, cv2.COLOR_GRAY2BGR)

processed = resized_img.copy()
processed[y:y+h, x:x+w] = roi_bgr_blur

yellow = (0, 255, 255)
cv2.rectangle(processed, (x, y), (x+w,y+h), yellow, 2)

cv2.imshow("islemli goruntu", processed)
cv2.waitKey(0)
cv2.destroyAllWindows()

