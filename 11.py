import cv2


img = cv2.imread('lacoste.png')

#yatay çevir
flip_horizontal = cv2.flip(img, 1)

#dikey çevir
flip_vertical = cv2.flip(img, 0)

#hem yatay hem dikey çevir
flip_both = cv2.flip(img, -1)


cv2.imshow('Orijinal', img)
cv2.imshow('Yatay Flip', flip_horizontal)
cv2.imshow('Dikey Flip', flip_vertical)
cv2.imshow('Yatay+Dikey Flip', flip_both)
cv2.waitKey(0)
cv2.destroyAllWindows()
