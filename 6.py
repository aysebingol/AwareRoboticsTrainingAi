import cv2

# Yaprak resmini oku
input_image_name = 'yaprak.png'
img = cv2.imread(input_image_name)

if img is None:
    print(f"HATA: '{input_image_name}' dosyası bulunamadı.")
else:
    # Gri tonlamaya çevir
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # thresholding 
    _, thresh = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Sonucu göster kaydet
    cv2.imshow('Orijinal Görüntü', img)
    cv2.imshow('Thresholding Uygulanmış Görüntü', thresh)
    cv2.imwrite('yaprak_threshold.png', thresh)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
