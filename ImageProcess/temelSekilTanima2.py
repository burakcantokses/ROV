import cv2

# resimi okutur
image = cv2.imread('C:\\Users\\Burak\\Desktop\\test.png')

# görüntünün rengini gri ton yapar
img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# görüntüdeki şekilleri tanımlama
ret, thresh = cv2.threshold(img_gray, 110, 255, cv2.THRESH_BINARY)

# cv2.CHAIN_APPROX_NONE kullanarak ikili görüntüdeki konturları tespit eder
contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)

# orijinal görüntüde konturlar çizer
image_copy = image.copy()
cv2.drawContours(image=image_copy, contours=contours, contourIdx=-1, color=(255, 0, 0), thickness=2,
                 lineType=cv2.LINE_AA)

# çıktı verir
cv2.imshow('None approximation', image_copy)
cv2.waitKey(0)
cv2.imwrite('C:\\Users\\Burak\\Desktop\\tcontours_none_image1.jpg', image_copy)
cv2.destroyAllWindows()