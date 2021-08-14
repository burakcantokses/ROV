import cv2

# resimi okutur
image = cv2.imread('C:\\Users\\Burak\\Desktop\\test.png')

# görüntünün rengini gri ton yapar
img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# görüntüdeki şekilleri tanımlama
ret, thresh = cv2.threshold(img_gray, 130, 255, cv2.THRESH_BINARY)

# işlenen görüntüyü gösterir
cv2.imshow('Binary image', thresh)
cv2.waitKey(0)

# işlenen görüntüyü yazar
cv2.imwrite('C:\\Users\\Burak\\Desktop\\test2.png', thresh)
cv2.destroyAllWindows()