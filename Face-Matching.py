import cv2
import face_recognition as fr

img1 = fr.load_image_file("C:\\Users\\AR7 legion\\Desktop\\myPic.jpeg")
img2 = fr.load_image_file("C:\\Users\\AR7 legion\\Desktop\\anshul.jpeg")

img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2RGB)

enc1 = fr.face_encodings(img1)[0]
enc2 = fr.face_encodings(img2)[0]

res = fr.compare_faces([enc1], enc2, tolerance=0.5)

print(res)