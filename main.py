import cv2
from cvzone import HandTrackingModule, overlayPNG
import numpy as np
intro = cv2.imread('frames/img1.jpeg')
kill = cv2.imread('frames/img2.png')
winner = cv2.imread('frames/img3.png')
cam = cv2.VideoCapture(0)
detector = HandTrackingModule.HandDetector(maxHands=1, detectionCon=0.77)

sqr_img = cv2.imread('img/sqr(2).png')
mlsa = cv2.imread('img/mlsa.png')

gameOver = False
NotWon = True
while not gameOver:
    cv2.imshow("Squid Game Dalgona", intro)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

score = 0
dalgona_displayed = False
gameOver = False

# while NotWon and not gameOver:
#
#     success, img = cam.read()
#     if not success:
#         print("Error: failed to read webcam frame")
#         continue
#
#     img = cv2.flip(img, 1)

while NotWon and not gameOver:
    success, img = cam.read()
    if success:
        img = cv2.flip(img, 1)
        cv2.imshow("Squid Game Dalgona", img)
        key = cv2.waitKey(10)
        if key == ord('q'):
            break
    else:
        print("Failed to read webcam frame")


hands = detector.findHands(img)
if hands:
    for hand in hands:
        lmList = hand.landmark
        bboxInfo = hand.bbox

        if lmList:
            x, y = lmList[8].x, lmList[8].y
            bx, by, bw, bh = bboxInfo
            if bx < x < bx+bw and by < y < by+bh:

                if dalgona_displayed:
                    score += 1

                    img = overlayPNG(img, sqr_img, (bx, by))
                else:

                    NotWon = False
                    gameOver = True
                    for i in range(10):
                        img = kill
                        cv2.imshow("Squid Game Dalgona", img)
                        cv2.waitKey(100)
                    break

    cv2.putText(img, f"Score: {score}", (10, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    cv2.imshow("Squid Game Dalgona", img)
    cv2.waitKey(1)

    if not dalgona_displayed and np.random.randint(0, 100) < 5:
        dalgona_displayed = True
        img = overlayPNG(img, mlsa, (100, 100))
        cv2.imshow("Squid Game Dalgona", img)
        cv2.waitKey(np.random.randint(3, 5) * 1000)
        dalgona_displayed = False

if NotWon:

    for i in range(10):
        img = kill
        cv2.imshow("Squid Game Dalgona", img)
        cv2.waitKey(100)
else:

    img = winner
    cv2.imshow("Squid Game Dalgona", img)

while True:
    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()
