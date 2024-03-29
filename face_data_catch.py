import cv2 as cv2
import os


# import sys
#
# from PIL import Image

def CatchPICFromVideo(window_name: str, camera_idx: int, catch_pic_num: int, path_name: str, person_name: str):
    path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

    cv2.namedWindow(window_name)
    # 视频来源，可以选择摄像头或者视频
    # cap = cv2.VideoCapture(camera_idx)
    cap = cv2.VideoCapture(path + '/' + person_name + '.mp4')

    # 告诉OpenCV使用人脸识别分类器（这里填你自己的OpenCV级联分类器地址）
    classfier = cv2.CascadeClassifier(path + '\haarcascades\haarcascade_frontalface_alt2.xml')

    # 识别出人脸后要画的边框的颜色，RGB格式q
    color = (0, 255, 0)
    num = 0
    while cap.isOpened():
        ok, frame = cap.read()  # 读取一帧数据
        if not ok:
            break

        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 将当前桢图像转换成灰度图像

        # 人脸检测，1.2和2分别为图片缩放比例和需要检测的有效点数
        faceRects = classfier.detectMultiScale(grey, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
        if len(faceRects) > 0:  # 大于0则检测到人脸
            for faceRect in faceRects:  # 单独框出每一张人脸
                x, y, w, h = faceRect

                # 将当前帧保存为图片
                img_name = '%s/%d.jpg' % (path_name, num)
                image = frame[y - 10: y + h + 10, x - 10: x + w + 10]
                cv2.imwrite(img_name, image)

                num += 1
                if num > catch_pic_num:  # 如果超过指定最大保存数量退出循环
                    break

                # 画出矩形框
                cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, 2)

                # 左上角显示当前捕捉到了多少人脸图片了
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame, 'num:%d' % (num), (30, 30), font, 1, (255, 0, 0), 1)

                # 超过指定最大保存数量结束程序
        if num > catch_pic_num:
            break

        # 显示图像
        cv2.imshow(window_name, frame)
        c = cv2.waitKey(10)
        if c & 0xFF == ord('q'):
            break

            # 释放摄像头并销毁所有窗口
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    # person_name = 'sissy'
    person_name = 'shenteng'
    CatchPICFromVideo("catch_face_data", 0, 200 - 1, path + '/facedata/' + person_name, person_name)  # 采集200张，保存在chengzihang这个文件夹下面
