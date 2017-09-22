
import cv2
import grabCut
import numpy as np


def opitmize_contours(old_contours, threshold):
    list_area = []
    list_rect = []

    for contour in old_contours:
        area = cv2.contourArea(contour)
        list_area.append(float(area))
        list_rect.append(cv2.boundingRect(contour))

    max_area = max(list_area)
    max_rect = list_rect[np.argmax(list_area)]

    points = []
    new_contours = [old_contours[np.argmax(list_area)]]
    for i in new_contours[0]:
        points.append(i)

    for contour_ind in range(len(old_contours)):
        if list_area[contour_ind] * threshold > max_area:
            if max_rect[0] + 0.8 * max_rect[3] < list_rect[contour_ind][0] or list_rect[contour_ind][0] + 0.8 * list_rect[contour_ind][3] < max_rect[0]:
                new_contours.append(old_contours[contour_ind])
                for i in old_contours[contour_ind]:
                    points.append(i)

    return new_contours, points


def get_corners(points, width, height):
    min_dist1 = 10000000
    min_dist2 = 10000000
    min_dist3 = 10000000
    min_dist4 = 10000000

    for point in points:
        dist1 = (500 + point[0][0]) ** 2 + (500 + point[0][1]) ** 2
        if min_dist1 > dist1:
            min_dist1 = dist1
            min_point1 = (point[0][0] - 2, point[0][1] - 2)

        dist2 = (300 + width - point[0][0]) ** 2 + (500 + point[0][1]) ** 2
        if min_dist2 > dist2:
            min_dist2 = dist2
            min_point2 = (point[0][0] + 2, point[0][1] - 2)

        dist3 = (500 + width - point[0][0]) ** 2 + (500 + height - point[0][1]) ** 2
        if min_dist3 > dist3:
            min_dist3 = dist3
            min_point3 = (point[0][0] + 2, point[0][1] + 2)

        dist4 = (500 + point[0][0]) ** 2 + (500 + height - point[0][1]) ** 2
        if min_dist4 > dist4:
            min_dist4 = dist4
            min_point4 = (point[0][0] - 2, point[0][1] + 2)

    return [min_point1, min_point2, min_point3, min_point4]


def calc_corners_score(corners):
    ang = np.arctan2([corners[0][1] - corners[1][1], corners[1][1] - corners[2][1], corners[2][1] - corners[3][1],
                       corners[3][1] - corners[0][1]],
                      [corners[0][0] - corners[1][0], corners[1][0] - corners[2][0], corners[2][0] - corners[3][0],
                       corners[3][0] - corners[0][0]])
    ang = ang * 180 / np.pi

    for i in range(4):
        if ang[i] < 0:
            ang[i] += 180

    return min((ang[0] - ang[2]), 180 - (ang[0] - ang[2])) ** 2 + min((ang[1] - ang[3]), 180 - (ang[1] - ang[3])) **2

def get_billboard_corners(img_crop):

    # img_crop_remove = img_crop
    f_start = False

    """ ------------------------------ Delete background roughly --------------------------- """
    print("Delete Background ...")
    height, width = img_crop.shape[:2]
    rect = (4, 4, width - 8, height - 8)

    img_crop_remove = grabCut.grab_cut(img_crop, rect)
    # cv2.imshow("Image3_Remove", img_crop_remove)

    """ --------------------------- Threshold for delete bright color --------------------- """
    print("Threshold and Mask ...")
    img_crop_gray = cv2.cvtColor(img_crop_remove, cv2.COLOR_BGR2GRAY)
    img_crop_blur1 = cv2.medianBlur(img_crop_gray, 5)

    for thresh in range(30, 110, 2):
        # print thresh
        # thresh = 40
        _, img_crop_th1 = cv2.threshold(img_crop_blur1, thresh, 255, cv2.THRESH_BINARY)
        # cv2.imshow("Image4_Threshold1", img_crop_th1)

        """ ------------------------------------- Remove noise -------------------------------- """
        img_crop_blur2 = cv2.medianBlur(img_crop_th1, 5)
        # cv2.imshow("Image7_Blur", img_crop_blur2)

        """ ------------------------------------- Get Contours -------------------------------- """
        _, contours, _ = cv2.findContours(img_crop_blur2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        img_crop_contours = img_crop.copy()
        cv2.drawContours(img_crop_contours, contours, -1, (0, 255, 0), 1)
        # cv2.imshow("Image8_Contours", img_crop_contours)

        """ ------------------------------- Remove Unnecessary Contours ------------------------ """
        new_contours, points = opitmize_contours(contours, 2)

        img_crop_contours_new = img_crop.copy()
        cv2.drawContours(img_crop_contours_new, new_contours, -1, (0, 255, 0), 1)
        # cv2.imshow("Image9_Contours_New", img_crop_contours_new)

        """ ------------------------------------- Get Corners ---------------------------------- """
        corners = get_corners(points, width, height)
        score = calc_corners_score(corners)
        area = cv2.contourArea(np.array(corners))

        for corner in corners:
            cv2.circle(img_crop_contours_new, corner, 5, (255, 0, 0), -1)

        cv2.putText(img_crop_contours_new, '%.1f,%.1f' % (score, area), (0,25), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255))
        # cv2.imshow("Image10_Corners", img_crop_contours_new)

        cv2.imwrite('temp1/' + str(thresh) + '.jpg', img_crop_contours_new)

        """ ---------------------------------- Get Optimal Corner ------------------------------ """
        if score < 160:
            if f_start == False:
                area_start = area
                score_min = score
                img_min = img_crop_contours_new.copy()
                corners_min = corners
            else:
                if score < score_min and area > area_start*0.8:
                    score_min = score
                    img_min = img_crop_contours_new.copy()
                    corners_min = corners

            f_start = True

    cv2.imwrite('temp1/min.jpg', img_min)
    # cv2.waitKey(0)

    # return img_min
    return corners_min


if __name__ == '__main__':
    # image_file = '10.jpg'
    # img = cv2.imread('image_set/s' + image_file)
    img = cv2.imread('temp1.jpg')
    img1 = get_billboard_corners(img)
    # cv2.imwrite('image_set/c' + image_file, img1)
