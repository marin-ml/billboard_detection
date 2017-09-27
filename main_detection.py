
import cv2
import adaptive_crop
import numpy as np
import get_billboard_region


def detect(img_file):
    print("Starting detection.")

    """ ----------------------------- Load image and display ----------------------------- """
    img_color = cv2.imread(img_file)
    # cv2.imshow('Image1_Original', img_color)

    """ ------------- Crop the Billboard region using Text with Google API --------------- """
    print("Cropping Billboard with text detection ...")
    score, crop_region, text_region = adaptive_crop.adaptive_crop_text(img_file, 0.6)

    if score is not None:
        print(score)
        # img_crop = img_color.copy()
        # cv2.rectangle(img_crop, (crop_region[0], crop_region[1]), (crop_region[2], crop_region[3]), (255, 0, 0), 3)
        # cv2.rectangle(img_crop, (text_region[0], text_region[1]), (text_region[2], text_region[3]), (0, 0, 255), 2)
        # cv2.imshow("Image2_Crop_Text", img_crop)
    else:
        """ ---------- Crop the Billboard region using Label with Google API ------------- """
        print("Cropping Billboard with label detection ...")
        crop_region, score = adaptive_crop.adaptive_crop_label(img_file, 16, 3)
        print(score)
        if crop_region is None:
            print("Billboard detection is failed!")
            return None

    """ ------------- Crop the Billboard region using Label with Google API -------------- """
    img_crop = img_color[crop_region[1]:crop_region[3], crop_region[0]:crop_region[2]]
    # cv2.imwrite('temp1.jpg', img_crop)
    # cv2.imshow("Image3_Cropped", img_crop)

    """ --------------------- Get corners of region from cropped image ------------------- """
    corners = get_billboard_region.get_billboard_corners(img_crop)

    """ -------------------- Make New contours connecting corners and fill --------------- """
    corners_org = []
    for i in range(4):
        corners_org.append([corners[i][0] + crop_region[0], corners[i][1] + crop_region[1]])

    final_contours = [np.array(corners_org, dtype=np.int32)]
    img_final = img_color.copy()
    cv2.drawContours(img_final, final_contours, 0, (255, 255, 0), -1)

    cv2.imshow('Image11_crop_mask', img_final)
    cv2.waitKey(0)

    return img_final


if __name__ == '__main__':
    image_file = '9.jpg'
    img1 = detect('image_set/' + image_file)

    if img1 is not None:
        cv2.imwrite('image_set/t' + image_file, img1)

