
from GoogleAPI import GoogleAPI
import cv2


google_ocr = GoogleAPI()

def get_label_score(json_data):
    ret_score = []
    for sub_data in json_data:
        if sub_data['description'] == 'advertising':
            ret_score.append(sub_data['score'])
        elif sub_data['description'] == 'billboard':
            ret_score.append(sub_data['score'])

    if ret_score:
        return max(ret_score)
    else:
        return 0


def adaptive_crop_label(img_file, scale, margin):
    json_label = google_ocr.get_google_json(img_file, 'label')
    score = get_label_score(json_label)
    # print(score)

    img = cv2.imread(img_file)
    height, width = img.shape[:2]
    dx = int(width / scale)
    dy = int(height / scale)
    temp_file = 'temp.jpg'

    if score > 0.6:     # if first image is billboard
        y1 = 0
        y2 = height
        x1 = 0
        x2 = width

        state = 0
        f_loop = True

        while f_loop:

            if state == 0:
                x1 += dx
            elif state == 1:
                x2 -= dx
            elif state == 2:
                y1 += dy
            elif state == 3:
                y2 -= dy
            else:
                break

            img_new = img[y1:y2, x1:x2]
            cv2.imwrite(temp_file, img_new)
            json_label = google_ocr.get_google_json(temp_file, 'label')
            score_new = get_label_score(json_label)
            # print score_new

            if score_new < score * 0.95:
                if state == 0:
                    x1 -= margin*dx
                elif state == 1:
                    x2 += margin*dx
                elif state == 2:
                    y1 -= margin*dy
                elif state == 3:
                    y2 += margin*dy

                state += 1

        return [max(x1-dx, 0), max(y1-dy, 0), min(width, x2+dx), min(height, y2)], score_new

    else:       # if first image isn't billboard

        win_size = 8
        max_score = 0
        for i in range(0, scale-win_size, 2):
            for j in range(0, scale-win_size, 2):
                img_new = img[i*dy:(i+win_size)*dy, j*dx:(j+win_size)*dx]
                cv2.imwrite(temp_file, img_new)
                json_label = google_ocr.get_google_json(temp_file, 'label')
                score_new = get_label_score(json_label)
                # print score_new
                if score_new > max_score:
                    max_score = score_new
                    max_rect = [j*dx, i*dy, (j+win_size)*dx, (i+win_size)*dy]

        if max_score > 0.8:
            return max_rect, max_score
        else:
            return None, max_score


def adaptive_crop_text(img_file, margin):
    img = cv2.imread(img_file)
    img_height, img_width = img.shape[:2]
    json_text = google_ocr.get_google_json(img_file, 'text')

    if json_text is None:
        return None, None, None

    region = json_text[0]['boundingPoly']['vertices']
    if 'x' in region[0]:
        x1 = region[0]['x']
    else:
        x1 = 0

    if 'y' in region[0]:
        y1 = region[0]['y']
    else:
        y1 = 0

    x2 = region[1]['x']
    y2 = region[3]['y']

    text_width = x2 - x1
    text_height = y2 - y1

    if text_height * 6 < text_width:
        y_extra = 6 * margin
    elif text_height * 4 < text_width:
        y_extra = 4 * margin
    else:
        y_extra = margin

    nx1 = max(0, int(x1 - text_width * margin))
    nx2 = min(img_width, int(x2 + text_width * margin))
    ny1 = max(0, int(y1 - text_height * y_extra))
    ny2 = min(img_height, int(y2 + text_height * y_extra))

    img_new = img[ny1:ny2, nx1:nx2]
    cv2.imwrite('temp.jpg', img_new)
    json_label = google_ocr.get_google_json('temp.jpg', 'label')
    score = get_label_score(json_label)
    # print score

    if score < 0.92:
        ny1 = max(0, int(ny1 - text_height * 3 * margin))
        ny2 = min(img_height, int(ny2 + text_height * 3 * margin))
        img_new = img[ny1:ny2, nx1:nx2]
        cv2.imwrite('temp.jpg', img_new)
        json_label = google_ocr.get_google_json('temp.jpg', 'label')
        score = get_label_score(json_label)
        # print score

    if score < 0.92:
        nx1 = max(0, int(nx1 - text_width * 5 * margin))
        nx2 = min(img_height, int(nx2 + text_width * 5 * margin))
        ny1 = max(0, int(ny1 - text_height * 3 * margin))
        ny2 = min(img_height, int(ny2 + text_height * 3 * margin))
        img_new = img[ny1:ny2, nx1:nx2]
        cv2.imwrite('temp.jpg', img_new)
        json_label = google_ocr.get_google_json('temp.jpg', 'label')
        score = get_label_score(json_label)
        # print score

    if score < 0.92:
        return None, None, None
    else:
        return score, [nx1, ny1, nx2, ny2], [x1, y1, x2, y2]


if __name__ == '__main__':

    filename = '10.jpg'
    image_file = 'image_set/' + filename

    crop_region = adaptive_crop_label(image_file, 16, 3)
    # crop_region = adaptive_crop_text(image_file, 0.6)

    # if crop_region is not None:
    #     img = cv2.imread(image_file)
    #     cv2.imwrite('L' + filename, img[crop_region[1]:crop_region[3], crop_region[0]:crop_region[2]])
