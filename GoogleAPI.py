
import base64
import json
import requests


class GoogleAPI:

    def __make_request_json(self, img_file, output_filename, type):

        # Read the image and convert to json
        with open(img_file, 'rb') as image_file:
            content_json_obj = {'content': base64.b64encode(image_file.read()).decode('UTF-8')}

        feature_json_obj = {'type': type}
        request_list = {'features': feature_json_obj, 'image': content_json_obj}

        # Write the object to a file, as json
        with open(output_filename, 'w') as output_json:
            json.dump({'requests': request_list}, output_json)

    def __get_text_info(self, json_file, field):

        data = open(json_file, 'rb').read()
        response = requests.post(
            url='https://vision.googleapis.com/v1/images:annotate?key=AIzaSyCkNNeL7iPi5qsGbDJSAExcPOOqKFZD42Y',
            data=data,
            headers={'Content-Type': 'application/json'})

        ret_json = json.loads(response.text)
        try:
            return ret_json['responses'][0][field]
        except:
            return None

    def get_google_json(self, img_file, type):
        temp_json = 'temp.json'
        if type == 'text':
            self.__make_request_json(img_file, temp_json, 'TEXT_DETECTION')
            ret_json = self.__get_text_info(temp_json, 'textAnnotations')
        elif type == 'label':
            self.__make_request_json(img_file, temp_json, 'LABEL_DETECTION')
            ret_json = self.__get_text_info(temp_json, 'labelAnnotations')

        return ret_json

