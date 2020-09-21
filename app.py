from flask import Flask, request, Response, jsonify, send_from_directory, abort
from datetime import timedelta
import cv2
import os
import logging
from logging import Formatter, FileHandler
import imutils
import numpy as np

from ocr import process_image

app = Flask(__name__)

@app.route('/')
def home():
    return home()

@app.route('/ocr', methods=['POST'])
def ocr():
    try:
        url = request.json['image_url']
        if 'jpg' in url:
            output = process_image(url)
            return jsonify({"output",output})
        else:
            return jsonify({"error" : "only .jpg files"})
    except:
        return jsonify(
            {"error": "Did you mean to send: {'image_url': 'some_jpeg_url'}"}
        )

if __name__ == "__main__":
    app.run(debug=True)