import numpy as np
import PIL.Image as Image
import tensorflow as tf
import tensorflow_hub as hub
from flask import Flask,request,jsonify
app = Flask(__name__)


IMAGE_SHAPE=(224,224)
classifier=tf.keras.Sequential([
    hub.KerasLayer('https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4',input_shape=IMAGE_SHAPE+(3,))
])
with open('ImageNetLabels.txt','r') as f:
    image_labels=f.read().splitlines()

@app.route("/",methods=['GET','POST'])
def hello_world():
    return jsonify({'Message':"Welcome to Flask API created by Sunny! Send your picture via 'POST' request on '/check' end point",'Note':'This API is an image classifier.'})

@app.route("/check",methods=['GET','POST'])
def check():

    if request.method=='POST':
        model_type=request.form['model_type']
        # print(model_type)
        image=request.files['image']
        img_path='static/'+image.filename
        image.save(img_path)

        img = Image.open(img_path).resize(IMAGE_SHAPE)
        img = np.array(img) / 255.0
        # prediction expect multiple omages as input thats why we are
        # onverting to 4d array
        img4d = img[np.newaxis, ...]
        print('#####Before Prediction#####')
        result = classifier.predict(img4d)
        print(result)
        predicted_label_index = np.argmax(result)
        print(predicted_label_index)
        print(image_labels[predicted_label_index])
        return jsonify({'result':image_labels[predicted_label_index],'API Type':'Flask API'})
    return jsonify({'Message':"Send your picture via 'POST' request on this end point"})

if __name__=='__main__':
    app.run(debug=True,port=9000)
