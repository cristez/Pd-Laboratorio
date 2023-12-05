from flask import Flask, request, render_template
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import tensorflow as tf
from io import BytesIO
from PIL import Image
import base64

app = Flask(__name__)
modelo = tf.keras.models.load_model('miModelo.h5')  

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        image_file = request.files['image']  
        if image_file:
            
            image = Image.open(BytesIO(image_file.read()))
            image = image.convert('RGB') 
            image = image.resize((64, 64))  
            img_array = img_to_array(image) / 255.0  
            img_array = tf.expand_dims(img_array, axis=0)

            prediccion = modelo.predict(img_array)
            prediccion_label = "Infected" if prediccion[0][0] > 0.5 else "Uninfected"

            # Convertir la imagen a Base64 para enviarla de vuelta a la plantilla
            buffered = BytesIO()
            image.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode()

            return render_template('index.html', prediction=prediccion_label, image_data=img_str)

    return render_template('index.html')

            

if __name__ == '__main__':
    app.run(debug=True)