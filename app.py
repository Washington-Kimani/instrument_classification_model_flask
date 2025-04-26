from flask import Flask, render_template, request
import os

app = Flask(__name__)



# home route render the index page
@app.route('/', methods=['GET', 'POST'])
def main():
    return render_template('index.html')

# about page
@app.route('/about')
def about_page():
    return render_template('about.html', title="About Page")

# prediction route for making the prediction
@app.route('/submit', methods=['GET', 'POST'])
def get_prediction():
    if request.method == 'POST':
        img = request.files['my_image']
        img_path = 'static/' + img.filename

        # save image to static folder
        img.save(img_path)

        #call the prediction method
        p = predict_label(img_path)

        return render_template("index.html", prediction = p, img_path = img_path)

if __name__ == '__main__':
    app.run(debug=True)