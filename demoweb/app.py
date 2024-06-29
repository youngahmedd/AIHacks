from flask import Flask, render_template

def create_app():
    app = Flask(__name__)

    @app.route('/')
    def hello_world():
        return render_template('index.html')  # Look for index.html in the templates folder

    return app

app = create_app()

if __name__ == '__main__':
    app.run(debug=True)