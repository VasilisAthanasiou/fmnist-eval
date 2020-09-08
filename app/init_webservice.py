from src.flask import views

if __name__ == '__main__':
    views.app.secret_key = 'secret'
    views.app.run(host='localhost', debug=True)