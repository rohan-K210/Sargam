from flask import Flask
from api.routes.predict_local import bp as predict_local_bp

def create_app():
    app = Flask(__name__)
    
    # Register blueprints
    app.register_blueprint(predict_local_bp)
    
    # Existing Gemini endpoint would be registered here
    # from api.routes.predict_gemini import bp as predict_gemini_bp
    # app.register_blueprint(predict_gemini_bp)
    
    @app.route('/health', methods=['GET'])
    def health():
        return {'status': 'ok', 'service': 'sargam-api'}
    
    return app


if __name__ == '__main__':
    app = create_app()
    app.run(host='0.0.0.0', port=5000, debug=True)