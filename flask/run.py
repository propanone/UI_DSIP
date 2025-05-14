from app import app  # UPDATED import

if __name__ == '__main__':
    app.run(debug=app.config.get('DEBUG', True), 
            port=app.config.get('PORT', 5003), # Ensure port matches config
            host='0.0.0.0') # Makes it accessible on the network if needed