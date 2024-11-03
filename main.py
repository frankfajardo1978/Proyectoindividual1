# main.py
from flask import Flask, request, jsonify
from recomendacion import recomendacion

# Inicializar la aplicación Flask
app = Flask(__name__)

# Ruta para la recomendación
@app.route('/recomendacion', methods=['GET'])
def get_recommendations():
    titulo = request.args.get('titulo')
    if not titulo:
        return jsonify({"error": "Debe proporcionar un título de película."}), 400
    
    recomendaciones = recomendacion(titulo)
    if not recomendaciones:
        return jsonify({"error": "Título no encontrado en la base de datos."}), 404

    return jsonify({"recomendaciones": recomendaciones})

# Iniciar la aplicación
if __name__ == '__main__':
    app.run(debug=True)
