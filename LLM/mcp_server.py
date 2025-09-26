from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/weather", methods=["GET", "POST"])
def weather():
    if request.method == 'GET':
        return jsonify({"status": "ok"})
    if request.method == 'POST':
        data = request.json
        query = data.get("query", "").lower()

        # Very simple "NLU" simulation
        if "paris" in query:
            forecast = "Sunny in Paris with 24°C"
        elif "new york" in query:
            forecast = "Cloudy in New York with 18°C"
        else:
            forecast = f"Weather data for '{query}' is not available."

        return jsonify({
            "query": query,
            "forecast": forecast,
            "source": "Mock MCP Weather Server"
        })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
