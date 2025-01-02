from flask import Flask, request, jsonify, send_from_directory
import requests

# Initialize Flask app
app = Flask(__name__, static_folder='static')

# Server configuration
SERVER_PORT = 5555  # Your inference server port
HEADERS = {"Content-Type": "application/json"}

def request_inference(data):
    """
    Send inference request to the server running on port 5555.
    """
    try:
        response = requests.put(f'http://localhost:{SERVER_PORT}/generate', json=data, headers=HEADERS)
        response.raise_for_status()  # Raise an error for HTTP issues
        result = response.json()
        print(result)
        return result.get('sentences', [])
    except requests.exceptions.RequestException as e:
        print(f"Error communicating with inference server: {e}")
        return []

def extract_answer(response, question):
    """
    Extract the answer from the generated response.
    Assumes the response contains the question followed by the answer.
    """
    try:
        response_text = response[0]  # Assuming single output per batch
        question_lower = question.lower()

        # Find where the question ends in the response
        if question_lower in response_text.lower():
            answer_start = response_text.lower().index(question_lower) + len(question)
            answer = response_text[answer_start:].strip()

            # Remove repeated occurrences of the question or redundant text
            if question.strip('?').lower() in answer.lower():
                # Split the answer to remove the repeated question
                answer = answer.split(question.strip('?'))[0].strip()

            # Split by unnecessary content (like new questions or unexpected text)
            if '?' in answer:
                answer = answer.split('?')[0].strip()

            return answer
        else:
            return response_text.strip()
    except Exception as e:
        print(f"Error while extracting answer: {e}")
        return response[0] if response else ""

@app.route('/')
def index():
    # Serve the HTML interface
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/query', methods=['POST'])
def query():
    """
    Handle user queries and fetch responses from the inference server.
    """
    data = request.json
    user_query = data.get('query', '')

    if not user_query:
        return jsonify({'error': 'Query cannot be empty.'}), 400

    try:
        # Prepare the payload for the inference server
        payload = {
            "sentences": [user_query],
            "tokens_to_generate": 300,
            "temperature": 0.1,
            "add_BOS": True,
            "top_k": 0,
            "top_p": 0.9,
            "greedy": False,
            "all_probs": False,
            "repetition_penalty": 1.2,
            "min_tokens_to_generate": 2,
        }

        # Request inference from the external server
        sentences = request_inference(payload)
        
        # Use the extract_answer function to process the output
        extracted_answer = extract_answer(sentences, user_query)

        return jsonify({'answer': extracted_answer})
    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
