from flask import Flask, request, jsonify
import flask_cors

from bert import QA
from gcp import GCP

app = Flask(__name__)
flask_cors.CORS(app)

model = QA()
gcp = GCP()


@app.route("/file/upload", methods=['POST'])
def file_upload():
    try:
        file = request.files.get('file')

        if not file:
            return 'No file uploaded.', 400

        doc_id = gcp.upload(file)

        return jsonify({'success': True, 'id': doc_id})
    except Exception as e:
        print(e)
        return jsonify({"success": False})


@app.route("/qa", methods=['GET', 'POST'])
def create_qa_record():
    try:
        if request.method == 'GET':
            doc_ids = request.args.get('doc_ids')
            doc_ids = list(map(int, doc_ids.split(',')))

            if len(doc_ids) == 0:
                return jsonify({"success": False, "message": "error_invalid_length"})
            return gcp.get_doc_records(doc_ids)
        elif request.method == 'POST':
            doc_list = request.json["doc_list"]
            if len(doc_list) == 0:
                return jsonify({"success": False, "message": "error_invalid_length"})
            qa_id = gcp.create_qa_record(doc_list)
            return {"success": True, 'id': qa_id}
    except Exception as e:
        print(e)
        return jsonify({"success": False})


@app.route("/ask/<qa_id>", methods=['GET'])
def get_answer(qa_id):
    try:
        question = request.args.get('question')
        if qa_id and question:
            qa = gcp.get_qa(int(qa_id))

            para_list = []
            for doc in qa['documents']:
                para_list += doc['paragraph']

            text_id, text, ans = model.predict(para_list, question)
            text_doc = ''
            count = 0
            for doc in qa['documents']:
                count += len(doc['paragraph'])
                if text_id < count:
                    text_doc = doc
            text_doc['paragraph'] = []

            return jsonify({"success": True, 'answer': {'doc': text_doc, 'text': text, 'ans': ans.replace('#', '')}})
        return jsonify({"success": False, 'message': 'id_not_define'})
    except Exception as e:
        print(e)
        return jsonify({"success": False})


# print a nice greeting.
def say_hello(username="World"):
    return '<p>Hello %s!</p>\n' % username


# some bits of text for the page.
header_text = '''
    <html>\n<head> <title>EB Flask Test</title> </head>\n<body>'''
instructions = '''
    <p><em>Hint</em>: This is a RESTful web service! Append a username
    to the URL (for example: <code>/Thelonious</code>) to say hello to
    someone specific.</p>\n'''
home_link = '<p><a href="/">Back</a></p>\n'
footer_text = '</body>\n</html>'

app.add_url_rule('/', 'index', (lambda: header_text + say_hello() + instructions + footer_text))

if __name__ == "__main__":
    app.run()