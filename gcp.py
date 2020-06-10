import time
import string
import random
import io
from google.cloud import storage
from google.cloud import datastore

from pdf import extract_text_pdf, extract_text_txt

bucket_name = 'fyp-bert-qa'
documents_dir = 'documents/'
kind_qa = u'qa'
kind_doc = u'doc'


def hash_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))


class GCP:

    def __init__(self):
        self.storage_client = storage.Client().from_service_account_json('fyp-qa-eb7816dfb87e.json')
        self.bucket = self.storage_client.get_bucket(bucket_name)
        self.dc = datastore.Client().from_service_account_json('fyp-qa-eb7816dfb87e.json')

    def upload(self, file):
        file_obj = file.read()
        ext = file.filename.split('.')[1]
        para_list = []
        if ext == 'txt':
            para_list = extract_text_txt(file_obj.decode())
            print(para_list)
        else:
            para_list = extract_text_pdf(io.BytesIO(file_obj))

        doc_id = self.insert(kind_doc, {'filename': file.filename, 'paragraph': para_list})

        blob = self.bucket.blob(documents_dir + "{}.{}".format(doc_id, ext))

        blob.upload_from_string(
            file.read(),
            content_type=file.content_type
        )
        return doc_id

    def get_doc_records(self, doc_ids):
        docs = self.get_multi(kind_doc, doc_ids)

        for doc in docs:
            doc['id'] = doc.key.id

        return {'doc_list': docs}

    def create_qa_record(self, doc_list):
        documents = []
        for doc in doc_list:
            arr_entity = datastore.Entity(exclude_from_indexes=['paragraph'])
            arr_entity.update(doc)
            documents.append(arr_entity)

        return self.insert(kind_qa, {'documents': documents})

    def get_qa(self, qa_id):
        qa = self.get_one(kind_qa, qa_id)
        return qa

    def insert(self, kind, data):
        with self.dc.transaction():
            key = self.dc.key(kind)
            record = datastore.Entity(key=key, exclude_from_indexes=['paragraph'])
            record.update(data)
            self.dc.put(record)
        return record.key.id

    def get_one(self, kind, id):
        key = self.dc.key(kind, id)
        records = self.dc.get(key)
        return records

    def get_multi(self, kind, ids):
        keys = []
        for item in ids:
            keys.append(self.dc.key(kind, item))
        records = self.dc.get_multi(keys)
        return records
