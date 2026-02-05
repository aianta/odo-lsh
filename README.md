# MinHash LSH Flask Boilerplate

This repository contains a small Flask application with four API endpoint stubs for you to implement.

Endpoints:

- `POST /minhashLSH` - create a new MinHash LSH instance (stub)
- `PUT /minhashLSH/:id` - update an LSH instance (stub)
- `POST /minhash/:id/query` - query an LSH instance (stub)
- `GET /minhashLSH/:id/clustering` - get clustering info (stub)

Quick start:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app.py
```

The server will run on `http://0.0.0.0:5000` by default.

Implement the actual MinHash/LSH logic in `app.py` where marked.
