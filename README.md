# How to run 

Create your env (.venv) for example
```
python -m venv .venv
```

install the requirements
```
pip install -r requirement.txt
```

train the models (optional)
```
cd src/models/
python randomforest.py
python xgboost.py
```

run the streamlit app 
```
streamlit run app_v1.py --server.port 80 --server.address 0.0.0.0
```