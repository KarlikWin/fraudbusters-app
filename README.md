# FraudBusters — Deployed App 

## Layout

```
deployed_app/
├── app.py              # Streamlit UI
├── train_model.py      
├── requirements.txt
├── README.md
└── models/            
    ├── fraud_pipeline.joblib
    ├── metadata.json
    └── background_sample.npy
```

Outputs:

- Fraud probability + decision (threshold tuned on test PR curve)
- **Global SHAP** importance on background sample
- **Local SHAP waterfall** with top-3 driver explanation in plain English