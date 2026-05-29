# Columbia Course Evaluation Dashboard

A Streamlit dashboard that reads course evaluation responses from a Google Sheet and surfaces difficulty/usefulness ratings, professor breakdowns, AI-generated comment summaries, and core-vs-elective views.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# edit .streamlit/secrets.toml with your real values

streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Push this repo to GitHub.
2. Go to https://share.streamlit.io, sign in with GitHub, click **New app**.
3. Pick this repo, branch `main`, main file `app.py`.
4. Open **Advanced settings → Secrets** and paste the contents of your local `.streamlit/secrets.toml`.
5. Deploy.

## Secrets

All credentials live in `.streamlit/secrets.toml` (gitignored). See `.streamlit/secrets.toml.example` for the schema. Required keys: `sheet_id`, `worksheet_name`, `[gcp_service_account]`. Optional: `HF_API_TOKEN` for AI summaries.

The Google service account needs read access to the target sheet — share the sheet with the service account's `client_email`.
