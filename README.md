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

## AI summary reuse

Successful AI text is cached on the app server's local disk without a daily
expiry. Review summaries reuse the same response when comments are unchanged,
including when sheet rows are reordered or only ratings change. Rating-based
template text is computed from current data. Added, edited, or removed comments
change the AI input; review filters that change the comment set do too.

Each successful section is saved separately, so a failed section can retry
without regenerating the others. Unavailable responses are held only in memory
for five minutes to avoid repeated requests; a later visit can retry. Existing
quote and structured-data fallbacks remain in use. Comparisons also reuse saved
AI text, but refresh when the ratings or sampled comments in their prompt change.

This cache adds no paid service. It does not make inference free or change the
configured AI provider. Local disk storage is not a durable backup: a cache clear
or a Cloud rebuild that replaces the server disk can remove saved responses.

Run the cache regression checks without making AI requests:

```bash
python -m unittest discover -s tests -v
```
