# Databricks notebook source
# MAGIC %md
# MAGIC # Nyaya Dhwani — legal RAG ingestion (BNS → IPC → `legal_rag_corpus`)
# MAGIC
# MAGIC Focused pipeline for the hackathon app. **Run sections in order.**
# MAGIC
# MAGIC | Step | What |
# MAGIC |------|------|
# MAGIC | 0–2 | Install + config + Unity Catalog Volume |
# MAGIC | 1b | **Sarvam** — optional API ping |
# MAGIC | 3 | **BNS sections** — CSV on Volume (uploaded), then mirrors, then optional PDF |
# MAGIC | 4 | BNS ↔ IPC mapping |
# MAGIC | 5 | `legal_rag_corpus` for retrieval / `build_rag_index.ipynb` |
# MAGIC | 6 | Helpers + verification SQL |
# MAGIC
# MAGIC _Legacy scrapes (data.gov.in, DAKSH, PRS, schemes) were removed from this notebook to reduce noise; restore from git history if you need them._

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. Install dependencies
# MAGIC Run once per cluster/session restart.

# COMMAND ----------

!pip install -q requests pandas beautifulsoup4 lxml openpyxl pymupdf
print('✅ Packages installed')

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Config & shared helpers
# MAGIC **Edit `DATAGOV_API_KEY` before running anything else.**

# COMMAND ----------

# Secrets: create scope `nyaya-dhwani` in Databricks (User Settings → Secrets) or set env vars locally.
import os

def _read_secret(env_name: str, scope_key: str = None):
    """Databricks secret scope `nyaya-dhwani`, else upper-case env var."""
    try:
        return dbutils.secrets.get(scope="nyaya-dhwani", key=scope_key or env_name.lower())  # noqa: F821
    except Exception:
        return os.environ.get(env_name, "")

DATAGOV_API_KEY = _read_secret("DATAGOV_API_KEY", "datagov_api_key")
CATALOG  = 'workspace'
SCHEMA   = 'india_legal'
VOLUME   = 'legal_files'

import re, time, requests
import pandas as pd
from io       import StringIO, BytesIO
from datetime import datetime
from bs4      import BeautifulSoup

HEADERS  = {'User-Agent': 'Mozilla/5.0 (research/educational use)'}
VOL_PATH = f'/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}'
PDF_DIR  = f'{VOL_PATH}/pdfs'
pdf_paths = {}  # filled by §3c Legal PDFs

def clean_cols(df):
    df.columns = [
        re.sub(r'[\s,;{}()\n\t=]+', '_', str(c)).strip('_')
        for c in df.columns
    ]
    return df

def save_table(df, table):
    df  = clean_cols(df)
    sdf = spark.createDataFrame(df.astype(str))
    sdf.write.format('delta').mode('overwrite') \
       .option('overwriteSchema', 'true') \
       .saveAsTable(f'{CATALOG}.{SCHEMA}.{table}')
    print(f'  💾 {CATALOG}.{SCHEMA}.{table}  ({df.shape[0]} rows × {df.shape[1]} cols)')

if not DATAGOV_API_KEY:
    print('⚠️  DATAGOV_API_KEY is empty — set env DATAGOV_API_KEY or secret nyaya-dhwani/datagov_api_key')
else:
    print('✅ Config loaded (API key present)')
print(f'   Volume : {VOL_PATH}')

print("hello world")


# COMMAND ----------

# MAGIC %md
# MAGIC ## 1b. Test Sarvam connectivity (optional)
# MAGIC Runs one minimal `chat/completions` call. Requires secret `nyaya-dhwani` / `sarvam_api_key` or env `SARVAM_API_KEY`. [Dashboard](https://dashboard.sarvam.ai)
# MAGIC

# COMMAND ----------

# ── Sarvam connectivity ping ───────────────────────────────────────────────
SARVAM_API_URL = 'https://api.sarvam.ai/v1/chat/completions'
SARVAM_MODEL = 'sarvam-m'
_sk = _read_secret('SARVAM_API_KEY', 'sarvam_api_key')

if not _sk:
    print('⚠️  No Sarvam key — set SARVAM_API_KEY or secret nyaya-dhwani/sarvam_api_key')
else:
    print(f'🔑 Key present (len={len(_sk)}), calling {SARVAM_MODEL}...')
    try:
        r = requests.post(
            SARVAM_API_URL,
            headers={'Authorization': f'Bearer {_sk}', 'Content-Type': 'application/json'},
            json={
                'model': SARVAM_MODEL,
                'messages': [{'role': 'user', 'content': 'Reply with exactly: ok'}],
                'temperature': 0,
                'max_tokens': 8,
            },
            timeout=30,
        )
        print(f'   HTTP {r.status_code}')
        r.raise_for_status()
        data = r.json()
        text = data['choices'][0]['message']['content'].strip()
        print(f'✅ Sarvam OK — reply: {text!r}')
    except requests.HTTPError as e:
        sc = getattr(e.response, 'status_code', None)
        body = (e.response.text or '')[:300] if e.response is not None else ''
        print(f'❌ HTTP {sc}: {e}')
        if body:
            print(f'   Body (truncated): {body!r}')
        if sc == 403:
            print('   403 often means invalid key, wrong model access, or billing — check dashboard.sarvam.ai')
    except Exception as e:
        print(f'❌ Error: {type(e).__name__}: {e}')


# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Unity Catalog setup
# MAGIC Creates the schema and Volume. Safe to re-run.

# COMMAND ----------

spark.sql(f'CREATE DATABASE IF NOT EXISTS {CATALOG}.{SCHEMA}')
spark.sql(f'CREATE VOLUME  IF NOT EXISTS {CATALOG}.{SCHEMA}.{VOLUME}')
os.makedirs(PDF_DIR, exist_ok=True)
print(f'✅ Schema : {CATALOG}.{SCHEMA}')
print(f'✅ Volume : {VOL_PATH}')
print(f'✅ PDF dir: {PDF_DIR}')

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. BNS sections — load CSV from Volume (primary)

# COMMAND ----------

# ── BNS sections: Volume CSV first (Nyaya Dhwani) ───────────────────────────
df_bns = pd.DataFrame()

# Add any path where you uploaded bns_sections.csv (driver must see /Volumes/...)
BNS_CSV_PATHS = [
    '/Volumes/main/india_legal/bns_sections/bns_sections.csv',
    f'{VOL_PATH}/bns_sections.csv',
    f'{VOL_PATH}/bns_sections/bns_sections.csv',
]

for _p in BNS_CSV_PATHS:
    try:
        if os.path.exists(_p):
            df_bns = pd.read_csv(_p)
            print(f'✅ BNS CSV: {len(df_bns)} rows from {_p}')
            display(df_bns.head(3))
            break
    except Exception as _e:
        print(f'⚠️  {_p}: {_e}')

if df_bns.empty:
    print('No Volume CSV — next cell tries GitHub mirrors.')

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3a. GitHub mirrors (if Volume empty)

# COMMAND ----------

# ── GitHub mirrors (only if Volume CSV missing) ─────────────────────────────
if df_bns.empty:
    BNS_CSV_MIRRORS = [
        'https://raw.githubusercontent.com/OpenNyAI/Opennyai/main/datasets/bns_sections.csv',
        'https://raw.githubusercontent.com/nandr39/bns-dataset/main/bns_sections.csv',
    ]
    for url in BNS_CSV_MIRRORS:
        print(f'  Trying: {url}')
        try:
            r = requests.get(url, timeout=30, headers=HEADERS)
            r.raise_for_status()
            df_bns = pd.read_csv(StringIO(r.text))
            if df_bns.shape[0] > 10:
                print(f'  ✅ {df_bns.shape[0]} sections from mirror')
                display(df_bns.head(3))
                break
        except Exception as e:
            print(f'  ⚠️  {e}')
            df_bns = pd.DataFrame()
else:
    print('Skipping mirrors — Volume CSV already loaded.')

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3c. Legal PDFs — download BNS gazette (for 3b)

# COMMAND ----------

# ── 3c: Download official legal PDFs to Unity Catalog Volume ─────────────
#
# WHY THE OLD URLS FAILED:
#   indiacode.nic.in /bitstream/.../bns_2023.pdf  →  redirects to a login/error
#   page (~18 KB HTML), not the actual PDF.
#
# FIXED URLS:
#   BNS  & BNSS → MHA (Ministry of Home Affairs) direct Gazette PDFs
#   BSA  & Constitution → indiacode.nic.in bitstream (different handle numbers)
#
# SIZE CHECK: a real PDF will be > 500 KB. Anything < 100 KB is an error page.
# ─────────────────────────────────────────────────────────────────────────────

LEGAL_PDFS = {
    'bns_2023': {
        'url':  'https://www.mha.gov.in/sites/default/files/250883_english_01042024.pdf',
        'desc': 'Bharatiya Nyaya Sanhita 2023 — replaces IPC (MHA Gazette)',
        'min_bytes': 500_000,   # real PDF is ~1.5 MB
    },
    'bnss_2023': {
        'url':  'https://www.mha.gov.in/sites/default/files/2024-04/250884_2_english_01042024.pdf',
        'desc': 'Bharatiya Nagarik Suraksha Sanhita 2023 — replaces CrPC (MHA Gazette)',
        'min_bytes': 500_000,
    },
    'bsa_2023': {
        # Primary: MHA Gazette (same source as BNS/BNSS, confirmed pattern)
        # Fallback 1: Tripura HC hosts a clean copy from NIC
        # Fallback 2: Indian Railways NIC upload
        'url':  'https://www.mha.gov.in/sites/default/files/2024-04/250882_english_01042024_0.pdf',
        'fallback_urls': [
            'https://thc.nic.in/Central%20Governmental%20Acts/Bharatiya%20Sakshya%20Adhiniyam,%202023.pdf',
            'https://ncr.indianrailways.gov.in/uploads/files/1748345531944-Bharatiya%20Sakshya%20Adhiniyam-2023(English).pdf',
        ],
        'desc': 'Bharatiya Sakshya Adhiniyam 2023 — replaces Evidence Act (MHA Gazette)',
        'min_bytes': 200_000,
    },
    'constitution_of_india': {
        'url':  'https://www.indiacode.nic.in/bitstream/123456789/19151/1/constitution_of_india.pdf',
        'desc': 'Constitution of India (as amended)',
        'min_bytes': 1_000_000,  # ~3 MB
    },
}

pdf_paths = {}

for name, meta in LEGAL_PDFS.items():
    dest = f'{PDF_DIR}/{name}.pdf'

    # Check if already downloaded and valid
    if os.path.exists(dest):
        size = os.path.getsize(dest)
        if size >= meta['min_bytes']:
            print(f'  ✅ Already valid: {name}.pdf  ({size:,} bytes)')
            pdf_paths[name] = dest
            continue
        else:
            print(f'  ⚠️  Existing {name}.pdf is too small ({size:,} bytes) — re-downloading')
            os.remove(dest)

    # Build list of URLs to try: primary + any fallbacks
    urls_to_try = [meta['url']] + meta.get('fallback_urls', [])

    saved = False
    for attempt, url in enumerate(urls_to_try):
        label = 'primary' if attempt == 0 else f'fallback {attempt}'
        print(f'  📥 {name}.pdf  [{label}]')
        print(f'     {meta["desc"]}')
        print(f'     {url}')

        try:
            r = requests.get(
                url,
                timeout=120,
                headers={**HEADERS, 'Accept': 'application/pdf,*/*'},
                stream=True,
                allow_redirects=True,
            )
            r.raise_for_status()

            content_type = r.headers.get('Content-Type', '')
            if 'html' in content_type.lower():
                print(f'     ⚠️  Got HTML ({content_type}) — trying next URL')
                continue

            with open(dest, 'wb') as f:
                for chunk in r.iter_content(chunk_size=65536):
                    f.write(chunk)

            size = os.path.getsize(dest)
            if size < meta['min_bytes']:
                print(f'     ⚠️  Too small ({size:,} bytes) — trying next URL')
                os.remove(dest)
                continue

            pdf_paths[name] = dest
            print(f'     ✅ Saved ({size:,} bytes)')
            saved = True
            break

        except Exception as e:
            print(f'     ⚠️  {e} — trying next URL')

    if not saved:
        print(f'  ❌ All URLs failed for {name}.pdf')
        print(f'     Manual download options:')
        for url in urls_to_try:
            print(f'       {url}')
        print(f'     Upload to Volume and rename to {name}.pdf')

    print()

# ── Summary ───────────────────────────────────────────────────────────────────
print('─' * 60)
print(f'✅ {len(pdf_paths)}/{len(LEGAL_PDFS)} PDFs ready in {PDF_DIR}')
for name, path in pdf_paths.items():
    print(f'   {name}.pdf  →  {os.path.getsize(path):,} bytes')

if len(pdf_paths) < len(LEGAL_PDFS):
    missing = [n for n in LEGAL_PDFS if n not in pdf_paths]
    print(f'\n⚠️  Missing: {missing}')
    print('   For each missing PDF:')
    print('   1. Open the URL above in your browser → Save As PDF')
    print('   2. Catalog → Volumes → legal_files → Upload')
    print('   3. Rename to <name>.pdf in the volume')
    print('   4. Cell 9b will find it via pdf_paths dict — re-run this cell after uploading')
    # Still register manually uploaded files if they exist
    for name in missing:
        manual_path = f'{PDF_DIR}/{name}.pdf'
        if os.path.exists(manual_path) and os.path.getsize(manual_path) > 10_000:
            pdf_paths[name] = manual_path
            print(f'   Found manually uploaded: {name}.pdf ✅')

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3b. Optional — PDF + `ai_parse_document` + Sarvam (only if no BNS rows yet)

# COMMAND ----------

# ── Optional: BNS from PDF + ai_parse_document + Sarvam (only if CSV empty) ─
# Requires §3c Legal PDFs (above) to have populated pdf_paths, or place bns_2023.pdf on Volume.

from pyspark.sql import functions as F
from pyspark.sql.functions import col, explode, expr, get_json_object, to_json, from_json
from pyspark.sql.types import ArrayType, StructField, StringType, IntegerType, StructType
import os
import requests, json, re, time

SARVAM_API_KEY = _read_secret("SARVAM_API_KEY", "sarvam_api_key")
SARVAM_API_URL = 'https://api.sarvam.ai/v1/chat/completions'
SARVAM_MODEL   = 'sarvam-m'

BNS_PDF_PATH = f'{PDF_DIR}/bns_2023.pdf'

elements_df = None

if not df_bns.empty:
    print('✅ Skipping PDF pipeline — df_bns already populated from CSV/mirrors.')
elif 'bns_2023' not in pdf_paths:
    print('⚠️  bns_2023.pdf not found — download cell "Legal PDFs" or upload bns_sections.csv.')
else:
    print(f'📄 Parsing BNS PDF with ai_parse_document...')
    print(f'   {BNS_PDF_PATH}  ({os.path.getsize(BNS_PDF_PATH):,} bytes)')

    raw_df = spark.read.format('binaryFile').load(BNS_PDF_PATH)

    parsed_json_df = raw_df.select(
        expr("to_json(ai_parse_document(content, map('version', '2.0')))").alias('parsed_json')
    )
    parsed_json_df.write.format('delta').mode('overwrite') \
             .option('overwriteSchema', 'true') \
             .saveAsTable(f'{CATALOG}.{SCHEMA}.bns_parsed_raw')
    parsed_df = spark.table(f'{CATALOG}.{SCHEMA}.bns_parsed_raw')
    print('  ✅ ai_parse_document complete — persisted as parsed_json (STRING)')

    _elem_struct = StructType([
        StructField('type', StringType(), True),
        StructField('content', StringType(), True),
        StructField('page_number', StringType(), True),
    ])
    _doc_schema = StructType([
        StructField('elements', ArrayType(_elem_struct), True),
    ])

    elements_df = (
        parsed_df
        .select(from_json(col('parsed_json'), _doc_schema).alias('doc'))
        .select(explode(col('doc.elements')).alias('elem'))
        .select(
            col('elem.type').alias('element_type'),
            col('elem.content').alias('content'),
            col('elem.page_number').cast('int').alias('page_number'),
        )
        .filter(col('element_type').isin('TEXT', 'PARAGRAPH', 'HEADING'))
        .orderBy('page_number')
    )
    n_el = elements_df.count()
    print(f'  📄 Parsed elements: {n_el}')
    if n_el == 0:
        print('  ⚠️  0 elements — JSON shape may not match schema. Inspect:')
        print('     display(spark.table(f"{CATALOG}.{SCHEMA}.bns_parsed_raw").selectExpr("substring(parsed_json,1,1200)").limit(1))')
    (
        elements_df.write.format('delta')
        .mode('overwrite').option('overwriteSchema', 'true')
        .saveAsTable(f'{CATALOG}.{SCHEMA}.bns_parsed_elements')
    )
    print(f'  💾 bns_parsed_elements saved')

    all_rows   = elements_df.collect()
    full_text  = '\n'.join(
        r['content'] for r in all_rows
        if r['content'] and len(r['content'].strip()) > 3
    )
    print(f'  Full bilingual text: {len(full_text):,} chars')

    section_pat = re.compile(
        r'(?m)^(\d{1,3})\.\s+(.{2,150}?)(?=[\.—–\n])',
    )
    matches   = list(section_pat.finditer(full_text))
    seen_nums = set()
    raw_sections = []
    for i, m in enumerate(matches):
        num   = m.group(1).strip()
        title = m.group(2).strip().rstrip('.—– ')
        if num in seen_nums:
            continue
        body_start = m.end()
        body_end   = matches[i+1].start() if i+1 < len(matches) else body_start + 3000
        body       = re.sub(r'\s+', ' ', full_text[body_start:body_end]).strip()
        if len(body) < 10:
            continue
        seen_nums.add(num)
        raw_sections.append({
            'section_number': num,
            'section_title':  title,
            'section_text':   body[:3000],
            'source':         'BNS_2023',
        })
    raw_sections.sort(key=lambda x: int(x['section_number'])
                      if x['section_number'].isdigit() else 999)
    print(f'  Extracted {len(raw_sections)} sections from PDF text')

    def sarvam_enrich(section_number: str, section_text: str, retries: int = 2) -> dict:
        prompt = f"""You are a legal expert fluent in Hindi and English.
The following is BNS Section {section_number} text (may contain Hindi and English):

{section_text[:1500]}

Please respond ONLY with a JSON object with these exact keys:
{{
  "english_summary": "2-sentence plain English summary of what this section says",
  "hindi_explanation": "2-sentence explanation in Hindi (Devanagari script)",
  "key_terms": ["term1 (हिंदी)", "term2 (हिंदी)"],
  "ipc_equivalent": "IPC section number this replaced, or 'New provision'"
}}"""
        if not SARVAM_API_KEY:
            return {'english_summary': '', 'hindi_explanation': '', 'key_terms': [], 'ipc_equivalent': ''}
        for attempt in range(retries + 1):
            try:
                r = requests.post(
                    SARVAM_API_URL,
                    headers={'Authorization': f'Bearer {SARVAM_API_KEY}', 'Content-Type': 'application/json'},
                    json={'model': SARVAM_MODEL, 'messages': [{'role': 'user', 'content': prompt}],
                          'temperature': 0.1, 'max_tokens': 400},
                    timeout=30,
                )
                r.raise_for_status()
                content = r.json()['choices'][0]['message']['content']
                content = re.sub(r'^```json\s*|```$', '', content.strip(), flags=re.MULTILINE)
                return json.loads(content)
            except requests.HTTPError as he:
                if getattr(he.response, 'status_code', None) == 403:
                    print('  ⚠️  Sarvam 403 Forbidden — check API key / billing / model access.')
                return {'english_summary': '', 'hindi_explanation': '', 'key_terms': [], 'ipc_equivalent': ''}
            except Exception:
                if attempt < retries:
                    time.sleep(1)
                else:
                    return {'english_summary': '', 'hindi_explanation': '', 'key_terms': [], 'ipc_equivalent': ''}
        return {'english_summary': '', 'hindi_explanation': '', 'key_terms': [], 'ipc_equivalent': ''}

    enriched_sections = []
    ENRICH_LIMIT = min(50, len(raw_sections))
    if SARVAM_API_KEY and raw_sections:
        print('🔍 Sarvam enrichment (sample up to %d sections)...' % ENRICH_LIMIT)
        for i, sec in enumerate(raw_sections[:ENRICH_LIMIT]):
            enrichment = sarvam_enrich(sec['section_number'], sec['section_text'])
            enriched_sections.append({
                **sec,
                'english_summary':   enrichment.get('english_summary', ''),
                'hindi_explanation': enrichment.get('hindi_explanation', ''),
                'key_terms':         ', '.join(enrichment.get('key_terms', []) or []),
                'ipc_equivalent':    enrichment.get('ipc_equivalent', ''),
            })
            time.sleep(0.5)
            if (i+1) % 10 == 0:
                print(f'  Enriched {i+1}/{ENRICH_LIMIT}...')
        for sec in raw_sections[ENRICH_LIMIT:]:
            enriched_sections.append({**sec, 'english_summary': '', 'hindi_explanation': '', 'key_terms': '', 'ipc_equivalent': ''})
    else:
        enriched_sections = [{**s, 'english_summary': '', 'hindi_explanation': '', 'key_terms': '', 'ipc_equivalent': ''} for s in raw_sections]

    df_bns = pd.DataFrame(enriched_sections)
    print(f'  ✅ df_bns from PDF: {len(df_bns)} rows')


# COMMAND ----------

# MAGIC %md
# MAGIC ## 3d. Save `bns_sections` Delta table

# COMMAND ----------

if not df_bns.empty:
    display(df_bns.head(5))
    save_table(df_bns, 'bns_sections')
else:
    print('⚠️  No BNS data available.')
    print('  Option A — Download CSV from Kaggle:')
    print('    kaggle.com/datasets/nandr39/bharatiya-nyaya-sanhita-dataset-bns')
    print('  Option B — Upload to Volume and run the Manual upload helper below:')
    print('    load_uploaded_file("bns_sections.csv", "bns_sections")')

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. BNS ↔ IPC mapping table
# MAGIC
# MAGIC Source: **[nandhakumarg/IPC_and_BNS_transformation](https://huggingface.co/datasets/nandhakumarg/IPC_and_BNS_transformation)** on Hugging Face. If the download fails, the notebook uses a small built-in stub.
# MAGIC

# COMMAND ----------

print('📥 Fetching BNS ↔ IPC mapping...')
# Hugging Face — nandhakumarg/IPC_and_BNS_transformation (Apache-2.0)
# https://huggingface.co/datasets/nandhakumarg/IPC_and_BNS_transformation
import ast

HF_MAPPING_CSV = (
    'https://huggingface.co/datasets/nandhakumarg/IPC_and_BNS_transformation/'
    'resolve/main/IPC%20and%20BNS%20transformation%20.csv'
)

def _rows_from_hf(df_raw):
    rows = []
    for _, row in df_raw.iterrows():
        try:
            d = ast.literal_eval(row['response'])
        except Exception:
            continue
        ipc_sec = str(d.get('IPC Section', '')).strip()
        bns_sec = str(d.get('BNS Section', '')).strip()
        ipc_t = str(d.get('IPC Heading', '')).strip()
        bns_t = str(d.get('BNS Heading', '')).strip()
        if 'repealed' in bns_sec.lower() or 'repealed' in bns_t.lower():
            st = 'Repealed'
        elif 'repealed' in ipc_t.lower() and not ipc_sec:
            st = 'Repealed'
        else:
            st = 'Mapped'
        rows.append({
            'bns_section': bns_sec[:128],
            'ipc_section': ipc_sec[:64],
            'bns_title': bns_t[:4000],
            'ipc_title': ipc_t[:4000],
            'status': st,
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.drop_duplicates(subset=['ipc_section', 'bns_section'], keep='first')
    return out

df_mapping = pd.DataFrame()
try:
    r = requests.get(HF_MAPPING_CSV, timeout=120, headers=HEADERS)
    r.raise_for_status()
    raw = pd.read_csv(StringIO(r.text))
    df_mapping = _rows_from_hf(raw)
    print(f'  ✅ {len(df_mapping)} mapping rows from Hugging Face ({HF_MAPPING_CSV.split("/")[-1]})')
except Exception as e:
    print(f'  ⚠️  Hugging Face mapping failed: {e}')

if df_mapping.empty:
    df_mapping = pd.DataFrame([
        {'bns_section':'103','bns_title':'Murder',             'ipc_section':'302','ipc_title':'Punishment for murder',     'status':'Modified'},
        {'bns_section':'63', 'bns_title':'Rape',               'ipc_section':'375','ipc_title':'Rape',                      'status':'Modified'},
        {'bns_section':'111','bns_title':'Organised Crime',    'ipc_section':'None','ipc_title':'New section in BNS',       'status':'New'},
        {'bns_section':'113','bns_title':'Terrorist Act',      'ipc_section':'None','ipc_title':'New section in BNS',       'status':'New'},
        {'bns_section':'303','bns_title':'Theft',              'ipc_section':'378','ipc_title':'Theft',                    'status':'Retained'},
        {'bns_section':'318','bns_title':'Cheating',           'ipc_section':'415','ipc_title':'Cheating',                 'status':'Retained'},
        {'bns_section':'226','bns_title':'Attempt to Suicide', 'ipc_section':'309','ipc_title':'Attempt to commit suicide', 'status':'Modified'},
    ])
    print(f'  ✅ Using stub: {len(df_mapping)} rows')

save_table(df_mapping, 'bns_ipc_mapping')


# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Build `legal_rag_corpus`

# COMMAND ----------

SKIP_GOV_SCHEMES = True

corpus = []

# BNS sections (prefer english_summary for RAG when present)
try:
    df_b = spark.table(f'{CATALOG}.{SCHEMA}.bns_sections').toPandas()
    tc  = next((c for c in df_b.columns if 'text' in c.lower() or 'content' in c.lower()), df_b.columns[-1])
    nc  = next((c for c in df_b.columns if 'number' in c.lower() or c.lower() == 'section'), None)
    ttc = next((c for c in df_b.columns if 'title' in c.lower()), None)
    has_summary = 'english_summary' in df_b.columns
    for _, row in df_b.iterrows():
        num = str(row[nc]) if nc else ''
        if has_summary and pd.notna(row.get('english_summary')) and str(row.get('english_summary', '')).strip():
            body = str(row['english_summary'])[:2000]
        else:
            body = str(row[tc])[:2000]
        corpus.append({'chunk_id': f'BNS_S{num}', 'source': 'BNS_2023', 'doc_type': 'criminal_law',
                       'title': f'BNS Section {num}: {str(row[ttc]) if ttc else ""}',
                       'text': body})
    print(f'  ✅ BNS: {len(df_b)} sections')
except Exception as e:
    print(f'  ⚠️  BNS sections: {e}')

# BNS-IPC mapping
try:
    df_m = spark.table(f'{CATALOG}.{SCHEMA}.bns_ipc_mapping').toPandas()
    for _, row in df_m.iterrows():
        corpus.append({'chunk_id': f'MAP_BNS{row.get("bns_section","?")}', 'source': 'BNS_IPC_MAPPING',
                       'doc_type': 'law_mapping',
                       'title': f'BNS {row.get("bns_section","")} replaces IPC {row.get("ipc_section","")}',
                       'text': (f'BNS Section {row.get("bns_section","")} ({row.get("bns_title","")}) '
                                f'corresponds to IPC {row.get("ipc_section","")} ({row.get("ipc_title","")}). '
                                f'Status: {row.get("status","")}')})
    print(f'  ✅ Mapping: {len(df_m)} rows')
except Exception as e:
    print(f'  ⚠️  Mapping: {e}')

# Gov Schemes (skipped when SKIP_GOV_SCHEMES)
if not SKIP_GOV_SCHEMES:
    try:
        df_s  = spark.table(f'{CATALOG}.{SCHEMA}.gov_schemes_myscheme').toPandas()
        n_col = next((c for c in df_s.columns if 'name' in c.lower() or 'scheme' in c.lower()), df_s.columns[0])
        d_col = next((c for c in df_s.columns if 'desc' in c.lower()), None)
        e_col = next((c for c in df_s.columns if 'elig' in c.lower()), None)
        b_col = next((c for c in df_s.columns if 'bene' in c.lower()), None)
        for i, row in df_s.iterrows():
            parts = [f'Scheme: {row[n_col]}']
            if d_col and pd.notna(row.get(d_col)): parts.append(f'Description: {row[d_col]}')
            if e_col and pd.notna(row.get(e_col)): parts.append(f'Eligibility: {row[e_col]}')
            if b_col and pd.notna(row.get(b_col)): parts.append(f'Benefits: {row[b_col]}')
            corpus.append({'chunk_id': f'SCHEME_{i:04d}', 'source': 'MYSCHEME_GOV',
                           'doc_type': 'government_scheme', 'title': str(row[n_col]),
                           'text': ' | '.join(parts)[:2000]})
        print(f'  ✅ Schemes: {len(df_s)} rows')
    except Exception as e:
        print(f'  ⚠️  Schemes: {e}')

if corpus:
    df_corpus = pd.DataFrame(corpus)
    save_table(df_corpus, 'legal_rag_corpus')
    print(f'\n🏛️  legal_rag_corpus: {len(df_corpus)} total chunks')
    display(df_corpus.groupby('source')['chunk_id'].count().reset_index(name='chunks'))
else:
    print('⚠️  No chunks — run sections 9 and 10 first.')

# COMMAND ----------

# MAGIC %md
# MAGIC ## 12. Manual upload helper
# MAGIC Upload files via **Catalog → Volumes → legal_files → Upload to this volume**,
# MAGIC then call `load_uploaded_file(filename, table_name)` below.

# COMMAND ----------

def load_uploaded_file(filename, table_name, fmt='csv', skip_rows=0):
    """
    Load a file from Unity Catalog Volume into a Delta table.
    Upload path: Catalog → main → india_legal → Volumes → legal_files → Upload

    Args:
        filename   : just the filename, e.g. 'daksh_hc.csv'
        table_name : Delta table name (no schema prefix)
        fmt        : 'csv' | 'excel' | 'parquet'
        skip_rows  : header rows to skip (NCRB Excel needs skip_rows=3)
    """
    path = f'{VOL_PATH}/{filename}'
    print(f'📂 Loading {path} → {CATALOG}.{SCHEMA}.{table_name}')
    try:
        if fmt == 'csv':
            pdf = spark.read.option('header','true').option('inferSchema','false') \
                       .option('multiLine','true').csv(path).toPandas()
        elif fmt == 'excel':
            pdf = pd.read_excel(path, engine='openpyxl', skiprows=skip_rows)
            pdf.dropna(how='all', inplace=True)
            pdf.dropna(axis=1, how='all', inplace=True)
        elif fmt == 'parquet':
            pdf = spark.read.parquet(path).toPandas()
        else:
            raise ValueError(f'Unsupported format: {fmt}')
        save_table(pdf, table_name)
    except Exception as e:
        print(f'  ⚠️  {e}')

# Examples — uncomment after uploading:
# load_uploaded_file('daksh_hc_cases.csv',  'daksh_hc_cases')
# load_uploaded_file('njdg_state.csv',      'njdg_state_pendency')
# load_uploaded_file('ncrb_ipc_2022.xlsx',  'ncrb_ipc_2022', fmt='excel', skip_rows=3)
# load_uploaded_file('bns_sections.csv',    'bns_sections')
print('✅ load_uploaded_file() ready')

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Verify tables

# COMMAND ----------

print(f'📋 All tables in {CATALOG}.{SCHEMA}:')
display(spark.sql(f'SHOW TABLES IN {CATALOG}.{SCHEMA}'))

# COMMAND ----------

display(spark.sql(f'''
    SELECT chunk_id, source, LEFT(text, 200) AS preview
    FROM   {CATALOG}.{SCHEMA}.legal_rag_corpus
    ORDER  BY source, chunk_id
    LIMIT  20
'''))

# COMMAND ----------

display(spark.sql(f'''
    SELECT bns_section, ipc_section, bns_title, ipc_title, status
    FROM   {CATALOG}.{SCHEMA}.bns_ipc_mapping
    LIMIT  25
'''))