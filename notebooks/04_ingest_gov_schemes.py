# Databricks notebook source
# MAGIC %md
# MAGIC # Notebook 04: Government Schemes Ingestion + Spark TF-IDF
# MAGIC
# MAGIC Loads Indian government welfare schemes from **CSV or PDF files** stored in the UC Volume,
# MAGIC computes Spark DataFrame-based TF-IDF features, and upserts scheme chunks into `legal_rag_corpus`.
# MAGIC
# MAGIC ## Data Sources (tried in order)
# MAGIC
# MAGIC | Priority | Source | How to set up |
# MAGIC |----------|--------|--------------|
# MAGIC | 1 | **CSV in UC Volume** | Upload your CSV to the path in `SCHEMES_CSV_PATH` below |
# MAGIC | 2 | **PDF in UC Volume** | Upload a schemes PDF to `SCHEMES_PDF_PATH` below |
# MAGIC | 3 | **data.gov.in open API** | Auto-downloaded (no setup needed, requires internet) |
# MAGIC | 4 | **Hardcoded stubs** | 50+ major central schemes, always available |
# MAGIC
# MAGIC ## Recommended CSV Download
# MAGIC
# MAGIC Download scheme CSVs from the **official Government of India open data portal**:
# MAGIC - https://data.gov.in/catalog/schemes-various-ministries-or-departments
# MAGIC - https://myscheme.gov.in  (use browser export / download option)
# MAGIC
# MAGIC **Expected CSV columns** (flexible — notebook auto-maps common column names):
# MAGIC `scheme_name, ministry, state, description, eligibility, benefits`
# MAGIC Optional: `min_age, max_age, gender, income_limit_inr, caste_category, occupation_tags`
# MAGIC
# MAGIC Upload the CSV to:
# MAGIC `/Volumes/workspace/india_legal/legal_files/gov_schemes.csv`

# COMMAND ----------

import os
import subprocess
import sys

REPO_ROOT = "/Workspace/Users/11avanshdhar@gmail.com/nyaya-dhwani-hackathon"  # Edit me

_src = os.path.join(REPO_ROOT, "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    "pandas>=2.0,<3",
    "pymupdf>=1.23.0",   # PDF parsing
    "requests>=2.28.0",
])
print("✅ dependencies installed")

# COMMAND ----------

# ── Configuration — EDIT THIS CELL ───────────────────────────────────────────
CATALOG = "workspace"
SCHEMA  = "india_legal"

SCHEMES_TABLE  = f"{CATALOG}.{SCHEMA}.gov_welfare_schemes"
TFIDF_TABLE    = f"{CATALOG}.{SCHEMA}.scheme_tfidf_scores"
CORPUS_TABLE   = f"{CATALOG}.{SCHEMA}.legal_rag_corpus"

# ┌─────────────────────────────────────────────────────────────────────────┐
# │  SET YOUR CSV PATH HERE                                                 │
# │                                                                         │
# │  Upload your CSV to the UC Volume and set the path below.              │
# │  To find the path: Catalog Explorer → workspace → india_legal →        │
# │  legal_files → right-click file → Copy path                            │
# │                                                                         │
# │  The notebook auto-detects the my_gov_schemes.csv format               │
# │  (columns: scheme_name, slug, details, benefits, eligibility,          │
# │   application, documents, level, schemeCategory, tags)                 │
# └─────────────────────────────────────────────────────────────────────────┘
VOLUME_ROOT      = f"/Volumes/{CATALOG}/{SCHEMA}/legal_files"
SCHEMES_CSV_PATH = f"{VOLUME_ROOT}/my_gov_schemes.csv"  # ← default; change if filename differs
SCHEMES_PDF_PATH = f"{VOLUME_ROOT}/gov_schemes.pdf"

# data.gov.in open API (auto-tried if CSV/PDF not found)
DATAGOV_API_URL  = (
    "https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070"
    "?api-key=579b464db66ec23bdd000001cdd3946e44ce4aab825ef8952f8e3b9a"
    "&format=csv&limit=1000"
)

import pandas as pd

# COMMAND ----------

# MAGIC %md ## Step 1: Load Scheme Data

# COMMAND ----------

# ── Required output columns ───────────────────────────────────────────────────
_REQUIRED_COLS = [
    "scheme_id", "scheme_name", "ministry", "state", "description",
    "eligibility", "benefits", "min_age", "max_age", "gender",
    "income_limit_inr", "caste_category", "occupation_tags",
]

# ── Eligibility field parser (regex-based) ────────────────────────────────────
import re as _re

def _extract_age(text: str) -> tuple[str, str]:
    """Extract (min_age, max_age) from eligibility text."""
    # "age group of 18-60 years", "18 to 60 years", "aged 18-60"
    m = _re.search(r"(\d{1,3})\s*[-–to]+\s*(\d{1,3})\s*years", text, _re.I)
    if m:
        return m.group(1), m.group(2)
    # "above 18 years", "minimum age.*18"
    m = _re.search(r"(?:above|minimum age[^0-9]*|at least)\s*(\d{1,3})\s*years?", text, _re.I)
    if m:
        return m.group(1), "99"
    # "below 60 years", "not exceeded 60"
    m = _re.search(r"(?:below|under|not exceed(?:ing)?)\s*(\d{1,3})\s*years?", text, _re.I)
    if m:
        return "0", m.group(1)
    # single age "18 years" with context word
    m = _re.search(r"age[^0-9]*(\d{1,3})\s*years?", text, _re.I)
    if m:
        return m.group(1), "99"
    return "", ""

def _extract_gender(text: str) -> str:
    t = text.lower()
    has_f = any(w in t for w in ("women", "female", "girl", "widow", "mother"))
    has_m = any(w in t for w in ("men ", "male", " boy", "father", " man "))
    if has_f and not has_m:
        return "F"
    if has_m and not has_f:
        return "M"
    return "ALL"

def _extract_income(text: str) -> str:
    """Extract annual income limit in INR (returns empty string if not found)."""
    # "Rs 2,00,000", "₹2,00,000", "Rs. 2 lakh", "₹ 2.5 lakh"
    lakh_m = _re.search(r"(?:Rs\.?|₹)\s*([\d.]+)\s*lakh", text, _re.I)
    if lakh_m:
        try:
            return str(int(float(lakh_m.group(1)) * 100000))
        except ValueError:
            pass
    crore_m = _re.search(r"(?:Rs\.?|₹)\s*([\d.]+)\s*crore", text, _re.I)
    if crore_m:
        try:
            return str(int(float(crore_m.group(1)) * 10000000))
        except ValueError:
            pass
    # "Rs 2,00,000" or "₹2,00,000"
    num_m = _re.search(r"(?:Rs\.?|₹)\s*([\d,]+)", text, _re.I)
    if num_m:
        try:
            return str(int(num_m.group(1).replace(",", "")))
        except ValueError:
            pass
    return ""

def _extract_caste(text: str) -> str:
    t = text.lower()
    if any(w in t for w in ("scheduled caste", " sc ", "sc community", "dalit")):
        return "SC"
    if any(w in t for w in ("scheduled tribe", " st ", "st community", "tribal", "adivasi")):
        return "ST"
    if any(w in t for w in ("other backward", " obc ", "obc community")):
        return "OBC"
    if any(w in t for w in ("minority", "muslim", "christian", "sikh", "buddhist", "jain", "parsi")):
        return "Minority"
    return "ALL"

def _category_to_occupation(category: str, tags: str) -> str:
    """Map schemeCategory + tags to an occupation_tags string."""
    parts = []
    combined = (category + " " + tags).lower()
    mapping = {
        "farmer": "Farmer", "agricultur": "Farmer", "kisan": "Farmer",
        "student": "Student", "education": "Student", "scholarship": "Student",
        "labour": "Labour/Worker", "worker": "Labour/Worker", "construction": "Labour/Worker",
        "entrepreneur": "Self-employed", "business": "Self-employed", "msme": "Self-employed",
        "women": "Women", "child": "Women",
        "senior": "Senior Citizen", "old age": "Senior Citizen", "pension": "Senior Citizen",
        "disab": "Differently-abled",
        "bpl": "BPL", "poverty": "BPL",
    }
    seen = set()
    for keyword, tag in mapping.items():
        if keyword in combined and tag not in seen:
            parts.append(tag)
            seen.add(tag)
    return ",".join(parts) if parts else ""


# ── Loader 1: my_gov_schemes.csv format (primary) ────────────────────────────
def _load_myscheme_csv(path: str) -> pd.DataFrame | None:
    """Parse the my_gov_schemes.csv format:
    Columns: scheme_name, slug, details, benefits, eligibility,
             application, documents, level, schemeCategory, tags
    """
    if not os.path.exists(path):
        print(f"ℹ️  No CSV found at {path}")
        return None
    try:
        raw = pd.read_csv(path, encoding="utf-8", on_bad_lines="skip")
        print(f"✅ Loaded CSV: {len(raw)} rows | columns: {list(raw.columns)}")
    except Exception as e:
        print(f"⚠️  CSV load failed: {e}")
        return None

    raw = raw.fillna("")

    # Detect if this is the my_gov_schemes format
    is_myscheme_fmt = "slug" in raw.columns and "details" in raw.columns
    if not is_myscheme_fmt:
        print("  → Generic CSV detected (not my_gov_schemes format) — using generic normaliser")
        return _load_generic_csv_df(raw)

    print("  → my_gov_schemes.csv format detected — using dedicated parser")
    rows = []
    for _, r in raw.iterrows():
        eligibility_text = str(r.get("eligibility", ""))
        category         = str(r.get("schemeCategory", ""))
        tags             = str(r.get("tags", ""))
        level            = str(r.get("level", "")).strip()

        min_age, max_age = _extract_age(eligibility_text)
        gender           = _extract_gender(eligibility_text)
        income           = _extract_income(eligibility_text)
        caste            = _extract_caste(eligibility_text)
        occ_tags         = _category_to_occupation(category, tags)

        # state: Central schemes → ALL; State schemes keep "State-specific"
        state = "ALL" if level.lower() == "central" else "State-specific"

        rows.append({
            "scheme_id":        str(r.get("slug", ""))[:60],
            "scheme_name":      str(r.get("scheme_name", "")).strip(),
            "ministry":         category,           # best proxy available
            "state":            state,
            "description":      str(r.get("details", "")).strip(),
            "eligibility":      eligibility_text.strip(),
            "benefits":         str(r.get("benefits", "")).strip(),
            "min_age":          min_age,
            "max_age":          max_age,
            "gender":           gender,
            "income_limit_inr": income,
            "caste_category":   caste,
            "occupation_tags":  occ_tags,
            # extra fields kept for richer corpus chunks
            "_application":     str(r.get("application", "")).strip(),
            "_documents":       str(r.get("documents",   "")).strip(),
            "_tags":            tags,
        })

    df = pd.DataFrame(rows)
    df = df[df["scheme_name"].str.strip().str.len() > 2].reset_index(drop=True)
    print(f"  → Parsed {len(df)} valid schemes")
    print(f"  → Gender breakdown: {df['gender'].value_counts().to_dict()}")
    print(f"  → Caste breakdown:  {df['caste_category'].value_counts().to_dict()}")
    print(f"  → Income extracted: {(df['income_limit_inr'] != '').sum()} schemes")
    print(f"  → Age extracted:    {(df['min_age'] != '').sum()} schemes")
    return df


# ── Loader 2: Generic CSV (any other format) ──────────────────────────────────
_COL_MAP = {
    "Scheme Name": "scheme_name", "name": "scheme_name", "Name": "scheme_name",
    "Ministry": "ministry", "Department": "ministry", "department": "ministry",
    "State": "state", "applicable_state": "state",
    "Description": "description", "About": "description", "details": "description",
    "Eligibility": "eligibility", "eligibility_criteria": "eligibility",
    "Benefits": "benefits", "Benefit": "benefits", "benefit": "benefits",
}

def _load_generic_csv_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={k: v for k, v in _COL_MAP.items() if k in df.columns})
    for col in _REQUIRED_COLS:
        if col not in df.columns:
            df[col] = ""
    if df["scheme_id"].eq("").all():
        df["scheme_id"] = ["SCH_" + str(i).zfill(5) for i in range(len(df))]
    df = df[_REQUIRED_COLS].fillna("").astype(str)
    return df[df["scheme_name"].str.strip().str.len() > 2].reset_index(drop=True)


# ── Loader 2: PDF from UC Volume ─────────────────────────────────────────────
def _load_pdf(path: str) -> pd.DataFrame | None:
    if not os.path.exists(path):
        print(f"ℹ️  No PDF found at {path}")
        return None
    try:
        import fitz  # pymupdf
        doc = fitz.open(path)
        rows = []
        current: dict = {}
        for page in doc:
            text = page.get_text()
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    if current.get("scheme_name"):
                        rows.append(current.copy())
                    current = {}
                    continue
                # Heuristic: lines starting with a number or capital word = scheme name
                if not current.get("scheme_name"):
                    current["scheme_name"] = line
                elif not current.get("description"):
                    current["description"] = line
                else:
                    current["description"] = current.get("description", "") + " " + line
        if current.get("scheme_name"):
            rows.append(current)
        doc.close()
        if not rows:
            print("⚠️  PDF parsed but no scheme rows extracted")
            return None
        df = pd.DataFrame(rows)
        print(f"✅ Parsed PDF {path}: {len(df)} scheme blocks extracted")
        return _normalise(df)
    except Exception as e:
        print(f"⚠️  PDF load failed: {e}")
        return None


# ── Loader 3: data.gov.in open API ───────────────────────────────────────────
def _load_datagov() -> pd.DataFrame | None:
    import requests, io
    try:
        print("⏳ Trying data.gov.in open API …")
        r = requests.get(DATAGOV_API_URL, timeout=30)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text), on_bad_lines="skip")
        if df.empty:
            return None
        print(f"✅ data.gov.in API: {len(df)} rows, columns: {list(df.columns)}")
        return _normalise(df)
    except Exception as e:
        print(f"⚠️  data.gov.in API failed: {e}")
        return None


# ── Loader 4: Expanded hardcoded stubs (50+ schemes) ─────────────────────────
def _hardcoded_stubs() -> pd.DataFrame:
    print("ℹ️  Using expanded hardcoded stubs (50 major central schemes)")
    stubs = [
        # Agriculture & Farmers
        ("PMKISAN001",  "PM-KISAN",                              "Ministry of Agriculture & Farmers Welfare",   "ALL", "Direct income support of Rs 6,000/year to farmer families owning cultivable land.", "Small and marginal farmers with land holding up to 2 hectares. Must be in PM-KISAN beneficiary list.", "Rs 6,000 per year in 3 installments of Rs 2,000 directly to bank account.", "18","99","ALL","","ALL","Farmer"),
        ("PMFBY001",   "Pradhan Mantri Fasal Bima Yojana",      "Ministry of Agriculture & Farmers Welfare",   "ALL", "Crop insurance scheme to provide financial support to farmers suffering crop loss.", "All farmers including sharecroppers and tenant farmers growing notified crops.", "Full claim for crop loss due to natural calamities, pest, disease. Premium: 2% Kharif, 1.5% Rabi.", "18","99","ALL","","ALL","Farmer"),
        ("KCC001",     "Kisan Credit Card (KCC)",               "Ministry of Agriculture & Farmers Welfare",   "ALL", "Short-term credit for farmers to meet agriculture and allied activity needs.", "All farmers, fishermen, animal husbandry workers.", "Revolving credit up to Rs 3 lakh at 7% interest (4% effective with subvention).", "18","99","ALL","","ALL","Farmer"),
        ("PMKSY001",   "Pradhan Mantri Krishi Sinchai Yojana",  "Ministry of Jal Shakti",                      "ALL", "Expands cultivable area under irrigation; improves water use efficiency.", "Farmers in water-scarce areas, land development, watershed management.", "Drip/sprinkler irrigation subsidy up to 55% for small/marginal farmers.", "18","99","ALL","","ALL","Farmer"),
        ("PKVY001",    "Paramparagat Krishi Vikas Yojana",      "Ministry of Agriculture & Farmers Welfare",   "ALL", "Promotes organic farming through cluster approach.", "Farmers willing to adopt organic farming in clusters of 50 acres.", "Rs 50,000 per hectare over 3 years for organic inputs, certification, and marketing.", "18","99","ALL","","ALL","Farmer"),

        # Housing
        ("PMAY_U001",  "Pradhan Mantri Awas Yojana – Urban",    "Ministry of Housing and Urban Affairs",       "ALL", "Affordable housing for urban poor through interest subsidy and direct grant.", "Families with annual income below Rs 18 lakh without pucca house.", "Interest subsidy up to Rs 2.67 lakh on home loan; EWS/LIG/MIG categories.", "18","60","ALL","1800000","ALL","BPL,Labour/Worker,Self-employed"),
        ("PMAY_G001",  "Pradhan Mantri Awas Yojana – Gramin",   "Ministry of Rural Development",               "ALL", "Housing for homeless rural poor and those living in kutcha houses.", "SECC 2011 listed homeless rural households and kutcha house dwellers.", "Rs 1.20 lakh (plain areas) or Rs 1.30 lakh (hilly/NE) directly to bank account.", "18","60","ALL","","ALL","BPL,Labour/Worker"),

        # Health
        ("PMJAY001",   "Ayushman Bharat PM-JAY",                "Ministry of Health and Family Welfare",       "ALL", "Health insurance of Rs 5 lakh per family per year for secondary and tertiary care.", "SECC 2011 database families, BPL households, SC/ST, construction workers.", "Cashless treatment up to Rs 5 lakh/year at empanelled public and private hospitals.", "0","99","ALL","300000","ALL","BPL,Farmer,Labour/Worker"),
        ("JSY001",     "Janani Suraksha Yojana",                "Ministry of Health and Family Welfare",       "ALL", "Safe motherhood intervention for institutional delivery among poor pregnant women.", "Pregnant women below poverty line in low-performing states.", "Rs 1,400 (rural) or Rs 1,000 (urban) cash incentive for institutional delivery.", "0","45","F","","ALL","BPL"),
        ("PMMVY001",   "Pradhan Mantri Matru Vandana Yojana",   "Ministry of Women and Child Development",     "ALL", "Maternity benefit for first live birth to partially compensate wage loss.", "Pregnant and lactating women for first live birth, excluding government employees.", "Rs 5,000 in three installments directly to bank account.", "19","99","F","","ALL","Labour/Worker,Self-employed"),
        ("AB_HWC001",  "Ayushman Bharat – Health and Wellness Centres", "Ministry of Health and Family Welfare","ALL", "Comprehensive primary healthcare at village level.", "All citizens in rural areas near Health and Wellness Centres.", "Free medicines, diagnostics, tele-consultation, maternal and child health services.", "0","99","ALL","","ALL",""),

        # Education & Scholarships
        ("NSP_SC001",  "Post Matric Scholarship – SC",          "Ministry of Social Justice and Empowerment",  "ALL", "Scholarships for SC students studying at post-matric level.", "SC students with annual family income below Rs 2.5 lakh.", "Maintenance allowance + tuition fee + study allowance per month.", "15","35","ALL","250000","SC","Student"),
        ("NSP_ST001",  "Post Matric Scholarship – ST",          "Ministry of Tribal Affairs",                  "ALL", "Scholarships for ST students studying at post-matric level.", "ST students with annual family income below Rs 2.5 lakh.", "Maintenance allowance + tuition fee reimbursement.", "15","35","ALL","250000","ST","Student"),
        ("NSP_OBC001", "Post Matric Scholarship – OBC",         "Ministry of Social Justice and Empowerment",  "ALL", "Scholarships for OBC students studying at post-matric level.", "OBC students with annual family income below Rs 1 lakh.", "Maintenance allowance and other fees as per rates.", "15","35","ALL","100000","OBC","Student"),
        ("NSP_MIN001", "Pre/Post Matric Scholarship – Minority","Ministry of Minority Affairs",                "ALL", "Scholarships for students from minority communities.", "Students from Muslim, Christian, Sikh, Buddhist, Jain, Parsi communities. Income below Rs 2 lakh.", "Tuition fee up to Rs 10,000 and maintenance allowance.", "0","35","ALL","200000","ALL","Student"),
        ("PMSSS001",   "PM Special Scholarship – J&K",          "Ministry of Education",                       "Jammu and Kashmir", "Scholarships for students from J&K to study in reputed institutions across India.", "Students from J&K who have passed Class 12. Family income below Rs 4.5 lakh.", "Rs 30,000/year for hostel + Rs 1 lakh/year for engineering/medical courses.", "15","25","ALL","450000","ALL","Student"),
        ("INSPIRE001", "INSPIRE Scholarship",                   "Department of Science and Technology",        "ALL", "Attract students to science education and research.", "Top 1% students in Class 12 from all boards pursuing natural/basic sciences.", "Rs 60,000/year for graduation + Rs 80,000/year for post-graduation.", "15","25","ALL","","ALL","Student"),
        ("NMMSS001",   "National Means-cum-Merit Scholarship",  "Ministry of Education",                       "ALL", "Scholarship for meritorious students of economically weaker sections.", "Class 8 students with family income below Rs 1.5 lakh/year.", "Rs 12,000/year from Class 9 to Class 12.", "13","18","ALL","150000","ALL","Student"),

        # Employment & Livelihood
        ("NREGA001",   "MGNREGA",                               "Ministry of Rural Development",               "ALL", "Guarantees 100 days of wage employment per year to rural households.", "Any adult member of rural household willing to do unskilled manual work.", "Minimum wage for up to 100 days/year; work within 5 km radius; unemployment allowance if work not provided.", "18","99","ALL","","ALL","Labour/Worker,BPL,Farmer"),
        ("PMEGP001",   "PM Employment Generation Programme",    "Ministry of MSME",                            "ALL", "Credit-linked subsidy for setting up new micro enterprises.", "Any individual above 18 years; SHGs, charitable trusts, cooperative societies.", "Subsidy 15–35% of project cost (up to Rs 25 lakh manufacturing, Rs 10 lakh service).", "18","55","ALL","","ALL","Self-employed"),
        ("PMMY001",    "Pradhan Mantri MUDRA Yojana",           "Ministry of Finance",                         "ALL", "Collateral-free loans to micro/small businesses.", "Non-corporate, non-farm small/micro enterprises. No existing bank default.", "Shishu: up to Rs 50K; Kishor: Rs 50K–5L; Tarun: Rs 5L–10L at concessional rates.", "18","65","ALL","","ALL","Self-employed,Labour/Worker"),
        ("STARTUP001", "Startup India Seed Fund",               "Department for Promotion of Industry and Internal Trade","ALL","Financial assistance for proof of concept and prototype development.", "DPIIT-recognised startups with innovative business idea, incorporated after April 2016.", "Up to Rs 20 lakh for POC; up to Rs 50 lakh for commercialisation.", "21","45","ALL","","ALL","Self-employed"),
        ("SVNIDHI001", "PM SVANidhi",                           "Ministry of Housing and Urban Affairs",       "ALL", "Micro-credit for street vendors to restart livelihoods post-COVID.", "Urban street vendors with Certificate of Vending or Letter of Recommendation from ULB.", "Rs 10,000 → Rs 20,000 → Rs 50,000 credit on timely repayment.", "18","65","ALL","","ALL","Self-employed,Labour/Worker"),
        ("NAPS001",    "National Apprenticeship Promotion Scheme","Ministry of Skill Development",             "ALL", "Promotes apprenticeship training to increase availability of skilled workers.", "School/college dropouts; ITI graduates; any youth wanting skill training.", "Government shares 25% of stipend (up to Rs 1,500/month) + basic training cost.", "14","35","ALL","","ALL","Student,Labour/Worker"),

        # Women & Child
        ("BBBP001",    "Beti Bachao Beti Padhao",               "Ministry of Women and Child Development",     "ALL", "Addresses declining child sex ratio; promotes education and welfare of the girl child.", "Girl children, especially in districts with low sex ratio.", "Conditional cash benefits for education, scholarship support, awareness drives.", "0","21","F","","ALL","Student"),
        ("SSA001",     "Sukanya Samriddhi Yojana",              "Ministry of Finance",                         "ALL", "Small savings scheme for education and marriage of girl child.", "Girl child below 10 years; account opened by parent/guardian.", "High interest rate (currently ~8.2% p.a.); deposits up to Rs 1.5 lakh/year; tax exemption under 80C.", "0","10","F","","ALL",""),
        ("WEAVERS001", "Mahila Shakti Kendra",                  "Ministry of Women and Child Development",     "ALL", "Empowers rural women through community participation and awareness.", "Rural women; focus on villages with population of 100+.", "Skill development, digital literacy, nutrition, health, legal rights awareness.", "18","60","F","","ALL","Labour/Worker,Self-employed"),
        ("IGMSY001",   "Indira Gandhi Matritva Sahyog Yojana",  "Ministry of Women and Child Development",     "ALL", "Conditional maternity benefit for better health and nutrition of pregnant women.", "Pregnant women aged 19+ for first two live births excluding government employees.", "Rs 6,000 in installments on meeting health and immunisation conditions.", "19","45","F","","ALL","Labour/Worker,BPL"),

        # Senior Citizens & Pension
        ("IGNOAPS001", "Indira Gandhi National Old Age Pension","Ministry of Rural Development",               "ALL", "Monthly pension for destitute elderly above 60 years below poverty line.", "BPL persons aged 60 and above.", "Rs 200/month (60–79 years); Rs 500/month (80+ years) from centre + state top-up.", "60","99","ALL","","ALL","BPL"),
        ("NFBS001",    "National Family Benefit Scheme",        "Ministry of Rural Development",               "ALL", "Lump sum family benefit on death of primary breadwinner.", "BPL households where breadwinner aged 18–64 dies due to natural or accidental causes.", "Rs 20,000 lump sum to bereaved household.", "0","99","ALL","","ALL","BPL"),
        ("VARISHTHA001","Varishtha Pension Bima Yojana",        "Ministry of Finance",                         "ALL", "Pension scheme for citizens aged 60+ providing assured return.", "Senior citizens aged 60 years and above.", "Assured 7.4% return p.a.; pension Rs 500–Rs 5,000/month depending on purchase price.", "60","99","ALL","","ALL",""),

        # Disability
        ("ADIP001",    "Assistance to Disabled Persons (ADIP)","Ministry of Social Justice and Empowerment",  "ALL", "Assistive devices and aids for persons with disabilities.", "Persons with disability with income below Rs 20,000/month.", "Free/subsidised hearing aids, wheelchairs, crutches, tricycles, braille kits.", "0","99","ALL","240000","ALL","Differently-abled"),
        ("NHFDC001",   "NHFDC Loan Scheme",                    "Ministry of Social Justice and Empowerment",  "ALL", "Concessional loans for self-employment and education of disabled persons.", "Persons with 40%+ disability.", "Loans up to Rs 50 lakh at 5–8% interest through state channelising agencies.", "18","55","ALL","","ALL","Differently-abled,Self-employed"),

        # Financial Inclusion
        ("PMJDY001",   "Pradhan Mantri Jan Dhan Yojana",       "Ministry of Finance",                         "ALL", "Universal access to banking, savings and deposit accounts, remittance, credit.", "Any unbanked Indian citizen, especially rural and economically weaker sections.", "Zero-balance account, RuPay debit card, Rs 2 lakh accident insurance, Rs 30,000 life cover.", "18","99","ALL","","ALL","BPL,Labour/Worker,Farmer"),
        ("PMJJBY001",  "PM Jeevan Jyoti Bima Yojana",          "Ministry of Finance",                         "ALL", "Life insurance cover for death due to any reason.", "Bank account holders aged 18–50 years.", "Rs 2 lakh life cover at Rs 330/year premium.", "18","50","ALL","","ALL",""),
        ("PMSBY001",   "PM Suraksha Bima Yojana",              "Ministry of Finance",                         "ALL", "Accident insurance cover for death/disability.", "Bank account holders aged 18–70 years.", "Rs 2 lakh for accidental death/total disability; Rs 1 lakh partial disability at Rs 12/year.", "18","70","ALL","","ALL",""),
        ("APY001",     "Atal Pension Yojana",                  "Ministry of Finance",                         "ALL", "Pension scheme for unorganised sector workers.", "Citizens aged 18–40 years with savings bank account; not income tax payee.", "Guaranteed pension of Rs 1,000–5,000/month at age 60 depending on contribution.", "18","40","ALL","","ALL","Labour/Worker,Self-employed,Farmer"),

        # Rural Development
        ("PMGSY001",   "PM Gram Sadak Yojana",                 "Ministry of Rural Development",               "ALL", "All-weather road connectivity to unconnected rural habitations.", "Unconnected rural habitations with population 500+ (250+ in hilly/tribal areas).", "Construction of all-weather roads to unconnected villages.", "0","99","ALL","","ALL",""),
        ("JJM001",     "Jal Jeevan Mission",                   "Ministry of Jal Shakti",                      "ALL", "Provide safe and adequate drinking water through household tap connections.", "Rural households without piped water supply.", "Functional household tap connection at minimum 55 LPCD.", "0","99","ALL","","ALL","BPL,Farmer,Labour/Worker"),
        ("DDUGJY001",  "Deen Dayal Upadhyaya Gram Jyoti Yojana","Ministry of Power",                         "ALL", "Electrification of all villages and un-electrified households.", "Unelectrified rural households and villages.", "Free electricity connection for BPL households; infrastructure for all villages.", "0","99","ALL","","ALL","BPL,Farmer"),
        ("SBM_G001",   "Swachh Bharat Mission – Gramin",       "Ministry of Jal Shakti",                      "ALL", "Accelerate sanitation coverage in rural India; eliminate open defecation.", "Rural households without toilets; focus on BPL and SC/ST families.", "Incentive of Rs 12,000 for construction of household toilet.", "0","99","ALL","","ALL","BPL,Farmer,Labour/Worker"),

        # Minority Welfare
        ("SEEKHO001",  "Seekho aur Kamao",                     "Ministry of Minority Affairs",                "ALL", "Skill development scheme for youth from minority communities.", "Youth from minority communities aged 14–45 years.", "Free market-oriented skill training with placement support; stipend during training.", "14","45","ALL","","ALL","Student,Labour/Worker"),
        ("NMDFC001",   "NMDFC Loan Scheme",                    "Ministry of Minority Affairs",                "ALL", "Concessional credit to minorities for income-generating activities.", "Minorities with annual family income below Rs 1.5 lakh (rural) or Rs 2 lakh (urban).", "Loans at 6% interest up to Rs 30 lakh through state channelising agencies.", "18","55","ALL","200000","ALL","Self-employed,Labour/Worker"),

        # Digital & Technology
        ("PMGDISHA001","PM Gramin Digital Saksharta Abhiyan",  "Ministry of Electronics and IT",              "ALL", "Make 6 crore rural households digitally literate.", "One member per rural household who is not digitally literate.", "Free digital literacy training covering internet, mobile banking, government e-services.", "14","60","ALL","","ALL",""),
        ("CSC001",     "Common Service Centres Scheme",        "Ministry of Electronics and IT",              "ALL", "Delivery of government and private services in rural and remote areas.", "Rural citizens needing access to government services.", "Access to 300+ services: Aadhaar, PAN, banking, insurance, certificates at doorstep.", "0","99","ALL","","ALL",""),

        # Self Help Groups & Women Entrepreneurship
        ("NRLM001",    "Deendayal Antyodaya Yojana – NRLM",   "Ministry of Rural Development",               "ALL", "Alleviate rural poverty through strong institutions of poor, especially women.", "Rural poor households with focus on women; SC/ST and minority communities prioritised.", "SHG formation, revolving fund Rs 10,000–15,000, bank linkage, skill training, livelihood support.", "18","60","F","","ALL","BPL,Labour/Worker,Farmer"),
        ("NULM001",    "Deendayal Antyodaya Yojana – NULM",   "Ministry of Housing and Urban Affairs",       "ALL", "Reduce urban poverty through self-employment and skill development.", "Urban poor including street vendors, rag pickers, construction workers.", "Skill training, credit at 7% interest, SHG support, shelter for homeless.", "18","60","ALL","","ALL","BPL,Labour/Worker,Self-employed"),

        # Tribal Welfare
        ("VGF001",     "Van Dhan Vikas Kendra",                "Ministry of Tribal Affairs",                  "ALL", "Creating livelihoods by value addition to Minor Forest Produce.", "Tribal gatherers and artisans in forest areas.", "Setting up 300 tribal members per VDVK for MFP processing; working capital support.", "18","65","ALL","","ST","Farmer,Labour/Worker"),
        ("EMRS001",    "Eklavya Model Residential Schools",    "Ministry of Tribal Affairs",                  "ALL", "Quality education to tribal students in remote areas.", "ST students in sub-districts with 50%+ ST population.", "Free residential education with state-of-the-art infrastructure; Rs 1.09 lakh/student/year.", "6","18","ALL","","ST","Student"),

        # MSME & Industry
        ("ZED001",     "ZED Certification Scheme",             "Ministry of MSME",                            "ALL", "Promote Zero Defect Zero Effect manufacturing among MSMEs.", "All MSMEs in manufacturing sector.", "Subsidy up to 80% on certification cost (Rs 3 lakh for micro, Rs 5 lakh for small).", "18","99","ALL","","ALL","Self-employed"),
        ("CGTMSE001",  "CGTMSE – Credit Guarantee Scheme",     "Ministry of MSME",                            "ALL", "Collateral-free loans to MSEs through credit guarantee cover.", "Micro and small enterprises for loans up to Rs 2 crore.", "Guarantee cover 75–85% of credit facility; no collateral required.", "18","65","ALL","","ALL","Self-employed"),
    ]

    rows = []
    for (sid, name, ministry, state, desc, elig, benefits,
         min_age, max_age, gender, income, caste, occ) in stubs:
        rows.append({
            "scheme_id": sid, "scheme_name": name, "ministry": ministry,
            "state": state, "description": desc, "eligibility": elig,
            "benefits": benefits, "min_age": min_age, "max_age": max_age,
            "gender": gender, "income_limit_inr": income,
            "caste_category": caste, "occupation_tags": occ,
        })
    df = pd.DataFrame(rows)
    return _normalise(df)


# ── Run loaders in priority order ─────────────────────────────────────────────
schemes_df = None
source_used = None

for loader_name, loader_fn in [
    ("my_gov_schemes CSV (primary)",  lambda: _load_myscheme_csv(SCHEMES_CSV_PATH)),
    ("PDF from UC Volume",            lambda: _load_pdf(SCHEMES_PDF_PATH)),
    ("data.gov.in open API",          _load_datagov),
]:
    result = loader_fn()
    if result is not None and len(result) > 0:
        schemes_df = result
        source_used = loader_name
        break

if schemes_df is None:
    schemes_df = _hardcoded_stubs()
    source_used = "hardcoded stubs (50 major central schemes)"

print(f"\n✅ Source  : {source_used}")
print(f"   Schemes : {len(schemes_df)}")
display(schemes_df[["scheme_name", "ministry", "state", "gender",
                     "caste_category", "min_age", "max_age",
                     "income_limit_inr", "occupation_tags"]].head(10))

# COMMAND ----------

# MAGIC %md ## Step 2: Write `gov_welfare_schemes` Delta Table

# COMMAND ----------

from pyspark.sql.types import StructType, StructField, StringType

schemes_schema = StructType([
    StructField(c, StringType(), True) for c in _REQUIRED_COLS
])

# Rename eligibility → eligibility_raw for the Delta table
schemes_df_out = schemes_df.rename(columns={"eligibility": "eligibility_raw"})
schemes_df_out["eligibility_raw"] = schemes_df_out.get("eligibility_raw", schemes_df_out.get("eligibility", ""))

# Ensure all schema columns exist
for col_name in [f.name for f in schemes_schema.fields]:
    actual = col_name.replace("eligibility_raw", "eligibility_raw")
    if actual not in schemes_df_out.columns:
        schemes_df_out[actual] = ""

schemes_sdf = spark.createDataFrame(
    schemes_df_out[[f.name for f in schemes_schema.fields]].astype(str),
    schema=schemes_schema,
)

(
    schemes_sdf.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(SCHEMES_TABLE)
)
spark.sql(f"""
    ALTER TABLE {SCHEMES_TABLE}
    SET TBLPROPERTIES ('delta.enableChangeDataFeed' = 'true')
""")
print(f"✅ {schemes_sdf.count()} schemes written to {SCHEMES_TABLE}")

# COMMAND ----------

# MAGIC %md ## Step 3: TF-IDF Features (Spark SQL/DataFrame)

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.window import Window

# Build a combined text column for TF-IDF
schemes_sdf_loaded = spark.table(SCHEMES_TABLE)
schemes_sdf_text = schemes_sdf_loaded.withColumn(
    "scheme_text",
    F.concat_ws(" ",
        F.coalesce(F.col("scheme_name"), F.lit("")),
        F.coalesce(F.col("description"), F.lit("")),
        F.coalesce(F.col("eligibility_raw"), F.lit("")),
        F.coalesce(F.col("benefits"), F.lit("")),
        F.coalesce(F.col("occupation_tags"), F.lit("")),
        F.coalesce(F.col("state"), F.lit("")),
        F.coalesce(F.col("caste_category"), F.lit("")),
    )
)

# Databricks shared/access-restricted clusters may block SparkML constructors via Py4J whitelist.
# Compute TF-IDF with Spark SQL/DataFrame APIs to avoid blocked JVM constructor calls.
cleaned = schemes_sdf_text.withColumn(
    "clean_text",
    F.trim(F.regexp_replace(F.lower(F.col("scheme_text")), r"[^a-z0-9]+", " ")),
)

tokens = (
    cleaned
    .withColumn("term", F.explode(F.split(F.col("clean_text"), r"\s+")))
    .where(F.length("term") > 1)
)

tf = (
    tokens.groupBy("scheme_id", "scheme_name", "scheme_text", "term")
    .agg(F.count(F.lit(1)).alias("tf"))
)

df_counts = tf.groupBy("term").agg(F.countDistinct("scheme_id").alias("df"))
num_docs = schemes_sdf_text.select("scheme_id").distinct().count()

tfidf = (
    tf.join(df_counts, on="term", how="left")
    .withColumn(
        "idf",
        F.log((F.lit(float(num_docs)) + F.lit(1.0)) / (F.col("df") + F.lit(1.0))) + F.lit(1.0),
    )
    .withColumn("tfidf", F.col("tf") * F.col("idf"))
)

top_terms = (
    tfidf
    .withColumn(
        "term_rank",
        F.row_number().over(
            Window.partitionBy("scheme_id").orderBy(F.desc("tfidf"), F.asc("term"))
        )
    )
    .where(F.col("term_rank") <= 20)
    .groupBy("scheme_id")
    .agg(F.concat_ws(", ", F.collect_list("term")).alias("top_terms"))
)

transformed = schemes_sdf_text.join(top_terms, on="scheme_id", how="left")
print(f"✅ TF-IDF features computed for {num_docs} schemes")

(
    transformed.select("scheme_id", "scheme_name", "scheme_text", "top_terms")
    .write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(TFIDF_TABLE)
)
print(f"✅ TF-IDF top-terms saved to {TFIDF_TABLE}")

# COMMAND ----------

# MAGIC %md ## Step 4: MLflow — Log Run Metadata

# COMMAND ----------

try:
    import mlflow
    with mlflow.start_run(run_name="scheme_tfidf_training"):
        mlflow.log_param("source", source_used)
        mlflow.log_param("num_schemes", len(schemes_df))
        mlflow.log_param("tfidf_impl", "spark_sql_dataframe")
        mlflow.log_param("top_terms_per_scheme", 20)
        run_id = mlflow.active_run().info.run_id
        print(f"✅ MLflow run logged (run_id={run_id})")
except Exception as e:
    print(f"⚠️  MLflow logging failed (non-fatal): {e}")

# COMMAND ----------

# MAGIC %md ## Step 5: Build Corpus Chunks and MERGE into `legal_rag_corpus`

# COMMAND ----------

corpus_rows = []
for _, row in schemes_df.iterrows():
    sid      = str(row.get("scheme_id", "")).strip()
    name     = str(row.get("scheme_name", "")).strip()
    ministry = str(row.get("ministry", "")).strip()
    state    = str(row.get("state", "ALL")).strip()
    desc     = str(row.get("description", "")).strip()
    elig     = str(row.get("eligibility", row.get("eligibility_raw", ""))).strip()
    benefits = str(row.get("benefits", "")).strip()
    min_age  = str(row.get("min_age", "")).strip()
    max_age  = str(row.get("max_age", "")).strip()
    gender   = str(row.get("gender", "ALL")).strip()
    income   = str(row.get("income_limit_inr", "")).strip()
    caste    = str(row.get("caste_category", "ALL")).strip()
    occ      = str(row.get("occupation_tags", "")).strip()
    # Extra fields from my_gov_schemes format
    how_to   = str(row.get("_application", "")).strip()
    docs     = str(row.get("_documents",   "")).strip()
    tags     = str(row.get("_tags",        "")).strip()

    base_id  = f"SCH_{sid}" if sid else f"SCH_{abs(hash(name)) % 100000}"

    # ── Structured eligibility summary (for retrieval signal) ──────────────
    elig_parts = []
    if min_age and min_age not in ("", "None"):
        elig_parts.append(f"Age: {min_age}–{max_age or '+'} years")
    if gender not in ("", "ALL", "None"):
        elig_parts.append(f"Gender: {'Female' if gender=='F' else 'Male' if gender=='M' else gender}")
    if income and income not in ("", "None"):
        try:
            elig_parts.append(f"Income limit: ₹{int(float(income)):,}/year")
        except ValueError:
            pass
    if caste and caste not in ("", "ALL", "None"):
        elig_parts.append(f"Category: {caste}")
    if occ:
        elig_parts.append(f"Occupation: {occ}")
    if state and state not in ("ALL", ""):
        elig_parts.append(f"Applicable: {state}")
    elig_summary = " | ".join(elig_parts)

    # ── Chunk A: Overview (description + benefits) ─────────────────────────
    chunk_a = (
        f"Government Scheme: {name}\n"
        f"Category: {ministry}\n"
        + (f"Eligibility snapshot: {elig_summary}\n" if elig_summary else "")
        + f"\nDescription: {desc[:800]}\n\n"
        f"Benefits: {benefits[:600]}"
        + (f"\n\nKeywords: {tags}" if tags else "")
    )[:2000]

    corpus_rows.append({
        "chunk_id": base_id + "_A",
        "source":   "MYSCHEME_GOV_IN",
        "doc_type": "government_scheme",
        "title":    name[:200],
        "text":     chunk_a,
    })

    # ── Chunk B: Eligibility + How to Apply (only when content exists) ─────
    if elig or how_to:
        chunk_b = (
            f"Government Scheme: {name} — Eligibility & Application\n"
            + (f"Eligibility snapshot: {elig_summary}\n\n" if elig_summary else "")
            + (f"Full eligibility criteria:\n{elig[:700]}\n\n" if elig else "")
            + (f"How to apply:\n{how_to[:500]}\n\n" if how_to else "")
            + (f"Documents required:\n{docs[:300]}" if docs else "")
        )[:2000]

        corpus_rows.append({
            "chunk_id": base_id + "_B",
            "source":   "MYSCHEME_GOV_IN",
            "doc_type": "government_scheme",
            "title":    f"{name[:190]} – Eligibility",
            "text":     chunk_b,
        })

print(f"✅ Prepared {len(corpus_rows)} corpus chunks "
      f"({len(schemes_df)} schemes × ~2 chunks each)")

from pyspark.sql.types import StructType, StructField, StringType

corpus_schema = StructType([
    StructField("chunk_id", StringType(), False),
    StructField("source",   StringType(), True),
    StructField("doc_type", StringType(), True),
    StructField("title",    StringType(), True),
    StructField("text",     StringType(), True),
])

stage_sdf = spark.createDataFrame(pd.DataFrame(corpus_rows), schema=corpus_schema)
stage_sdf.createOrReplaceTempView("scheme_corpus_stage")
print(f"✅ Prepared {len(corpus_rows)} scheme corpus chunks")

spark.sql(f"""
    MERGE INTO {CORPUS_TABLE} AS target
    USING scheme_corpus_stage AS source
    ON target.chunk_id = source.chunk_id
    WHEN MATCHED THEN UPDATE SET *
    WHEN NOT MATCHED THEN INSERT *
""")
print(f"✅ MERGE complete → {CORPUS_TABLE}")

# COMMAND ----------

# MAGIC %md ## Step 6: Verify

# COMMAND ----------

display(spark.sql(f"""
    SELECT doc_type, COUNT(*) AS cnt
    FROM {CORPUS_TABLE}
    GROUP BY doc_type ORDER BY cnt DESC
"""))

display(spark.sql(f"""
    SELECT scheme_name, ministry, state, min_age, max_age, gender, caste_category
    FROM {SCHEMES_TABLE}
    ORDER BY scheme_name
    LIMIT 20
"""))

print(f"\n📋 How to add more schemes:")
print(f"   1. Download CSV from https://data.gov.in or https://myscheme.gov.in")
print(f"   2. Upload to: {SCHEMES_CSV_PATH}")
print(f"   3. Re-run this notebook")
