"""
data_prep.py — Multi-Dataset Fashion Training Pipeline
-------------------------------------------------------
Scans data/raw_datasets/ for every CSV, normalises column names,
merges all files, engineers features, encodes categoricals, and saves.

Engineered features
-------------------
    brand_tier          luxury / mid / fast_fashion / unknown
    is_vintage          1 if item signals vintage origin, else 0
    season              winter / summer / spring / autumn / all_season
    retail_price_ratio  resale price ÷ retail price (actual or estimated)

Adding a new dataset: drop a CSV into data/raw_datasets/ and re-run.
Adding a new brand:   append to the appropriate set below.
Adding a new alias:   append to COLUMN_ALIASES.
"""

import os
import glob
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# ── Paths ─────────────────────────────────────────────────────────────────────
RAW_DATASETS_DIR = "data/raw_datasets"
PROCESSED_PATH   = "data/processed/fashion_training_data.csv"
ENCODER_PATH     = "models/encoders.pkl"

# ── Schema ────────────────────────────────────────────────────────────────────
REQUIRED_COLS = ["brand", "category", "condition", "size", "price"]
OPTIONAL_COLS = ["retail_price"]   # kept when present; used for retail_price_ratio

# Categorical cols split by when they exist:
#   BASE   → present right after loading
#   ENGINEERED → created by engineer_features()
BASE_CATEGORICAL_COLS       = ["brand", "category", "condition", "size"]
ENGINEERED_CATEGORICAL_COLS = ["brand_tier", "season"]
ALL_CATEGORICAL_COLS        = BASE_CATEGORICAL_COLS + ENGINEERED_CATEGORICAL_COLS

# ── Column-name aliases ───────────────────────────────────────────────────────
# NOTE: "retail_price" is intentionally NOT an alias for "price".
#       If a dataset contains both a resale price and an original retail price,
#       they map to different columns and are handled separately.
COLUMN_ALIASES = {
    "brand": [
        "brand", "brand_name", "designer", "make",
        "manufacturer", "label", "vendor",
    ],
    "category": [
        "category", "cat", "item_type", "type",
        "item_category", "product_type", "department",
        "product_category", "subtype",
    ],
    "condition": [
        "condition", "item_condition", "quality",
        "state", "grade", "wear",
    ],
    "size": [
        "size", "item_size", "clothing_size",
        "garment_size", "product_size",
    ],
    "price": [
        "price", "selling_price", "sale_price",
        "asking_price", "resale_price",
    ],
    "retail_price": [
        "retail_price", "original_price", "msrp", "rrp",
        "full_price", "market_price", "tag_price", "list_price",
    ],
}

_ALIAS_LOOKUP = {
    alias: standard
    for standard, aliases in COLUMN_ALIASES.items()
    for alias in aliases
}

# ── Brand-tier classification ─────────────────────────────────────────────────
# Brand names are matched after lowercasing. Add new brands as needed.

LUXURY_BRANDS = {
    "gucci", "louis vuitton", "lv", "prada", "chanel", "hermes", "hermès",
    "dior", "christian dior", "versace", "burberry", "valentino", "givenchy",
    "balenciaga", "saint laurent", "ysl", "bottega veneta", "fendi", "celine",
    "céline", "loewe", "miu miu", "alexander mcqueen", "off-white", "amiri",
    "rick owens", "maison margiela", "margiela", "acne studios", "isabel marant",
    "jacquemus", "toteme", "a.p.c", "apc", "max mara", "lanvin", "balmain",
    "moschino", "vivienne westwood", "stella mccartney", "the row",
    "brunello cucinelli", "loro piana", "tom ford", "dolce & gabbana", "d&g",
}

MID_BRANDS = {
    "ralph lauren", "polo ralph lauren", "calvin klein", "tommy hilfiger",
    "michael kors", "coach", "kate spade", "tory burch", "banana republic",
    "j.crew", "jcrew", "gap", "anthropologie", "free people", "allsaints",
    "ted baker", "reiss", "cos", "sandro", "maje", "rag & bone", "rag and bone",
    "madewell", "vince", "joie", "alice + olivia", "equipment", "ba&sh",
    "reformation", "aritzia", "club monaco", "hugo boss", "boss", "lacoste",
    "gant", "barbour", "patagonia", "north face", "columbia", "filson",
    "eileen fisher", "frame", "ag jeans", "paige", "7 for all mankind",
    "citizens of humanity", "true religion", "dl1961", "joe's jeans",
    "levi's", "levis", "lee", "wrangler", "nike", "adidas", "new balance",
    "converse", "vans", "timberland", "ugg", "dr. martens", "doc martens",
}

FAST_FASHION_BRANDS = {
    "zara", "h&m", "hm", "forever 21", "forever21", "shein", "asos",
    "primark", "boohoo", "missguided", "fashion nova", "urban outfitters",
    "topshop", "uniqlo", "old navy", "express", "american eagle",
    "hollister", "abercrombie", "abercrombie & fitch", "next", "river island",
    "new look", "dorothy perkins", "warehouse", "oasis", "wallis",
    "prettylittlething", "plt", "nasty gal", "romwe", "zaful", "monki",
    "bershka", "pull&bear", "stradivarius",
}

# Typical resale price as a fraction of original retail price, by tier.
# Used to *estimate* retail_price when the dataset doesn't include it.
BRAND_TIER_RESALE_RATIOS = {
    "luxury":       0.38,   # luxury resells at ~38 % of retail
    "mid":          0.22,   # mid-range at ~22 %
    "fast_fashion": 0.10,   # fast fashion at ~10 %
    "unknown":      0.18,   # conservative default
}

# ── Vintage keywords ──────────────────────────────────────────────────────────
# Checked across condition, category, and brand fields (all lowercased).
VINTAGE_KEYWORDS = {
    "vintage", "retro", "antique", "classic", "deadstock",
    "y2k", "90s", "80s", "70s", "60s", "50s",
    "pre-owned", "pre owned", "second hand", "secondhand",
}

# ── Season keywords ───────────────────────────────────────────────────────────
# Matched against the category field. First match wins.
SEASON_KEYWORDS: dict[str, set[str]] = {
    "winter": {
        "coat", "puffer", "parka", "down jacket", "fleece", "thermal",
        "ski", "snow", "boots", "gloves", "scarf", "beanie", "knitwear",
        "sweater", "jumper", "cardigan", "hoodie", "wool",
    },
    "summer": {
        "swimwear", "bikini", "swim", "shorts", "sandals", "sundress",
        "tank", "crop top", "linen", "beach", "resort", "halter",
        "camisole", "sleeveless", "flip flops",
    },
    "spring": {
        "trench", "raincoat", "windbreaker", "light jacket",
        "floral", "pastel", "blouse", "midi dress",
    },
    "autumn": {
        "leather jacket", "denim jacket", "flannel", "plaid",
        "check shirt", "corduroy", "bomber",
    },
}


# ── Step 1: Discover CSVs ─────────────────────────────────────────────────────

def find_datasets(folder: str) -> list[str]:
    """Return sorted list of all CSV paths inside folder."""
    paths = sorted(glob.glob(os.path.join(folder, "*.csv")))
    if not paths:
        raise FileNotFoundError(
            f"No CSV files found in '{folder}'.\n"
            "Add at least one CSV with columns: brand, category, condition, size, price\n"
            "(exact header names may vary — see COLUMN_ALIASES in data_prep.py)"
        )
    print(f"Found {len(paths)} dataset(s) in '{folder}':")
    for p in paths:
        print(f"  {p}")
    return paths


# ── Step 2: Load + normalise a single CSV ────────────────────────────────────

def normalise_columns(df: pd.DataFrame, source: str) -> pd.DataFrame | None:
    """
    Rename columns to the standard schema using COLUMN_ALIASES.

    Keeps all REQUIRED_COLS plus any OPTIONAL_COLS that happen to be present.
    Returns None (with a warning) if any required column cannot be mapped.
    """
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    rename_map = {col: _ALIAS_LOOKUP[col] for col in df.columns if col in _ALIAS_LOOKUP}
    df = df.rename(columns=rename_map)

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        print(f"  [SKIP] '{source}' — missing after normalisation: {missing}")
        print(f"         Available: {list(df.columns)}")
        print(f"         Add an alias to COLUMN_ALIASES to map them.")
        return None

    keep = REQUIRED_COLS + [c for c in OPTIONAL_COLS if c in df.columns]
    return df[keep]


def load_single_dataset(path: str) -> pd.DataFrame | None:
    """Load one CSV, normalise columns, return it (or None if unusable)."""
    filename = os.path.basename(path)
    try:
        df = pd.read_csv(path)
        print(f"\n  Loading '{filename}' — {len(df):,} rows, columns: {list(df.columns)}")
        df = normalise_columns(df, filename)
        if df is not None:
            has_retail = "retail_price" in df.columns
            print(f"  Normalised → {len(df):,} rows "
                  f"({'retail_price present' if has_retail else 'no retail_price'})")
        return df
    except Exception as e:
        print(f"  [SKIP] '{filename}' — could not read: {e}")
        return None


# ── Step 3: Merge ─────────────────────────────────────────────────────────────

def load_and_merge(folder: str) -> pd.DataFrame:
    """Load every CSV, tag each row with its source file, concatenate."""
    paths = find_datasets(folder)
    frames = []
    for path in paths:
        df = load_single_dataset(path)
        if df is not None:
            df["source"] = os.path.basename(path)
            frames.append(df)

    if not frames:
        raise ValueError(
            "No datasets could be loaded. Check column names against COLUMN_ALIASES."
        )

    merged = pd.concat(frames, ignore_index=True)
    print(f"\nMerged {len(frames)} dataset(s) → {len(merged):,} total rows")
    return merged


# ── Canonical category map ────────────────────────────────────────────────────
# Maps every raw first-level category token (after lowercasing and the slash
# split) to one of the 10 canonical buckets below.  Add new synonyms here as
# new datasets are introduced; unknown tokens fall back to "other" automatically.

CATEGORY_MAP: dict[str, str] = {
    # Women
    "women": "women", "womens": "women", "woman": "women",
    "ladies": "women", "female": "women",
    # Men
    "men": "men", "mens": "men", "man": "men", "male": "men",
    # Kids
    "kids": "kids", "children": "kids", "child": "kids",
    "baby": "kids", "boys": "kids", "girls": "kids", "youth": "kids",
    # Shoes
    "shoes": "shoes", "footwear": "shoes", "sneakers": "shoes",
    "boots": "shoes", "sandals": "shoes", "heels": "shoes",
    # Bags
    "bags": "bags", "bag": "bags", "handbags": "bags",
    "purses": "bags", "luggage": "bags", "wallets": "bags",
    # Accessories
    "accessories": "accessories", "jewelry": "accessories",
    "jewellery": "accessories", "watches": "accessories",
    "hats": "accessories", "scarves": "accessories",
    "belts": "accessories", "sunglasses": "accessories",
    "handmade": "accessories",
    # Outerwear
    "outerwear": "outerwear", "coats": "outerwear",
    "jackets": "outerwear", "coats & jackets": "outerwear",
    # Sportswear
    "sportswear": "sportswear", "sports": "sportswear",
    "athletic": "sportswear", "activewear": "sportswear",
    "gym": "sportswear", "fitness": "sportswear",
    # Vintage
    "vintage": "vintage", "retro": "vintage", "antique": "vintage",
    # Everything else collapses to "other" via .fillna() in simplify_category
}

# The 10 canonical values the model trains on and the UI displays.
CANONICAL_CATEGORIES: list[str] = sorted([
    "women", "men", "kids", "shoes", "bags",
    "accessories", "outerwear", "sportswear", "vintage", "other",
])


# ── Step 4: Clean ─────────────────────────────────────────────────────────────

def simplify_category(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reduce every category value to one of the 10 CANONICAL_CATEGORIES.

    Step 1 — slash split: "Men/Tops/T-shirts" → "men"
    Step 2 — canonical map: "men" → "men", "handmade" → "accessories",
                             anything unknown → "other"

    Both steps operate on already-lowercased, stripped strings so the
    mapping keys don't need case variants.
    """
    before = df["category"].nunique()
    df["category"] = (
        df["category"]
        .str.split("/").str[0].str.strip()   # step 1: first slash-segment
        .map(CATEGORY_MAP)                    # step 2: collapse to canonical
        .fillna("other")                      # unknown tokens → "other"
    )
    after = df["category"].nunique()
    print(f"Simplified categories: {before} raw → {after} canonical "
          f"({', '.join(sorted(df['category'].unique()))})")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise text, simplify category paths, coerce numerics, drop bad rows.

    Only BASE_CATEGORICAL_COLS are normalised here — engineered columns
    (brand_tier, season) don't exist yet and are created in the next step.
    """
    for col in BASE_CATEGORICAL_COLS:
        df[col] = df[col].astype(str).str.strip().str.lower()

    df = simplify_category(df)

    df["price"] = pd.to_numeric(df["price"], errors="coerce")

    if "retail_price" in df.columns:
        df["retail_price"] = pd.to_numeric(df["retail_price"], errors="coerce")

    before = len(df)
    df = df.dropna(subset=REQUIRED_COLS)
    df = df[df["price"] > 0]
    df = df.drop_duplicates(subset=REQUIRED_COLS)

    removed = before - len(df)
    if removed:
        print(f"Removed {removed:,} rows (missing / non-positive price / duplicates)")
    print(f"Clean dataset: {len(df):,} rows")
    return df


# ── Step 5: Feature engineering ───────────────────────────────────────────────
# All functions below read text columns (brand, category, condition) as
# normalised strings and must run BEFORE encode_categoricals replaces them
# with integers.

def _assign_brand_tier(brand: str) -> str:
    """Map a brand name to luxury / mid / fast_fashion / unknown."""
    b = brand.strip().lower()
    if b in LUXURY_BRANDS:
        return "luxury"
    if b in MID_BRANDS:
        return "mid"
    if b in FAST_FASHION_BRANDS:
        return "fast_fashion"
    return "unknown"


def _assign_season(category: str) -> str:
    """Infer season from keywords in the category string. First match wins."""
    cat = category.lower()
    for season, keywords in SEASON_KEYWORDS.items():
        if any(kw in cat for kw in keywords):
            return season
    return "all_season"


def _is_vintage(row: pd.Series) -> int:
    """Return 1 if any vintage keyword appears across condition, category, or brand."""
    text = f"{row['condition']} {row['category']} {row['brand']}".lower()
    return int(any(kw in text for kw in VINTAGE_KEYWORDS))


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add four derived features to the DataFrame.

    brand_tier
        Classified from the brand name against three curated sets
        (LUXURY_BRANDS, MID_BRANDS, FAST_FASHION_BRANDS). Brands not
        found in any set are labelled "unknown". Encoded as a categorical
        integer — the model can learn that tier is a stronger signal than
        the raw brand code for most items.

    is_vintage
        Binary flag (0/1). Set to 1 when any VINTAGE_KEYWORDS appear in
        condition, category, or brand. Vintage items consistently command
        a premium regardless of base condition, so a dedicated flag helps
        the model isolate that effect.

    season
        Inferred from category keywords (SEASON_KEYWORDS). Items with no
        seasonal signal are labelled "all_season". Seasonality affects
        supply and demand — a winter coat listed in July fetches less than
        the same coat in October.

    retail_price_ratio
        resale_price / retail_price.
        When the dataset includes retail_price, the ratio is computed
        directly per item.
        When retail_price is absent, it is estimated as:
            retail_price ≈ resale_price / BRAND_TIER_RESALE_RATIOS[tier]
        giving an estimated ratio equal to the tier's typical resale
        fraction. This is less precise than a real price but still adds
        signal — a luxury item at 80 % of retail is priced very differently
        from the same item at 15 %.
    """
    df = df.copy()

    # ── brand_tier ────────────────────────────────────────────────────────────
    df["brand_tier"] = df["brand"].apply(_assign_brand_tier)
    tier_counts = df["brand_tier"].value_counts()
    print(f"\nEngineered 'brand_tier':")
    for tier, count in tier_counts.items():
        print(f"  {tier:<15}: {count:,}")

    # ── is_vintage ────────────────────────────────────────────────────────────
    df["is_vintage"] = df.apply(_is_vintage, axis=1)
    vintage_count = df["is_vintage"].sum()
    print(f"\nEngineered 'is_vintage': {vintage_count:,} vintage, "
          f"{len(df) - vintage_count:,} non-vintage")

    # ── season ────────────────────────────────────────────────────────────────
    df["season"] = df["category"].apply(_assign_season)
    season_counts = df["season"].value_counts()
    print(f"\nEngineered 'season':")
    for season, count in season_counts.items():
        print(f"  {season:<12}: {count:,}")

    # ── retail_price_ratio ────────────────────────────────────────────────────
    if "retail_price" in df.columns:
        # Use actual retail price where valid; fall back to estimate elsewhere
        valid_retail = df["retail_price"].notna() & (df["retail_price"] > 0)
        estimated_retail = df["price"] / df["brand_tier"].map(BRAND_TIER_RESALE_RATIOS)
        retail = df["retail_price"].where(valid_retail, other=estimated_retail)
        actual_count = valid_retail.sum()
        print(f"\nEngineered 'retail_price_ratio': "
              f"{actual_count:,} rows use actual retail_price, "
              f"{len(df) - actual_count:,} rows use tier estimate")
    else:
        # No retail_price column — estimate for all rows
        retail = df["price"] / df["brand_tier"].map(BRAND_TIER_RESALE_RATIOS)
        print(f"\nEngineered 'retail_price_ratio': no retail_price column, "
              f"using tier-based estimates for all {len(df):,} rows")

    df["retail_price_ratio"] = (df["price"] / retail).clip(0, 2).round(4)

    # Remove the raw retail_price column — ratio is the feature the model uses
    df.drop(columns=["retail_price"], errors="ignore", inplace=True)

    return df


# ── Step 6: Encode ────────────────────────────────────────────────────────────

def encode_categoricals(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Fit a LabelEncoder on each categorical column (base + engineered).

    is_vintage and retail_price_ratio are numeric — they are not encoded.
    Encoders are saved so app.py can apply the same mapping at inference.
    """
    encoders = {}
    for col in ALL_CATEGORICAL_COLS:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le
        print(f"Encoded '{col}' → {len(le.classes_)} unique values")
    return df, encoders


# ── Step 7: Save ──────────────────────────────────────────────────────────────

def save_outputs(df: pd.DataFrame, encoders: dict) -> None:
    """Write the processed CSV and encoders to disk."""
    os.makedirs(os.path.dirname(PROCESSED_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(ENCODER_PATH), exist_ok=True)

    df.drop(columns=["source"], errors="ignore").to_csv(PROCESSED_PATH, index=False)
    joblib.dump(encoders, ENCODER_PATH)

    print(f"\nSaved combined dataset → '{PROCESSED_PATH}'")
    print(f"Saved encoders        → '{ENCODER_PATH}'")


# ── Full pipeline ──────────────────────────────────────────────────────────────

def run_pipeline():
    print("=== Data Preparation Pipeline ===")
    df = load_and_merge(RAW_DATASETS_DIR)
    df = clean_data(df)
    df = engineer_features(df)        # text columns still strings here
    df, encoders = encode_categoricals(df)
    save_outputs(df, encoders)
    print("=== Done ===\n")
    return df, encoders


if __name__ == "__main__":
    run_pipeline()
