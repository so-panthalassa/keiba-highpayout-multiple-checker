import re, math, logging, time, requests
import pandas as pd
from bs4 import BeautifulSoup
import numpy as np
import logging
import unicodedata

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

HDRS = {
    "User-Agent": "Mozilla/5.0",
    "Accept-Language": "ja,en;q=0.9",
}

results_df = pd.read_pickle("02_results_df.pickle")  # 学習時の結果データ

# =========================================================
# ===================== スクレイピング =====================
# =========================================================

def _norm(s: str | None) -> str:
    if s is None: return ""
    t = unicodedata.normalize("NFKC", str(s))
    return t.replace("\u00A0"," ").replace("－","-").replace("−","-").replace("―","-").strip()

def _norm_name(s: str | None) -> str:
    """馬名の突合用正規化：全角→半角、記号/スペース/括弧内を極力除去"""
    t = _norm(s)
    # 括弧内（全角/半角）を削除
    t = re.sub(r"[（(].*?[）)]", "", t)
    # 中黒・ハイフン・スラッシュ・ドット・スペース系を除去
    t = re.sub(r"[・\-/\.･\s]", "", t)
    return t

def _norm_person(s: str | None) -> str:
    """騎手/調教師の突合用正規化：接頭辞[東][西][地][外]等の除去＋余白整理"""
    t = _norm(s)
    t = re.sub(r"^\[[^\]]+\]\s*", "", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()

def _num(s):
    if s is None: return None
    t = _norm(s).replace(",", "")
    m = re.search(r"-?\d+(?:\.\d+)?", t)
    return float(m.group()) if m else None

def _int(s):
    v = _num(s)
    return int(v) if v is not None else None

def _to_date_from_header(txt: str) -> pd.Timestamp | None:
    m = re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})日", _norm(txt))
    if not m: return None
    y,mth,d = map(int, m.groups())
    try: return pd.Timestamp(year=y, month=mth, day=d)
    except Exception: return None

def _ensure_datetime_ja(s: pd.Series) -> pd.Series:
    s = s.astype(str)
    if s.str.contains("年").any():
        s2 = s.str.replace(r"年|月","-", regex=True).str.replace("日","", regex=False)
        return pd.to_datetime(s2, errors="coerce")
    return pd.to_datetime(s, errors="coerce")

def _derive_dist_bucket(course_len: pd.Series) -> pd.Series:
    x = pd.to_numeric(course_len, errors="coerce")
    bins   = [-np.inf, 1400, 1700, 2200, np.inf]
    labels = ["短距離","マイル","中距離","長距離"]
    return pd.cut(x, bins=bins, labels=labels)

# ===================== JRA出馬表スクレイパ =====================
def fetch_jra_shutuba(
    url: str,
    weather: str | None = None,
    ground_state: str | None = None,
    history_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    JRA公式の出馬表から当日特徴量を作り、（あれば）履歴DataFrameと“馬名マッチ”で
    直近/適性/コンビ特徴を付与して返す。
    """
    # HTML
    res = requests.get(url, headers=HDRS, timeout=20)
    res.raise_for_status()
    try:
        html = res.content.decode("cp932")
    except UnicodeDecodeError:
        html = res.content.decode("utf-8", errors="ignore")
    soup = BeautifulSoup(html, "lxml")

    # --- レース共通情報 ---
    course_len = None
    race_type  = None
    place_name = None
    month_num  = None
    race_date  = None

    # コース（距離/種別）
    if (div_course := soup.select_one("div.cell.course")):
        txt_all = _norm(div_course.get_text(" ", strip=True))
        if (m := re.search(r"(\d[\d,]*)\s*メートル", txt_all)):
            course_len = int(_norm(m.group(1)).replace(",", ""))
        # 種別
        if (sp := div_course.select_one("span.detail")):
            dtxt = _norm(sp.get_text(strip=True))  # 例: "（芝・左）"
            if (m := re.search(r"（\s*([^）]+?)\s*）", dtxt)):
                head = m.group(1).split("・")[0]
                if "芝" in head: race_type = "芝"
                elif "ダ" in head or "ダート" in head: race_type = "ダ"
                elif "障" in head or "障害" in head: race_type = "障"
        if race_type is None:
            if "芝" in txt_all: race_type = "芝"
            elif ("ダート" in txt_all) or ("ダ" in txt_all): race_type = "ダ"
            elif ("障" in txt_all) or ("障害" in txt_all): race_type = "障"

    # 日付・場名・月
    if (div_date := soup.select_one("div.cell.date")):
        txt = _norm(div_date.get_text(" ", strip=True))
        race_date = _to_date_from_header(txt)
        if (m := re.search(r"\d{4}年(\d{1,2})月\d{1,2}日", txt)):
            month_num = int(m.group(1))
        if (m := re.search(r"\d+回([^\d\s]+)\d+日", txt)):
            place_name = m.group(1)

    # --- 出走行 ---
    rows = [tr for tr in soup.find_all("tr") if tr.find("td","num") and tr.find("td","horse")]

    recs = []
    for tr in rows:
        # 枠番
        waku = None
        if (t_waku := tr.find("td","waku")):
            img = t_waku.find("img")
            alt = img.get("alt") if img else ""
            if (m := re.search(r"\d+", alt or "")):
                waku = int(m.group())

        # 馬番
        umaban = _int(tr.find("td","num").get_text(strip=True))
        td_horse = tr.find("td","horse")

        # 馬名
        uma_name = None
        if (name_div := td_horse.select_one(".name_line .name")):
            uma_name = _norm(name_div.get_text(strip=True))

        # 単勝/人気
        tansho = None; ninki = None
        if (od := td_horse.select_one(".odds .odds_line .num strong")):
            tansho = _num(od.get_text(strip=True))
        if (pr := td_horse.select_one(".odds .odds_line .pop_rank")):
            ninki = _int(pr.get_text())

        # 馬体重
        uma_w = uma_w_diff = None
        if (wc := td_horse.select_one(".result_line .cell.weight")):
            wt = _norm(wc.get_text(" ", strip=True))
            if (m := re.search(r"(\d+)\s*kg", wt)): uma_w = int(m.group(1))
            if (m := re.search(r"\(([-+]?\d+)\)", wt)): uma_w_diff = int(m.group(1))

        # 性齢・斤量・騎手
        td_info = tr.find("td","jockey")
        sex = age = None
        if td_info and (p := td_info.find("p","age")):
            t = _norm(p.get_text(strip=True))
            if t:
                sex = t[0]
                age = _int(t)
        kin = None
        if td_info and (p := td_info.find("p","weight")):
            kin = _num(p.get_text())
        jockey = None
        if td_info and (p := td_info.find("p","jockey")):
            a = p.find("a")
            jockey = _norm(a.get_text(strip=True) if a else p.get_text(strip=True))

        # 調教師 + 所属
        trainer, belong = None, None
        if (p_t := td_horse.find("p","trainer")):
            a = p_t.find("a")
            trainer_name = _norm(a.get_text(strip=True) if a else p_t.get_text(strip=True))
            if (div := p_t.find("span","division")):
                d = _norm(div.get_text())
                if "美浦" in d: belong = "東"
                elif "栗東" in d: belong = "西"
            if (icon := p_t.find("span","horse_icon")):
                img = icon.find("img")
                alt = _norm(img.get("alt") if img else "")
                if "カクチ" in alt:  belong = "地"
                elif "カクガイ" in alt: belong = "外"
            trainer = f"[{belong}] {trainer_name}" if belong else trainer_name

        recs.append({
            "枠番": waku, "馬番": umaban, "馬名": uma_name,
            "性": sex, "年齢": age, "斤量": kin,
            "騎手": jockey, "調教師": trainer, "所属": belong,
            "馬体重": uma_w, "馬体重_増減": uma_w_diff,
            "単勝": tansho, "人気": ninki,
            "course_len": course_len, "race_type": race_type,
            "競馬場": place_name, "月": (race_date.month if race_date is not None else month_num),
            "weather": weather if weather is not None else "晴",
            "ground_state": ground_state if ground_state is not None else "良",
            "date": race_date,
            "__name_key__": _norm_name(uma_name),
        })

    df = pd.DataFrame(recs)
    if df.empty:
        return df

    # --- レース内派生 ---
    df = df.sort_values("馬番", na_position="last").reset_index(drop=True)
    n = len(df)
    df["出走頭数"]   = n
    df["相対枠位置"] = df["馬番"] / n
    df["大外枠"]     = (df["馬番"] == n).astype(int)
    df["最内枠"]     = (df["馬番"] == 1).astype(int)

    # 単勝関連
    df["単勝"]      = pd.to_numeric(df["単勝"], errors="coerce")
    df["log単勝"]   = df["単勝"].apply(lambda x: math.log(x) if pd.notna(x) and x > 0 else np.nan)
    df["単勝ランク"] = df["単勝"].rank(method="average", ascending=True)
    df["単勝pct"]   = ((df["単勝ランク"] - 1) / (n - 1)) if n > 1 else 0.0
    if df["単勝"].notna().any():
        fav    = df["単勝"].min(skipna=True)
        arr    = np.sort(df["単勝"].dropna().values)
        second = arr[1] if len(arr) >= 2 else np.nan
        df["単勝_最人気差"]   = df["単勝"] - fav
        df["単勝_2番人気差"] = df["単勝"] - (second if np.isfinite(second) else np.nan)
    else:
        df["単勝_最人気差"] = np.nan
        df["単勝_2番人気差"] = np.nan

    # 距離区分（適性マージ用）
    df["距離区分"] = _derive_dist_bucket(df["course_len"]).astype(object)

        # ================== 履歴ベース特徴量（馬名マッチ） ==================
    if history_df is not None and len(history_df) > 0:
        hist = history_df.copy()

        # 当日レース日（なければ過去全件を対象）
        today_dt = df["date"].iloc[0] if "date" in df.columns and not df["date"].isna().all() else pd.NaT

        # 並び替え（馬名キー + 日付）
        hist = hist.sort_values(["__name_key__", "date"]).reset_index(drop=True)

        def rolling_rate(series, window, min_periods):
            return series.shift().rolling(window, min_periods=min_periods).mean()

        # 馬の直近
        hist["馬_直近3走_複勝率"]  = hist.groupby("__name_key__")["複勝フラグ"].transform(lambda s: rolling_rate(s, 3, 1))
        hist["馬_直近5走_複勝率"]  = hist.groupby("__name_key__")["複勝フラグ"].transform(lambda s: rolling_rate(s, 5, 1))
        hist["馬_直近10走_複勝率"] = hist.groupby("__name_key__")["複勝フラグ"].transform(lambda s: rolling_rate(s,10, 3))
        hist["馬_直近5走_平均着順"] = hist.groupby("__name_key__")["着順_num"].transform(lambda s: s.shift().rolling(5, min_periods=1).mean())

        # 騎手/調教師 直近
        if "__jockey_key__" in hist.columns:
            hist["騎手_直近30走_複勝率"] = hist.groupby("__jockey_key__")["複勝フラグ"].transform(lambda s: rolling_rate(s, 30, 5))
        if "__trainer_key__" in hist.columns:
            hist["調教師_直近50走_複勝率"] = hist.groupby("__trainer_key__")["複勝フラグ"].transform(lambda s: rolling_rate(s, 50, 5))

        # 馬×コース種
        if "race_type" in hist.columns:
            hist["馬_コース適性_複勝率"] = (
                hist.groupby(["__name_key__","race_type"])["複勝フラグ"]
                    .transform(lambda s: s.shift().expanding(min_periods=3).mean())
            )
        # 馬×距離区分
        if "距離区分" in hist.columns:
            hist["馬_距離適性_複勝率"] = (
                hist.groupby(["__name_key__","距離区分"])["複勝フラグ"]
                    .transform(lambda s: s.shift().expanding(min_periods=3).mean())
            )

        # コンビ（任意だが学習列にあるなら作る）
        if "__jockey_key__" in hist.columns and "__trainer_key__" in hist.columns:
            hist["コンビ_直近50走_複勝率"] = (
                hist.groupby(["__jockey_key__","__trainer_key__"])["複勝フラグ"]
                    .transform(lambda s: rolling_rate(s, 50, 5))
            )

        join_cols = [
            "馬_直近3走_複勝率","馬_直近5走_複勝率","馬_直近10走_複勝率","馬_直近5走_平均着順",
            "騎手_直近30走_複勝率","調教師_直近50走_複勝率",
            "馬_コース適性_複勝率","馬_距離適性_複勝率","コンビ_直近50走_複勝率"
        ]
        take_cols = ["__name_key__","__jockey_key__","__trainer_key__","date"] + [c for c in join_cols if c in hist.columns]
        hist2 = hist[take_cols].copy()

        # 当日より前の最新行を抽出（馬名キーで）
        latest_rows = []
        for _, r in df.iterrows():
            key = r["__name_key__"]
            h = hist2[hist2["__name_key__"] == key]
            if h.empty:
                continue
            if pd.notna(today_dt):
                h = h[h["date"] < today_dt]
            if h.empty:
                continue
            latest_rows.append(h.iloc[-1])
        latest = pd.DataFrame(latest_rows) if latest_rows else pd.DataFrame(columns=take_cols)

        if not latest.empty:
            df = df.merge(latest.drop(columns=["date"]), on="__name_key__", how="left", suffixes=("","_hist"))

        # ログ用
        exist_cols = [c for c in join_cols if c in df.columns]
        matched = df[exist_cols].notna().any(axis=1).sum() if exist_cols else 0
        logging.info(f"match by name: today_keys={df['__name_key__'].nunique()} -> matched_rows={matched}")

        # 欠損埋め（全体複勝率など）
        overall = float(hist.get("複勝フラグ", pd.Series([0.0])).mean())
        fill_rate_cols = [
            "馬_直近3走_複勝率","馬_直近5走_複勝率","馬_直近10走_複勝率",
            "騎手_直近30走_複勝率","調教師_直近50走_複勝率",
            "馬_コース適性_複勝率","馬_距離適性_複勝率","コンビ_直近50走_複勝率"
        ]
        for c in fill_rate_cols:
            if c in df.columns:
                df[c] = df[c].fillna(overall)
        if "馬_直近5走_平均着順" in df.columns:
            base_mean = hist["着順_num"].mean() if "着順_num" in hist.columns else df["馬_直近5走_平均着順"].mean()
            df["馬_直近5走_平均着順"] = df["馬_直近5走_平均着順"].fillna(base_mean)

    else:
        # 履歴なしのデフォルト（安全値）
        df["馬_直近3走_複勝率"] = 0.0
        df["馬_直近5走_複勝率"] = 0.0
        df["馬_直近10走_複勝率"] = 0.0
        df["騎手_直近30走_複勝率"] = 0.0
        df["調教師_直近50走_複勝率"] = 0.0
        df["馬_コース適性_複勝率"] = 0.0
        df["馬_距離適性_複勝率"] = 0.0
        df["コンビ_直近50走_複勝率"] = 0.0
        df["馬_直近5走_平均着順"] = float(df["馬番"].mean()) if df["馬番"].notna().any() else 6.5

    # どの分岐でも学習列が必ず存在するよう最終保証
    must_cols_hist = [
        "馬_直近3走_複勝率","馬_直近5走_複勝率","馬_直近10走_複勝率",
        "騎手_直近30走_複勝率","調教師_直近50走_複勝率",
        "馬_コース適性_複勝率","馬_距離適性_複勝率",
        "コンビ_直近50走_複勝率","馬_直近5走_平均着順"
    ]
    defaults = {c: 0.0 for c in must_cols_hist}
    if "馬_直近5走_平均着順" in defaults:
        defaults["馬_直近5走_平均着順"] = float(df["馬番"].mean()) if df["馬番"].notna().any() else 6.5
    for c in must_cols_hist:
        if c not in df.columns:
            df[c] = defaults[c]
        else:
            df[c] = df[c].fillna(defaults[c])

    # ===== 最後に必ず返す =====
    return df


# ===================== 履歴DFの前処理（馬名ベース） =====================
def prepare_history_by_name(results_df_raw: pd.DataFrame) -> pd.DataFrame:
    df = results_df_raw.copy()

    # 列名ゆらぎ
    if "馬名" not in df.columns:
        for cand in ["horse_name","HORSE_NAME"]:
            if cand in df.columns:
                df = df.rename(columns={cand:"馬名"}); break
    if "騎手" not in df.columns:
        for cand in ["jockey","JOCKEY"]:
            if cand in df.columns:
                df = df.rename(columns={cand:"騎手"}); break
    if "調教師" not in df.columns:
        for cand in ["trainer","TRAINER"]:
            if cand in df.columns:
                df = df.rename(columns={cand:"調教師"}); break

    # 日付
    date_col = None
    for cand in ["date","日付","開催日"]:
        if cand in df.columns:
            date_col = cand; break
    if date_col is None:
        raise ValueError("history_df に日付列が見つかりません（'date' or '日付' 等）。")
    df["date"] = pd.to_datetime(
        df[date_col].astype(str)
            .str.replace(r"年|月","-", regex=True)
            .str.replace("日","", regex=False),
        errors="coerce"
    )

    # 着順→数値、複勝フラグ
    if "着順_num" not in df.columns:
        rank_col = "着順" if "着順" in df.columns else None
        df["着順_num"] = pd.to_numeric(df[rank_col], errors="coerce") if rank_col else np.nan
    df["複勝フラグ"] = (df["着順_num"] <= 3).astype(float).fillna(0.0)

    # race_type 補完
    if "race_type" not in df.columns:
        if "コース" in df.columns:
            df["race_type"] = df["コース"].astype(str).map(
                lambda s: "芝" if "芝" in s else ("ダ" if ("ダ" in s or "ダート" in s) else ("障" if "障" in s else np.nan))
            )
        else:
            df["race_type"] = np.nan

    # 距離区分
    if "距離区分" not in df.columns:
        src = None
        for cand in ["course_len","距離","距離(m)"]:
            if cand in df.columns: src = cand; break
        df["距離区分"] = _derive_dist_bucket(df[src]).astype(object) if src else np.nan

    # 突合キー
    df["__name_key__"]    = df["馬名"].map(_norm_name) if "馬名" in df.columns else ""
    df["__jockey_key__"]  = df["騎手"].map(_norm_person) if "騎手" in df.columns else ""
    df["__trainer_key__"] = df["調教師"].map(_norm_person) if "調教師" in df.columns else ""

    keep = ["馬名","date","着順_num","複勝フラグ","騎手","調教師","race_type","距離区分",
            "__name_key__","__jockey_key__","__trainer_key__"]
    return df[[c for c in keep if c in df.columns]]

# =========================================================
# ===================== 予測関数 ===========================
# =========================================================

# ====必要ライブラリ ====
import os, re, json, unicodedata, math, logging
from pathlib import Path
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

# ====ログ ====
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ====モデル/特徴量ファイルのパス ====
MODEL_LGBM_PATH = Path("03_lgbm_nativecat_odds_tuned.joblib")      # LightGBM学習済みモデル
MODEL_CAT_PATH  = Path("03_catboost_odds_features.cbm")       # CatBoost学習済みモデル
LGBM_FEATS_JSON = Path("03_lgbm_feature_cols.json")  # あれば使う
CAT_FEATS_JSON  = Path("03_catboost_feature_cols.json")  # あれば使う

# ====HTTPヘッダ ====
HDRS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Accept-Language": "ja,en;q=0.9",
}

BASE_FEATURES = [
    '枠番','斤量','騎手','馬体重','調教師','course_len','weather','race_type',
    'ground_state','競馬場','所属','月','馬体重_増減','性','年齢','出走頭数','相対枠位置',
    '大外枠','最内枠','馬_直近3走_複勝率','馬_直近5走_複勝率','馬_直近10走_複勝率','馬_直近5走_平均着順',
    '騎手_直近30走_複勝率','調教師_直近50走_複勝率','馬_コース適性_複勝率','馬_距離適性_複勝率',
    'コンビ_直近50走_複勝率','log単勝','単勝ランク','単勝pct','単勝_最人気差','単勝_2番人気差'
]

DEBUG_DIR = Path("_debug_snapshot")
DEBUG_DIR.mkdir(exist_ok=True, parents=True)

def debug_show_df(df: pd.DataFrame, name: str):
    try:
        path = DEBUG_DIR / f"{name}.csv"
        df.to_csv(path, index=False, encoding="utf-8-sig")
        logging.info(f"debug csv saved: {path}")
    except Exception as e:
        logging.warning(f"debug csv save failed: {name}: {e}")

def load_feature_and_cats(json_path: Path, fallback_feats: list[str], fallback_cats: list[str]):
    feats = fallback_feats
    cats  = fallback_cats
    if json_path.exists():
        try:
            text = json_path.read_text(encoding="utf-8").strip()
            if not text:
                raise ValueError("empty json file")
            obj = json.loads(text)
            if isinstance(obj, dict):
                if "features" in obj:
                    feats = list(obj["features"])
                if "categorical_features" in obj:
                    cats = list(obj["categorical_features"])
            elif isinstance(obj, list):
                feats = list(obj)
            logging.info(f"feature list loaded: {json_path} (n={len(feats)}; cats={cats})")
        except Exception as e:
            logging.warning(f"feature list load failed: {json_path} ({e}) -> fallback")
    return feats, cats

def parse_month_from_url(url: str) -> float | int | None:
    # 末尾に YYYYMMDD が含まれているケース（例: ...20250830）
    m = re.search(r"(20\d{6})", url)
    if not m:
        return None
    ymd = m.group(1)
    mm = int(ymd[4:6])
    return mm

def _to_categorical(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = df[c].astype("string").astype("category")
    return df

def _safe_div(a, b, default=np.nan):
    try:
        return a / b
    except Exception:
        return default

def add_market_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['出走頭数'] = len(df)
    df['相対枠位置'] = _safe_div(
        df['枠番'] - 1,
        (df['枠番'].max() - 1) if pd.notna(df['枠番'].max()) and df['枠番'].max() > 1 else 1,
        0.5
    )
    df['最内枠'] = (df['枠番'] == df['枠番'].min()).astype(int)
    df['大外枠'] = (df['枠番'] == df['枠番'].max()).astype(int)

    if '単勝' in df and df['単勝'].notna().any():
        df['log単勝'] = df['単勝'].apply(lambda x: math.log(x) if pd.notna(x) and x > 0 else np.nan)
        df['単勝ランク'] = df['単勝'].rank(method='min', ascending=True, na_option='bottom')
        inv = df['単勝'].apply(lambda x: 1/x if pd.notna(x) and x > 0 else np.nan)
        denom = np.nansum(inv)
        df['単勝pct'] = inv / denom if denom and denom > 0 else np.nan

        try:
            sorted_odds = np.sort(df['単勝'].dropna().values)
            if len(sorted_odds) >= 2:
                best, second = sorted_odds[0], sorted_odds[1]
            elif len(sorted_odds) == 1:
                best, second = sorted_odds[0], np.nan
            else:
                best, second = np.nan, np.nan
            df['単勝_最人気差'] = df['単勝'] - best
            df['単勝_2番人気差'] = df['単勝'] - second if pd.notna(second) else np.nan
        except Exception:
            df['単勝_最人気差'] = np.nan
            df['単勝_2番人気差'] = np.nan
    else:
        for c in ['log単勝','単勝ランク','単勝pct','単勝_最人気差','単勝_2番人気差']:
            df[c] = np.nan
    return df

def ensure_feature_frame(df_feat: pd.DataFrame, feature_names: list[str], cat_cols: list[str]) -> pd.DataFrame:
    """
    学習時の 'feature_names' に完全一致するカラム構成を作る:
      - 足りない列は NaN を作る
      - 余計な列は落とす
      - 順番は feature_names に合わせる
      - カテゴリ列は category dtype にする
    """
    df = df_feat.copy()

    # 足りない列を作る
    for c in feature_names:
        if c not in df.columns:
            df[c] = np.nan

    # 余計な列は落とす + 並べ替え
    df = df.reindex(columns=feature_names)

    # dtype 調整（カテゴリ）
    df = _to_categorical(df, [c for c in cat_cols if c in df.columns])

    return df

# ============= 特徴量作成を強化 =============
BASE_FEATURES = [
    '枠番','斤量','騎手','馬体重','調教師','course_len','weather','race_type',
    'ground_state','競馬場','所属','月','馬体重_増減','性','年齢','出走頭数','相対枠位置',
    '大外枠','最内枠','馬_直近3走_複勝率','馬_直近5走_複勝率','馬_直近10走_複勝率','馬_直近5走_平均着順',
    '騎手_直近30走_複勝率','調教師_直近50走_複勝率','馬_コース適性_複勝率','馬_距離適性_複勝率',
    'コンビ_直近50走_複勝率','log単勝','単勝ランク','単勝pct','単勝_最人気差','単勝_2番人気差'
]
BASE_CAT_COLS = ['騎手','調教師','所属','性','競馬場','weather','race_type','ground_state']

def prepare_features(df_raw: pd.DataFrame,
                     feats_for_lgbm: list[str],
                     feats_for_cat: list[str],
                     cat_cols_lgbm: list[str],
                     cat_cols_cat: list[str],
                     src_url: str):
    df = add_market_features(df_raw.copy())

    for col in ['course_len','weather','race_type','ground_state','競馬場',
                '馬_直近3走_複勝率','馬_直近5走_複勝率','馬_直近10走_複勝率','馬_直近5走_平均着順',
                '騎手_直近30走_複勝率','調教師_直近50走_複勝率','馬_コース適性_複勝率','馬_距離適性_複勝率',
                'コンビ_直近50走_複勝率']:
        if col not in df.columns:
            df[col] = np.nan

    if '月' not in df.columns or df['月'].isna().all():
        mm = parse_month_from_url(src_url)
        df['月'] = float(mm) if mm is not None else np.nan

    # LGBM: “数値は数値”、カテゴリ予定列は “一旦string” にしておく（後でobjectへ）
    X_lgbm_native = ensure_feature_frame(df, feats_for_lgbm, [])
    for c in X_lgbm_native.columns:
        if c in cat_cols_lgbm:
            X_lgbm_native[c] = X_lgbm_native[c].astype('string')  # ←ここ重要
        else:
            X_lgbm_native[c] = pd.to_numeric(X_lgbm_native[c], errors='coerce')
    X_lgbm_native = X_lgbm_native.replace([np.inf, -np.inf], np.nan)

    # CatBoost: 数値は数値、カテゴリはstring(欠損NA)
    X_cat = ensure_feature_frame(df, feats_for_cat, [])
    for c in X_cat.columns:
        if c in cat_cols_cat:
            X_cat[c] = X_cat[c].astype('string').fillna('NA')
        else:
            X_cat[c] = pd.to_numeric(X_cat[c], errors='coerce')
    X_cat = X_cat.replace([np.inf, -np.inf], np.nan)

    # 最初は “学習時にカテゴリ扱いした列” を有効カテゴリとして返す
    cats_cat_eff = list(cat_cols_cat)
    cat_idx = [i for i, c in enumerate(X_cat.columns) if c in cats_cat_eff]
    
    return df, X_lgbm_native, X_cat, cat_idx, cats_cat_eff

def _get_model_categorical_names(lgbm_model, fallback: list[str] | None = None) -> list[str]:
    cats = getattr(lgbm_model, "categorical_feature_", None)
    feat_names = None
    if hasattr(lgbm_model, "feature_name_") and lgbm_model.feature_name_ is not None:
        feat_names = list(lgbm_model.feature_name_)
    elif hasattr(lgbm_model, "booster_") and lgbm_model.booster_ is not None:
        feat_names = list(lgbm_model.booster_.feature_name())

    if cats is None:
        return list(fallback or [])
    if isinstance(cats, (list, tuple)):
        # インデックスで持ってる場合は列名に変換
        if feat_names is not None and all(isinstance(x, (int, np.integer)) for x in cats):
            return [feat_names[int(i)] for i in cats]
        return list(cats)
    return list(fallback or [])


def _enforce_exact_categoricals_for_lgbm(X: pd.DataFrame, model_cat_names: list[str]) -> pd.DataFrame:
    X2 = X.copy()
    model_cat_set = set(model_cat_names)
    forced_cat, forced_noncat = [], []

    for c in X2.columns:
        if c in model_cat_set:
            if str(X2[c].dtype) != "category":
                X2[c] = X2[c].astype("string").astype("category")
                forced_cat.append(c)
        else:
            if str(X2[c].dtype) == "category":
                # 数値に戻せるなら戻す、無理なら object に
                try:
                    X2[c] = pd.to_numeric(X2[c].astype("string"), errors="raise")
                except Exception:
                    X2[c] = X2[c].astype("string")
                forced_noncat.append(c)

    if forced_cat or forced_noncat:
        logging.info(f"categoricals enforced -> to_category:{forced_cat} / to_noncategory:{forced_noncat}")
    return X2


def _dump_dtypes(df: pd.DataFrame, name: str):
    try:
        path = DEBUG_DIR / f"{name}_dtypes.txt"
        with open(path, "w", encoding="utf-8") as f:
            for c in df.columns:
                f.write(f"{c}\t{df[c].dtype}\n")
        logging.info(f"dtypes dumped: {path}")
    except Exception as e:
        logging.warning(f"dtypes dump failed: {e}")

def align_dtypes_for_lgbm(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # object にすべき列（学習時）
    obj_cols = ['騎手','調教師','所属','性','競馬場','weather','race_type','ground_state','月']
    for c in obj_cols:
        if c in df.columns:
            df[c] = df[c].astype(str)

    # float にすべき列
    if '馬体重' in df.columns:
        df['馬体重'] = df['馬体重'].astype(float)
    if '馬体重_増減' in df.columns:
        df['馬体重_増減'] = df['馬体重_増減'].astype(float)

    return df

def _enforce_numeric_except_cats(X: pd.DataFrame, cats: list[str]) -> pd.DataFrame:
    X = X.copy()
    non_cats = [c for c in X.columns if c not in cats]
    for c in non_cats:
        # 文字列やobjectが混ざっていても数値化。失敗は NaN
        X[c] = pd.to_numeric(X[c], errors="coerce")
    # ついでに inf を NaN へ
    X = X.replace([np.inf, -np.inf], np.nan)
    return X

import tempfile
def save_catboost_features_atomic(path: Path, features: list[str], cats_eff: list[str]):
    payload = {
        "features": features,
        "categorical_features": cats_eff
    }
    tmp = Path(str(path) + ".tmp")
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=str(path.parent)) as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        tmp_name = f.name
    os.replace(tmp_name, path)  # 原子的に置換
    logging.info(f"cats_eff saved -> {path} (cats={cats_eff})")
    
LGBM_CAT_COLS = ['騎手','調教師','所属','性','競馬場','weather','race_type','ground_state']

def load_lgbm_cat_levels(path=Path("03_lgbm_cat_levels.json")) -> dict[str, list[str]]:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj.get("levels", {})
    except Exception:
        return {}

def apply_lgbm_cat_levels(X: pd.DataFrame, levels: dict[str, list[str]]) -> pd.DataFrame:
    X = X.copy()
    for c in LGBM_CAT_COLS:
        if c not in X.columns: 
            continue
        lv = levels.get(c)
        # まず string 化（ここで NFKC とトリムも後述の正規化でやる）
        s = X[c].astype("string")
        if lv is not None:
            # 学習時と完全一致の dtype を付与（未知値は NaN になる）
            dtype = pd.api.types.CategoricalDtype(categories=lv, ordered=False)
            X[c] = s.astype(dtype)
        else:
            # レベルが無い列はとりあえず category 化
            X[c] = s.astype("category")
    return X

def _norm_for_lgbm(s: pd.Series) -> pd.Series:
    # 文字 -> NFKC、前後空白・中の連続空白も潰す
    s = s.astype("string").fillna("")
    s = s.map(lambda x: unicodedata.normalize("NFKC", x))
    s = s.str.strip()
    # 騎手名の中黒・全角スペースの揺れなどを単一スペースへ
    s = s.str.replace(r"\s+", " ", regex=True)
    return s.replace("", pd.NA)

# 列ごとの軽い統一（必要に応じて増やしてOK）
def normalize_for_lgbm(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    for c in X.columns:
        dt = str(X[c].dtype)
        if dt in ("object", "string", "category"):
            X[c] = X[c].astype("string").str.strip()
    return X

# --- LightGBM 用ヘルパー ---

def normalize_for_lgbm(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    for c in X.columns:
        dt = str(X[c].dtype)
        if dt in ("object", "string", "category"):
            X[c] = X[c].astype("string").str.strip()
    return X

def get_model_pandas_categories(lgbm) -> dict[str, list[str]]:
    """Booster 内に焼き付いた pandas_categorical を {feat: [levels...]} で返す"""
    levels = {}
    booster = getattr(lgbm, "booster_", None)
    if booster is None:
        return levels
    try:
        feat_names = list(booster.feature_name())
        pandas_cats = getattr(booster, "pandas_categorical", None)
        if pandas_cats is None:
            return levels
        for i, name in enumerate(feat_names):
            cats_i = pandas_cats[i] if i < len(pandas_cats) else None
            if cats_i is not None:
                levels[name] = [str(x) for x in cats_i]
    except Exception:
        pass
    return levels

def apply_exact_model_categories(
    X: pd.DataFrame,
    model_cat_map: dict[str, list[str]],
    lgbm_cat_cols: list[str]
) -> pd.DataFrame:
    """
    学習時に実際に使われたカテゴリ配列に“完全一致”させる。
    学習時にカテゴリ配列が空（=カテゴリとして使っていない）列は数値列として扱う。
    """
    X = X.copy()
    for c in lgbm_cat_cols:
        lv = model_cat_map.get(c, [])
        if len(lv) > 0:
            X[c] = X[c].astype("string").astype(pd.CategoricalDtype(categories=lv))
        else:
            # モデル側でカテゴリ配列が空 = 数値扱いだった
            X[c] = pd.to_numeric(X[c].astype("string"), errors="coerce")
    return X

def log_category_diff(X: pd.DataFrame, model_cat_map: dict[str, list[str]], lgbm_cat_cols: list[str]):
    for c in lgbm_cat_cols:
        model = model_cat_map.get(c, [])
        if len(model) == 0:
            logging.info(f"[LGBM] '{c}' is NUMERIC in model (no categories).")
            continue
        if str(X[c].dtype) == "category":
            now = list(X[c].cat.categories.astype(str))
        else:
            now = sorted(pd.Series(X[c].astype("string")).dropna().unique().tolist())
        if now == model:
            logging.info(f"[LGBM] categories OK at {c} (input={len(now)}, model={len(model)})")
        else:
            extra = sorted(set(now) - set(model))[:5]
            missing = sorted(set(model) - set(now))[:5]
            logging.info(f"[LGBM] categories MISMATCH at {c} (input={len(now)}, model={len(model)})")
            if extra:   logging.info(f"  extra_in_input(sample): {extra}")
            if missing: logging.info(f"  missing_in_input(sample): {missing}")

def _sync_booster_pandas_categories(lgbm, X: pd.DataFrame):
    # 今の入力で category dtype の列を拾う（順序は X の列順）
    cat_cols_current = [c for c in X.columns if str(X[c].dtype) == "category"]
    # その列ごとのカテゴリ配列を Booster へ渡す（len が cat 列数と一致する必要）
    pc = [list(X[c].cat.categories.astype(str)) for c in cat_cols_current]
    lgbm.booster_.pandas_categorical = pc
    
def _quick_diag(p_lgbm, p_cat, p_blend, name="diag"):
    import numpy as np, logging
    def s(a): 
        a = np.asarray(a)
        return f"min={np.nanmin(a):.3g}, max={np.nanmax(a):.3g}, mean={np.nanmean(a):.3g}"
    logging.info(f"[{name}] LGBM  : {s(p_lgbm)}")
    logging.info(f"[{name}] CAT   : {s(p_cat)} (gt1%={(np.asarray(p_cat)>0.01).sum()})")
    logging.info(f"[{name}] BLEND : {s(p_blend)}")
    
# ================== 推論本体（デバッグ出力強化） ==================
_lgbm_model = None
_cat_model  = None

def load_models():
    global _lgbm_model, _cat_model
    import joblib
    from catboost import CatBoostClassifier

    if _lgbm_model is None:
        if not MODEL_LGBM_PATH.exists():
            raise FileNotFoundError(f"LightGBMモデルが見つかりません: {MODEL_LGBM_PATH}")
        _lgbm_model = joblib.load(MODEL_LGBM_PATH)

    if _cat_model is None:
        if not MODEL_CAT_PATH.exists():
            raise FileNotFoundError(f"CatBoostモデルが見つかりません: {MODEL_CAT_PATH}")
        _cat_model = CatBoostClassifier()
        _cat_model.load_model(str(MODEL_CAT_PATH))
    return _lgbm_model, _cat_model

def predict_from_url(
    url: str,
    history_df: pd.DataFrame | None = None,
    w_lgbm: float = 0.5,
    w_cat: float = 0.5,
    topk_contrib: int = 5,
    weather: str | None = None,    
    ground_state: str | None = None,
) -> pd.DataFrame:
    """
    1) 出馬表をJRAから取得（馬名など表示用メタ含む）
    2) 学習時featureに完全整列した入力を作成（カテゴリdtypeも合わせる）
    3) LGBM / CatBoost 推論 → ブレンド
    4) 予測結果を“表示用メタ”と合体して返す（馬名を含む）
    """
    # 出馬表取得（history_df を渡すと馬名マッチの履歴特徴が付与される
    df0 = fetch_jra_shutuba(
    url,
    weather=weather,             # None のときは fetch 側で "晴" にフォールバック
    ground_state=ground_state,   # None のときは fetch 側で "良" にフォールバック
    history_df=history_df,
)
    if df0.empty:
        raise RuntimeError("出馬表の抽出に失敗しました。")
    debug_show_df(df0, "00a_scraped_immediate")

    # 特徴リスト＆カテゴリ列をロード（JSONにあればそれを最優先）
    feats_lgbm, cats_lgbm = load_feature_and_cats(LGBM_FEATS_JSON, BASE_FEATURES, BASE_CAT_COLS)
    feats_cat,  cats_cat  = load_feature_and_cats(CAT_FEATS_JSON,  BASE_FEATURES, BASE_CAT_COLS)

    # 特徴量生成（学習時 features に完全一致）
    df_feat, X_lgbm_native, X_cat, cat_idx, cats_cat_eff = prepare_features(
        df_raw=df0,
        feats_for_lgbm=feats_lgbm,
        feats_for_cat=feats_cat,
        cat_cols_lgbm=cats_lgbm,
        cat_cols_cat=cats_cat,
        src_url=url,
    )
    debug_show_df(df_feat,       "01_features_all")
    debug_show_df(X_lgbm_native, "02_lgbm_native")
    debug_show_df(X_cat,         "03_cat_native")

    # モデルロード
    lgbm, cat = load_models()

    # LightGBMの学習時feature順に再整列
    expected = None
    if hasattr(lgbm, "feature_name_") and lgbm.feature_name_ is not None:
        expected = list(lgbm.feature_name_)
    elif hasattr(lgbm, "booster_") and lgbm.booster_ is not None:
        expected = list(lgbm.booster_.feature_name())
    if expected is not None and len(expected) == X_lgbm_native.shape[1]:
        X_lgbm_native = X_lgbm_native.reindex(columns=expected)
        debug_show_df(X_lgbm_native, "02b_lgbm_native_aligned")

    # 学習時 categorical_feature 名（列名）をモデルから取得
    model_cat_names = _get_model_categorical_names(lgbm, fallback=cats_lgbm)

    # 文字列正規化
    levels = load_lgbm_cat_levels(Path("03_lgbm_cat_levels.json"))
    LGBM_CAT_COLS = ['騎手','調教師','所属','性','競馬場','weather','race_type','ground_state']

    # JSON にレベルがある列だけ category として扱う
    cat_cols_effective = [c for c in LGBM_CAT_COLS if levels.get(c)]
    num_cols_effective = [c for c in X_lgbm_native.columns if c not in cat_cols_effective]

    # 軽い正規化
    X_lgbm_native = normalize_for_lgbm(X_lgbm_native)

    # 数値列を強制数値化
    for c in num_cols_effective:
        X_lgbm_native[c] = pd.to_numeric(X_lgbm_native[c], errors="coerce")

    # カテゴリ列は学習時レベルで固定
    for c in cat_cols_effective:
        lv = levels[c]
        dtype = pd.api.types.CategoricalDtype(categories=lv, ordered=True)
        X_lgbm_native[c] = X_lgbm_native[c].astype("string").astype(dtype)

    # 念のため inf を NaN に
    X_lgbm_native = X_lgbm_native.replace([np.inf, -np.inf], np.nan)
    _dump_dtypes(X_lgbm_native, "02c_lgbm_after_enforce")

    model_cat_map = get_model_pandas_categories(lgbm)


    # 非カテゴリ列は数値化＋inf→NaN
    for c in X_lgbm_native.columns:
        if c not in LGBM_CAT_COLS:
            X_lgbm_native[c] = pd.to_numeric(X_lgbm_native[c], errors="coerce")
    X_lgbm_native = X_lgbm_native.replace([np.inf, -np.inf], np.nan)

    # ログ（任意）
    log_category_diff(X_lgbm_native, model_cat_map, LGBM_CAT_COLS)

    _sync_booster_pandas_categories(lgbm, X_lgbm_native)
    
    # 予測
    p_lgbm = np.asarray(lgbm.predict_proba(X_lgbm_native, validate_features=False))[:, 1]

    # 学習時 categorical_feature 名を取得
    model_cat_names = _get_model_categorical_names(lgbm, fallback=cats_lgbm)
    logging.info(f"model categorical_feature (names): {model_cat_names}")

    # ★ 推論直前の最終整形：LGBMには “カテゴリ列＝object(string)” で渡す
    # -------------------------------------------------
    num_cols = [c for c in X_lgbm_native.columns if c not in model_cat_names]
    for c in num_cols:
        X_lgbm_native[c] = pd.to_numeric(X_lgbm_native[c], errors="coerce")
    X_lgbm_native = X_lgbm_native.replace([np.inf, -np.inf], np.nan)
    # -------------------------------------------------
    _dump_dtypes(X_lgbm_native, "02c_lgbm_after_enforce")

    # 予測
    try:
        p_lgbm = np.asarray(lgbm.predict_proba(X_lgbm_native, validate_features=False))[:, 1]
    except Exception as e:
        logging.error(f"LGBM predict_proba failed: {e}")
        p_lgbm = np.zeros(len(X_lgbm_native), dtype=float)

    # ===== CatBoost =====
    from catboost import Pool, CatBoostError

    def _make_pool_for_cat(X_df: pd.DataFrame, cats: list[str]):
        X_local = X_df.copy()
        for c in X_local.columns:
            if c in cats:
                X_local[c] = X_local[c].astype("string").fillna("NA")
            else:
                X_local[c] = pd.to_numeric(X_local[c], errors="coerce")
        X_local = X_local.replace([np.inf, -np.inf], np.nan)
        cat_idx_local = [i for i, col in enumerate(X_local.columns) if col in cats]
        return X_local, Pool(X_local, cat_features=cat_idx_local)

    # 初期カテゴリは JSON の cats_cat（prepare_features から来た cats_cat_eff があるならそれを）
    cats_eff = list(cats_cat)

    # 最大5回まで「期待型に合わせて cats_eff を増減」して再試行
    for attempt in range(5):
        X_cat_cur, pool = _make_pool_for_cat(X_cat, cats_eff)
        try:
            proba_cat = cat.predict_proba(pool)
            break  # 成功！
        except CatBoostError as e:
            msg = str(e)
            m = re.search(r'Feature\s+(.+?)\s+is\s+(Categorical|Float)\s+in model', msg)
            if not m:
                raise
            feat = m.group(1).strip()
            kind = m.group(2)
            if kind == 'Categorical' and feat not in cats_eff:
                logging.info(f"CatBoost expects '{feat}' categorical -> add and retry")
                cats_eff.append(feat)
                continue
            if kind == 'Float' and feat in cats_eff:
                logging.info(f"CatBoost expects '{feat}' float -> remove and retry")
                cats_eff = [c for c in cats_eff if c != feat]
                continue
            # 期待と現状が矛盾してたら打ち切り
            raise
    else:
        # 5回やってもダメならそのまま例外に
        proba_cat = cat.predict_proba(pool)

    logging.info(f"cats_eff(final) = {cats_eff}")
    p_cat = (np.asarray(proba_cat)[:, 1]
            if np.asarray(proba_cat).ndim > 1
            else np.asarray(cat.predict(pool)).reshape(-1))

    # ✅ 最後に保存（原子的置換で安全に）
    save_catboost_features_atomic(
        CAT_FEATS_JSON,
        features=list(X_cat.columns),
        cats_eff=cats_eff
    )
    
    try:
        check = json.loads(CAT_FEATS_JSON.read_text(encoding="utf-8"))
        logging.info(f"saved cats = {check.get('categorical_features')}")
    except Exception as e:
        logging.warning(f"readback failed: {e}")
    
    p_cat = (np.asarray(proba_cat)[:, 1]
            if np.asarray(proba_cat).ndim > 1
            else np.asarray(cat.predict(pool)).reshape(-1))

    # ブレンドと市場確率・付加指標
    p_blend = w_lgbm * p_lgbm + w_cat * p_cat
    _quick_diag(p_lgbm, p_cat, p_blend, "post-blend")

    if "単勝" in df0 and df0["単勝"].notna().any():
        inv = df0["単勝"].apply(lambda x: 1 / x if pd.notna(x) and x > 0 else np.nan)
        total = np.nansum(inv)
        p_market = inv / total if total and total > 0 else np.nan
    else:
        p_market = pd.Series([np.nan] * len(df0))

    value_raw = p_blend - p_market.values
    value_pos = np.clip(value_raw, 0, None)
    vmax = value_pos.max() if np.isfinite(value_pos).any() else np.nan
    value_score = (value_pos / vmax) if (vmax and vmax > 0) else np.zeros_like(value_pos)

    # 馬の注目度（0-1）
    horse_attention = 0.6 * value_score + 0.4 * p_blend

    # 見送り率（参考）
    top_p = float(np.nanmax(p_blend)) if len(p_blend) > 0 else np.nan
    eps = 1e-12
    pm = p_market.fillna(0).values
    ent = -np.nansum(pm * np.log(pm + eps))
    ent_max = math.log(len(df0)) if len(df0) > 1 else 1.0
    ent_norm = ent / ent_max if ent_max > 0 else 0.0
    race_skip_rate = 0.6 * (1 - top_p) + 0.4 * ent_norm
    logging.info(f"race_skip_rate={race_skip_rate:.3f}")
    attention_score = 1 - race_skip_rate

    # SHAP寄与
    try:
        shap_lgbm = np.array(lgbm.predict(X_lgbm_native, pred_contrib=True))[:, :-1]
        lgbm_feat_names = list(X_lgbm_native.columns)
    except Exception:
        shap_lgbm, lgbm_feat_names = None, []

    try:
        shap_cat_full = cat.get_feature_importance(pool, type="ShapValues")
        shap_cat = np.array(shap_cat_full)[:, :-1]
        cat_feat_names = list(X_cat.columns)
    except Exception:
        shap_cat, cat_feat_names = None, []

    contrib_list = []
    for i in range(len(df0)):
        parts = {}
        if shap_lgbm is not None:
            for k, v in zip(lgbm_feat_names, shap_lgbm[i]):
                parts.setdefault(k, []).append(abs(float(v)))
        if shap_cat is not None:
            for k, v in zip(cat_feat_names, shap_cat[i]):
                parts.setdefault(k, []).append(abs(float(v)))
        agg = [(k, float(np.mean(vs))) for k, vs in parts.items()]
        agg_sorted = sorted(agg, key=lambda x: x[1], reverse=True)
        contrib_list.append(", ".join([f"{k}:{v:.3f}" for k, v in agg_sorted[:topk_contrib]]))

    # 8) 表示用に“メタ + 予測”を合体（馬名など残す）
    out = df0.copy()
    out["p_lgbm"] = p_lgbm
    out["p_cat"] = p_cat
    out["p_blend"] = p_blend
    out["market_p"] = p_market.values
    out["注目度_馬"]  = horse_attention
    out["寄与TOP"] = contrib_list
    out.attrs["レース注目度"] = attention_score

    # デバッグ保存
    debug_show_df(out, "10_pred_out")

    # 表示カラムを整えて返す（馬名を含める）
    SHOW_COLS = [
        "枠番","馬番","馬名","性","年齢","斤量","騎手","調教師",
        "馬体重","馬体重_増減","単勝","人気",
        "p_blend","p_lgbm","p_cat","market_p","高配当率","寄与TOP"
    ]
    show = [c for c in SHOW_COLS if c in out.columns]
    return out[show].sort_values(["枠番","馬番"]).reset_index(drop=True)

# ======================================
# ============ Streamlit UI ============
# ======================================

import streamlit as st

def load_baselines(path="03_baselines.json"):
    p = Path(path)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}

# 事前に学習データの最新日付を取得しておく（無ければ NaT）
try:
    _rd = pd.read_pickle("04_results_df.pickle")
    if "date" in _rd.columns:
        dates = pd.to_datetime(
            _rd["date"].astype(str)
            .str.replace("年","-", regex=False)
            .str.replace("月","-", regex=False)
            .str.replace("日","",  regex=False),
            errors="coerce"
        )
        latest_training_date = dates.max()
    else:
        latest_training_date = pd.NaT
except Exception:
    latest_training_date = pd.NaT

st.set_page_config(page_title="高配当チェッカー", layout="wide")
st.title("高配当馬券チェッカー")

with st.sidebar:
    st.header("入力")
    url = st.text_input(
        "JRA 出馬表URL",
        value="",
        help="JRA公式の出馬表ページURLを貼り付け"
    )
    col_w = st.columns(2)
    with col_w[0]:
        weather = st.selectbox("天候", ["自動","晴","曇","小雨","雨","小雪","雪"], index=0)
    with col_w[1]:
        ground = st.selectbox("馬場状態", ["自動","良","稍","重","不良"], index=0)

    st.divider()
    if pd.notna(latest_training_date):
        st.caption(f"学習データの最終日: {latest_training_date:%Y-%m-%d}")
    else:
        st.caption("学習データの最終日: 不明")

    st.subheader("ブレンド重み")
    w_lgbm = st.slider("LGBM重み", 0.0, 1.0, 0.5, 0.05)
    w_cat  = 1.0 - w_lgbm
    st.caption(f"CatBoost重み（1 - LGBM重み）: {w_cat:.2f}")

    st.divider()
    st.subheader("表示しきい値（利用者向け）")
    score_threshold = st.number_input("妙味スコアの下限（≧1を推奨）", min_value=0.0, value=1.0, step=0.1)

st.info("URLと条件を入れて、下のボタンを押してください。")
go = st.button("推論する", type="primary")

if go and url.strip():
    # 推論の実行
    kw = {}
    if weather != "自動": kw["weather"] = weather
    if ground  != "自動": kw["ground_state"] = ground

    try:
        df_pred = predict_from_url(url, w_lgbm=w_lgbm, w_cat=w_cat, **kw)
    except Exception as e:
        st.error(f"推論に失敗しました: {e}")
        st.stop()

    # p_blend が無ければ以降の計算ができないため安全停止
    if "p_blend" not in df_pred.columns:
        st.error("出力に p_blend が見つかりません。予測処理（predict_from_url）の出力列を確認してください。")
        st.stop()

    # 妙味スコアの組み立て
    eps = 1e-12
    has_market = ("market_p" in df_pred.columns) and df_pred["market_p"].notna().any()
    if has_market:
        df_pred["妙味スコア"] = df_pred["p_blend"] / (df_pred["market_p"] + eps)
    else:
        if "高配当率" in df_pred.columns:
            df_pred["妙味スコア"] = 1.0 + 9.0 * df_pred["高配当率"]
        else:
            df_pred["妙味スコア"] = df_pred["p_blend"] / (df_pred["p_blend"].mean() + eps)

    # 見送り率が無ければフォールバックで再計算 → 注目度 = 1 - 見送り率
    def _fallback_race_skip(df: pd.DataFrame) -> float:
        top_p = float(np.nanmax(df["p_blend"])) if len(df) else np.nan
        eps = 1e-12
        if "market_p" in df.columns and df["market_p"].notna().any():
            pm = df["market_p"].fillna(0).values
        else:
            pm = np.zeros(len(df), dtype=float)
        ent = -np.nansum(pm * np.log(pm + eps))
        ent_max = math.log(len(df)) if len(df) > 1 else 1.0
        ent_norm = ent / ent_max if ent_max > 0 else 0.0
        return 0.6 * (1 - (top_p if top_p == top_p else 0.0)) + 0.4 * ent_norm

    race_skip = df_pred.attrs.get("race_skip_rate", float("nan"))
    try:
        race_skip = float(race_skip)
    except Exception:
        race_skip = float("nan")
    if not np.isfinite(race_skip):
        race_skip = _fallback_race_skip(df_pred)

    race_attention = max(0.0, min(1.0, 1.0 - race_skip))

    # ベースライン読込（平均/中央値 両方）
    base = load_baselines()
    base_att_avg = base.get("avg_race_attention_oof")
    base_att_med = base.get("median_race_attention_oof")
    base_myo_avg = base.get("avg_myoumi_score_oof")
    base_myo_med = base.get("median_myoumi_score_oof")

    # 今回レースの平均/中央値
    myoumi_mean   = float(np.nanmean(df_pred["妙味スコア"])) if "妙味スコア" in df_pred else float("nan")
    myoumi_median = float(np.nanmedian(df_pred["妙味スコア"])) if "妙味スコア" in df_pred else float("nan")

    c1, c2, c3 = st.columns([1, 1.2, 2.2])
    with c1:
        st.metric(
            "レース注目度",
            f"{race_attention:.1%}",
            delta=(f"{(race_attention - base_att_avg):+.1%}" if isinstance(base_att_avg,(int,float)) else None),
            delta_color="normal"
        )
    with c2:
        st.metric(
            "妙味スコア",
            f"{myoumi_mean:.2f}",
            delta=(f"{(myoumi_mean - base_myo_avg):+.2f}" if isinstance(base_myo_avg,(int,float)) else None),
            delta_color="normal"
        )
    with c3:
        if base:
            st.caption(
                f"学習ベースライン：注目度 平均/中央値 = "
                f"{(f'{base_att_avg:.1%}' if isinstance(base_att_avg,(int,float)) else 'N/A')} / "
                f"{(f'{base_att_med:.1%}' if isinstance(base_att_med,(int,float)) else 'N/A')}｜"
                f"妙味スコア 平均/中央値 = "
                f"{(f'{base_myo_avg:.2f}' if isinstance(base_myo_avg,(int,float)) else 'N/A')} / "
                f"{(f'{base_myo_med:.2f}' if isinstance(base_myo_med,(int,float)) else 'N/A')} "
                f"(races={base.get('n_races','?')}, horses={base.get('n_horses','?')})"
            )
        else:
            st.caption("学習ベースラインは未計算です（03_modeling.ipynbで 03_baselines.json を出力してください）")

    # 2タブ構成： 利用者向け / 検証用
    tab_user, tab_debug = st.tabs(["👤 利用者向け", "🔧 検証用"])

    # ---------- 利用者向け ----------
    with tab_user:
        st.subheader("妙味のある馬（しきい値以上のみ表示）")
        cols_user = [
            "馬番","馬名","騎手","調教師","単勝","人気",
            "p_blend","妙味スコア","寄与TOP"
        ]
        show_cols = [c for c in cols_user if c in df_pred.columns]
        user_df = df_pred.copy()
        user_df = user_df[show_cols].sort_values("妙味スコア", ascending=False)
        user_df = user_df[user_df["妙味スコア"] >= score_threshold]

        if user_df.empty:
            st.warning("しきい値を満たす馬がいません。しきい値を下げるか条件を見直してください。")
        else:
            st.dataframe(user_df, use_container_width=True, hide_index=True)

        with st.expander("指標の説明", expanded=False):
            st.markdown(
                "- **注目度**: 1 − 見送り率。トップ確率と市場の不確実性から算出（高いほど良い）\n"
                "- **p_blend**: LGBM と CatBoost の予測を重み平均した勝利確率\n"
                "- **market_p**: 単勝オッズから計算した市場の暗黙確率（※オッズが無いときは未計算）\n"
                "- **妙味スコア**: `p_blend / market_p`。1を超えるほど“市場より割安”と見る指標\n"
                "- **寄与TOP**: 予測に効いた上位特徴の要約（SHAP等の近似）\n"
            )

    # ---------- 検証用 ----------
    with tab_debug:
        st.subheader("推論データ一式（検証用）")
        st.dataframe(df_pred, use_container_width=True, hide_index=True)

else:
    st.stop()