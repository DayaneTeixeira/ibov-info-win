import requests
import numpy as np
import pandas as pd
import json
import math
from datetime import datetime
import os

WINDOWS  = [11, 21, 63, 252]
LAMBDA   = 0.94

def fetch_brapi(token=None):
    headers = {"User-Agent": "Mozilla/5.0"}
    params  = {"range": "2y", "interval": "1d", "fundamental": "false"}
    if token:
        params["token"] = token
    r = requests.get("https://brapi.dev/api/quote/%5EBVSP", params=params, headers=headers, timeout=15)
    r.raise_for_status()
    hist = r.json()["results"][0]["historicalDataPrice"]
    df = pd.DataFrame(hist)
    close_col = "close" if "close" in df.columns else "adjclose"
    df = df[["date", close_col]].rename(columns={close_col: "Close"}).dropna()
    df = df[df["Close"] > 0]
    df["date"] = pd.to_datetime(df["date"], unit="s", errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").set_index("date")
    print(f"   brapi: {len(df)} pregões, último={df.index[-1].date()}, close={df['Close'].iloc[-1]:.0f}")
    return df

def fetch_yfinance():
    import yfinance as yf
    df = yf.Ticker("^BVSP").history(period="2y", interval="1d")[["Close"]].dropna()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df[df["Close"] > 0].sort_index()

def safe(v, dec=4):
    """Float seguro para JSON (NaN/Inf → None)."""
    if v is None: return None
    try:
        f = float(v)
        return None if (math.isnan(f) or math.isinf(f)) else round(f, dec)
    except: return None

# ── FÓRMULA EXATA DA PLANILHA ─────────────────────────────────────────────────
# H = LN( (MAX(cumsum) - MIN(cumsum)) / STDEV(retornos) ) / LN(10)
def hurst_planilha(log_rets):
    r = np.array(log_rets, dtype=float)
    if len(r) < 3: return None
    media   = r.mean()
    ajust   = r - media
    cumsum  = np.cumsum(ajust)
    R       = cumsum.max() - cumsum.min()
    S       = r.std(ddof=1)
    if S <= 0 or R <= 0: return None
    return safe(math.log(R / S) / math.log(10))

# ── VOL EWMA (fórmula da planilha) ────────────────────────────────────────────
# Linha 1: var = ret² ; demais: var = 0.94*var_ant + 0.06*ret²
# Vol a.a. = SQRT(var_última) * SQRT(252)
def ewma_vol(log_rets):
    r = np.array(log_rets, dtype=float)
    if len(r) < 2: return None
    var = r[0]**2
    for x in r[1:]:
        var = LAMBDA * var + (1 - LAMBDA) * x**2
    return safe(math.sqrt(var) * math.sqrt(252) * 100, 2)

def vol_regime(vol):
    if vol is None: return "N/A"
    if vol < 14.7:  return "Comprimida"
    if vol < 18.2:  return "Normal"
    if vol < 19.8:  return "Stress Elevado"
    return "Stress Extremo"

def classify_hurst(h):
    if h is None: return "N/A"
    if h < 0.45:   return "REVERSÃO"
    if h > 0.55:   return "TENDÊNCIA"
    return "ALEATÓRIO"

def sinal_desc(regime_h, vol):
    """Descrição exata da planilha."""
    v = vol if vol is not None else 0
    if regime_h == "ALEATÓRIO":
        return "ALEATÓRIO", "→ Sem vantagem direcional · operar menor"
    if regime_h == "TENDÊNCIA":
        if v > 18.2:
            return "TENDÊNCIA + STRESS", "→ Seguir tendência · gain amplo · stop firme"
        return "TENDÊNCIA + NORMAL", "→ Melhor cenário para seguir tendência"
    # REVERSÃO
    if v > 18.2:
        return "REVERSÃO + STRESS", "→ Operar contra movimento · stop curto · realizar rápido"
    return "REVERSÃO + NORMAL", "→ Operar contra extremos · alvo menor"

def percentil_vol(vol, all_vols):
    """Percentil da vol atual em relação ao histórico (igual PERCENTRANK do Excel)."""
    if vol is None or len(all_vols) == 0: return None
    below = sum(1 for v in all_vols if v < vol)
    return round(below / len(all_vols) * 100)

def range_pts(close, vol):
    if vol is None or close is None: return {}
    daily = close * (vol / 100) / math.sqrt(252)
    return {
        "meio":   int(round(daily * 0.5)),
        "um_sig": int(round(daily)),
        "um5":    int(round(daily * 1.5)),
    }

# ─────────────────────────────────────────────────────────────────────────────
def main():
    token = os.environ.get("BRAPI_TOKEN")
    print("📡 Buscando dados via brapi.dev...")
    try:
        df = fetch_brapi(token)
    except Exception as e:
        print(f"   ⚠️  brapi falhou ({e}), tentando yfinance...")
        df = fetch_yfinance()
        print(f"   ✅ yfinance: {len(df)} pregões")

    df["log_ret"] = np.log(df["Close"] / df["Close"].shift(1))
    df = df.dropna()
    log_rets_all = df["log_ret"].values

    # Histórico de vol EWMA para calcular percentil (janela 63d como referência)
    hist_vols = []
    for i in range(63, len(df)):
        seg = log_rets_all[i-63:i]
        v = ewma_vol(seg)
        if v is not None:
            hist_vols.append(v)

    results = []
    min_p = max(WINDOWS) + 5

    for i in range(min_p, len(df)):
        close    = safe(float(df["Close"].iloc[i]), 2)
        date_str = df.index[i].strftime("%Y-%m-%d")
        row = {"date": date_str, "close": close, "janelas": {}}

        for w in WINDOWS:
            if i < w: continue
            seg = log_rets_all[i-w:i]
            h   = hurst_planilha(seg)
            v   = ewma_vol(seg)
            rh  = classify_hurst(h)
            rv  = vol_regime(v)
            s, sd = sinal_desc(rh, v)
            pct = percentil_vol(v, hist_vols)
            rng = range_pts(close, v)

            row["janelas"][str(w)] = {
                "hurst":      h,
                "vol_ewma":   v,
                "regime_h":   rh,
                "regime_v":   rv,
                "percentil":  pct,
                "sinal":      s,
                "sinal_desc": sd,
                "range":      rng,
            }

        results.append(row)

    results = results[-500:]
    output  = {
        "updated_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "ticker": "^BVSP", "name": "IBOVESPA",
        "latest": results[-1] if results else {},
        "data":   results,
    }

    os.makedirs("docs", exist_ok=True)
    with open("docs/data.json", "w") as f:
        json.dump(output, f, separators=(",", ":"),
                  default=lambda x: None if isinstance(x, float) and (math.isnan(x) or math.isinf(x)) else x)

    l = results[-1]
    print(f"\n✅ {len(results)} pregões · Último: {l['date']} | Fechamento: {l['close']}")
    for w in WINDOWS:
        j = l["janelas"].get(str(w), {})
        print(f"   {w:>3}d → H={j.get('hurst')}  Vol={j.get('vol_ewma')}%  P{j.get('percentil')}  [{j.get('regime_h')} + {j.get('regime_v')}]")

if __name__ == "__main__":
    main()
