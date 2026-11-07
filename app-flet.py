# app-flet.py
# オプション戦略 損益グラフ（BS 付き）＋ yfinance 原資産
# Flet版

import io
import math
import os
import platform
import tempfile
from dataclasses import dataclass
from enum import Enum
from typing import List, Literal

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from scipy.stats import norm
import yfinance as yf
import flet as ft

import re
import unicodedata

OptionType = Literal["call", "put"]
SideType = Literal["long", "short"]

XSP_SCALE = 0.1
XSP_DEFAULT_STEP = 1.0   # XSPは通常・ウィークリーとも$1刻み
US_STOCK_DEFAULT_STEP = 5.0   # 米国個別株は通常5.0。ウィークリーは刻み幅欄で2.5に変更


def sanitize_numeric_text(val) -> str:
    normalized = unicodedata.normalize("NFKC", str(val or ""))
    return re.sub(r"[^0-9.-]", "", normalized)


def sanitize_numeric_input(e):
    if e and getattr(e, "control", None):
        e.control.value = sanitize_numeric_text(e.control.value)
        e.control.update()
        return e.control.value
    return ""


def safe_float(val, default=0.0) -> float:
    try:
        def_val = float(default) if default is not None else 0.0
    except (ValueError, TypeError):
        def_val = 0.0
    if not math.isfinite(def_val):
        def_val = 0.0

    if val is None:
        return def_val
    if isinstance(val, (int, float)):
        result = float(val)
        return result if math.isfinite(result) else def_val
    s = str(val).strip()
    if not s:
        return def_val
    s = sanitize_numeric_text(s)
    try:
        result = float(s)
        return result if math.isfinite(result) else def_val
    except (ValueError, TypeError):
        cleaned = s
        if cleaned.count(".") > 1:
            parts = cleaned.split(".")
            cleaned = parts[0] + "." + "".join(parts[1:])
        try:
            result = float(cleaned)
            return result if math.isfinite(result) else def_val
        except (ValueError, TypeError):
            return def_val


def safe_int(val, default=1) -> int:
    try:
        def_val = int(default) if default is not None else 1
    except (ValueError, TypeError):
        def_val = 1
    f = safe_float(val, default=float(def_val))
    return int(round(f))


def normalize_ticker_for_yfinance(raw: str) -> str:
    """
    ユーザー入力を yfinance 用の正式ティッカーに正規化する。
    例:
      "7203"  -> "7203.T"
      "7203.T" -> "7203.T"
      "n225"  -> "^N225"
      "topx" / "tpx" -> "^TOPX" （好みで変更可）
    """
    if not raw:
        return raw

    s = raw.strip()
    u = s.upper()

    # XSPはyfinanceに無い → SPXを取得し、別途1/10する
    if is_xsp(s):
        return "^GSPC"

    # すでに ^ や .T などが付いていれば、そのまま使う
    if u.startswith("^"):
        return s
    if u.endswith(".T"):
        return s

    # 日経225
    if u in ("N225", "NI225", "NK225", "NKI"):
        return "^N225"

    # TOPIX
    if u in ("TOPX", "TOPIX", "TPX"):
        return "^TOPX"  # 必要に応じて他のコードに変更

    # --- 米インデックス関連 ---
    # S&P500 index (SPX), S&P500 index mini (XSP)
    # XSPはSPXの1/10スケールなので、同じ^GSPCを参照
    if u in ("SPX", "SPX INDEX", "INX", "SP500"):
        return "^GSPC"

    # NASDAQ100
    if u in ("NDX", "NASDAQ100"):
        return "^NDX"

    # Russell 2000
    if u in ("RUT", "RUSSELL2000"):
        return "^RUT"

    # VIX
    if u in ("VIX", "VIX INDEX"):
        return "^VIX"

    # 数字だけなら日本株コードとみなして .T を付ける
    if u.isdigit():
        return f"{u}.T"

    # デフォルトはそのまま返す
    return s


def is_us_ticker(ticker: str) -> bool:
    """
    ティッカーが米国銘柄（ドル建て）かどうかを判定する。
    """
    if not ticker:
        return False
    
    u = ticker.upper().strip()
    
    # 日本株や日本の指数は円建て
    if u.endswith(".T") or "N225" in u or "TOPIX" in u or "TPX" in u:
        return False
    
    # 日本の株式コード（4桁の数字のみ）は円建て
    if u.isdigit() and len(u) == 4:
        return False
    
    # 米国指数
    if any(x in u for x in ["^GSPC", "^NDX", "^RUT", "SPX", "XSP", "NDX", "RUT", "VIX", "^VIX"]):
        return True
    
    # その他（米国個別株やETF）はドル建てと判定
    # 注：これは簡易的な判定。より正確には取引所情報が必要
    return True


def get_currency_symbol(ticker: str) -> str:
    """
    ティッカーに対応する通貨記号を返す。
    """
    return "$" if is_us_ticker(ticker) else "円"


def format_currency(value: float, ticker: str) -> str:
    """
    値を通貨フォーマットで表示する。
    ドル: $1,000 または -$1,000
    円: 1,000円 または -1,000円
    """
    if is_us_ticker(ticker):
        if value < 0:
            return f"-${abs(value):,.0f}"
        else:
            return f"${value:,.0f}"
    else:
        return f"{value:,.0f}円"


def get_last_close(ticker: str) -> float | None:
    try:
        norm = normalize_ticker_for_yfinance(ticker)
        data = yf.Ticker(norm).history(period="5d")
        if data.empty:
            return None
        return float(data["Close"].iloc[-1])
    except Exception:
        return None


def get_underlying_price(raw: str):
    """(換算後価格, 換算前の終値) を返す。XSPは SPX/10"""
    last = get_last_close(normalize_ticker_for_yfinance(raw))
    if last is None:
        return None, None
    return (last * XSP_SCALE if is_xsp(raw) else last), last


def guess_contract_size_from_ticker(ticker: str) -> int:
    if is_xsp(ticker):          # 乗数$100(SPXと同じ)
        return 100
    # 正規化してから判定
    norm = normalize_ticker_for_yfinance(ticker)
    t = norm.upper()
    
    if "N225" in t:
        if "MICRO" in t:
            return 10
        if "MINI" in t:
            return 100
        return 1000
    
    # SPX / NDX / RUT など米インデックス
    if any(x in t for x in ("SPX", "^GSPC", "^NDX", "NDX", "^RUT", "RUT")):
        return 100

    # ETF / 米個別株
    if not t.endswith(".T"):
        return 100

    return 100


def guess_strike_step_from_ticker(ticker: str, price: float = None) -> float:
    if is_xsp(ticker):
        return XSP_DEFAULT_STEP
    # 正規化してから判定（ただしXSPは元の入力もチェック）
    norm = normalize_ticker_for_yfinance(ticker)
    t = norm.upper()
    orig_t = ticker.upper()

    # --- 日本指数 ---
    if "TOPIX" in t or "TPX" in t or "^TOPX" in t:
        return 25.0
    if "N225" in t or "^N225" in t:
        return 125.0

    # --- 米国指数（SPX/XSP/NDX/RUT など） ---

    # SPX / ^GSPC
    if any(x in t for x in ("SPX", "^GSPC")):
        return 5.0  # SPXは通常5ドル刻み

    # NDX, RUT
    if any(x in t for x in ("^NDX", "NDX", "^RUT", "RUT")):
        return 5.0

    # --- ETF・米個別株（SPY, QQQ, AAPL など） ---
    if not t.endswith(".T"):
        return stock_step_from_price(price)

    # --- 日本個別株 (.T) - オプションのストライク刻みは100円が標準 ---
    # ※現物株の呼値とオプションのストライク刻みは異なる
    return 100.0


def round_to_step(x: float, step: float) -> float:
    if step <= 0:
        return x
    return round(round(x / step) * step, 4)


def stock_step_from_price(price: float | None) -> float:
    """米国株式・ETFのストライク刻みを株価から推定する(目安)。"""
    if price is None or price <= 0:
        return US_STOCK_DEFAULT_STEP
    if price < 50:
        return 0.5
    if price < 200:
        return 1.0
    if price < 500:
        return 2.5
    return 5.0

def bs_price(
    option_type: OptionType,
    S: np.ndarray,
    K: float,
    r: float,
    sigma: float,
    T: float
) -> np.ndarray:
    S = np.array(S, dtype=float)

    if T <= 0 or sigma <= 0:
        if option_type == "call":
            intrinsic = np.maximum(S - K, 0.0)
        else:
            intrinsic = np.maximum(K - S, 0.0)
        return intrinsic

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def is_xsp(raw) -> bool:
    # "XSP", "xsp", "^XSP", "$XSP" を同一視
    if not isinstance(raw, str):
        return False
    return raw.strip().upper().lstrip("^$") == "XSP"


def is_us_stock(raw) -> bool:
    """米国個別株・ETF(AAPL, NVDA, SPY など)か。指数・日本株・XSPは除く。"""
    if not isinstance(raw, str) or not raw.strip():
        return False
    u = raw.strip().upper()
    if u.startswith(("^", "$")) or u.endswith(".T") or u.isdigit():
        return False
    if u in ("SPX", "SPXW", "NDX", "RUT", "VIX", "INX", "SP500", "NASDAQ100",
            "N225", "NI225", "NK225", "NKI", "TOPX", "TOPIX", "TPX", "XSP"):
        return False
    return True

def uses_decimals(raw) -> bool:
    """小数表示(価格2桁・行使価格1桁)にするティッカーか。"""
    return is_xsp(raw) or is_us_stock(raw)


def fmt_k(k: float, ticker: str = "") -> str:
    """行使価格の表示用。XSPは小数点第1位、それ以外は整数(端数があれば小数点第1位)。"""
    if is_xsp(ticker):
        return f"{k:.1f}"
    return f"{k:.0f}" if float(k).is_integer() else f"{k:.1f}"

def theo_premium(opt, K, S, r, sigma, T_days) -> float:
    """BS理論価格をプレミアム初期値にする(価格水準に依存しない)"""
    v = float(bs_price(opt, np.array([S], dtype=float), K, r, sigma, T_days / 365.0)[0])
    return round(v, 2 if S < 1000 else 1)


@dataclass
class OptionPosition:
    option_type: OptionType
    side: SideType
    strike: float
    premium: float
    quantity: int = 1
    contract_size: int = 1
    r: float = 0.0
    sigma: float = 0.2
    T: float = 0.1


    def payoff_at_maturity(self, S: np.ndarray) -> np.ndarray:
        if self.option_type == "call":
            intrinsic = np.maximum(S - self.strike, 0.0)
        else:
            intrinsic = np.maximum(self.strike - S, 0.0)

        if self.side == "long":
            pl_per_contract = intrinsic - self.premium
        else:
            pl_per_contract = -intrinsic + self.premium

        return pl_per_contract * self.quantity * self.contract_size

    def pl_now_bs(self, S: np.ndarray) -> np.ndarray:
        theo = bs_price(
            option_type=self.option_type,
            S=S,
            K=self.strike,
            r=self.r,
            sigma=self.sigma,
            T=self.T
        )

        if self.side == "long":
            position_value_now = +theo
            initial_cf_per_contract = -self.premium
        else:
            position_value_now = -theo
            initial_cf_per_contract = +self.premium

        pl_per_contract = position_value_now + initial_cf_per_contract
        return pl_per_contract * self.quantity * self.contract_size


def get_tag_style_for_position(pos: OptionPosition):
    """
    Call/Put × Long/Short で、ストライクタグの色と塗り分けを決める。
    """
    if pos.option_type == "call":
        base_color = "#22c55e"  # 緑
    else:
        base_color = "#ef4444"  # 赤

    if pos.side == "short":
        # ショートは塗りつぶし＋白文字
        facecolor = base_color
        edgecolor = "white"
        textcolor = "white"
    else:
        # ロングは枠線だけ＋色文字
        facecolor = "none"
        edgecolor = base_color
        textcolor = base_color

    return facecolor, edgecolor, textcolor


def total_payoff_maturity(positions: List[OptionPosition], S: np.ndarray) -> np.ndarray:
    total = np.zeros_like(S, dtype=float)
    for p in positions:
        total += p.payoff_at_maturity(S)
    return total


def total_pl_now_bs(positions: List[OptionPosition], S: np.ndarray) -> np.ndarray:
    total = np.zeros_like(S, dtype=float)
    for p in positions:
        total += p.pl_now_bs(S)
    return total


class StrategyPreset(str, Enum):
    NONE = "手動で設定する"
    LONG_CALL = "ロング・コール（強気）"
    SHORT_CALL = "ショート・コール（弱気〜横ばい）"
    LONG_PUT = "ロング・プット（弱気）"
    SHORT_PUT = "ショート・プット（強気〜横ばい）"
    BULL_CALL_SPREAD = "ブル・コール・スプレッド（緩やかな強気）"
    BEAR_PUT_SPREAD = "ベア・プット・スプレッド（緩やかな弱気）"
    BULL_PUT_SPREAD = "ブル・プット・スプレッド（クレジット・強気〜横ばい）"
    BEAR_CALL_SPREAD = "ベア・コール・スプレッド（クレジット・弱気〜横ばい）"
    ATM_BULL_PUT_CREDIT_SPREAD = "ATM ブル・プット・クレジット・スプレッド"
    ATM_BEAR_CALL_CREDIT_SPREAD = "ATM ベア・コール・クレジット・スプレッド"
    LONG_STRADDLE = "ロング・ストラドル（変動大きめ予想）"
    SHORT_STRADDLE = "ショート・ストラドル（変動小さめ予想・高リスク）"
    LONG_STRANGLE = "ロング・ストラングル（大化け・急変狙い、低コスト）"
    SHORT_STRANGLE = "ショート・ストラングル（買いなし・上級者向け）"
    IRON_CONDOR = "アイアン・コンドル（広めレンジでコツコツ）"
    IRON_BUTTERFLY = "アイアン・バタフライ（狭いレンジをピンポイント）"
    COVERED_CALL = "カバード・コール（現物保有＋コール売り）"
    PROTECTIVE_PUT = "プロテクティブ・プット（現物下落の保険）"
    COLLAR = "カラー（低コストの現物保全）"
    CASH_SECURED_PUT = "キャッシュ・セキュアード・プット（安値での指値買い代用）"
    PUT_RATIO_SPREAD = "プット・レシオ・スプレッド（限定的な下落〜横ばい）"
    CALENDAR_SPREAD = "カレンダー・スプレッド（満期ズレ・時間価値狙い）"
    JADE_LIZARD = "ジェイド・リザード（レンジ〜やや強気・上値限定）"
    SAFE_JADE_LIZARD = "ジェイド・リザード（安全版・両サイド損失限定）"


def resolve_strategy_preset(val: str | StrategyPreset | None) -> StrategyPreset:
    if not val:
        return StrategyPreset.NONE
    if isinstance(val, StrategyPreset):
        return val
    # メンバー名で検索 (例: "LONG_CALL")
    if val in StrategyPreset.__members__:
        return StrategyPreset[val]
    # 値で検索 (例: "ロング・コール（強気）")
    for p in StrategyPreset:
        if p.value == val or p.name == val:
            return p
    return StrategyPreset.NONE


def build_positions_from_preset(
    preset: StrategyPreset | str,
    spot_center: float,
    contract_size: int,
    r: float,
    ticker: str,
    step: float | None = None,
) -> List[OptionPosition]:
    preset = resolve_strategy_preset(preset)
    positions: List[OptionPosition] = []

    sigma = 0.2
    T_short = 30 / 365
    T_long = 90 / 365

    step = step or guess_strike_step_from_ticker(ticker, spot_center)
    k_atm = round_to_step(spot_center, step)
    k_down1 = k_atm - step
    k_up1 = k_atm + step
    k_low = round_to_step(spot_center * 0.95, step)
    k_high = round_to_step(spot_center * 1.05, step)
    k_very_low = round_to_step(spot_center * 0.9, step)
    k_very_high = round_to_step(spot_center * 1.1, step)

    # プレミアム価格のスケーリング（原資産価格に比例）
    # 日本株（N225=32000円前後）を基準として、300円のプレミアムを使用
    # 米国株（SPX=6000ドル, XSP=600ドル）は適切にスケーリング
    base_price = 32000.0  # 日本株基準価格
    base_premium_atm = 300.0  # ATMオプションの基準プレミアム
    scale = spot_center / base_price
    
    # スケーリングしたプレミアム価格
    p_atm = round(base_premium_atm * scale, 1)
    p_high = round(p_atm * 1.33, 1)  # 400 相当
    p_mid = round(p_atm * 0.87, 1)   # 260 相当
    p_low = round(p_atm * 0.67, 1)   # 200 相当
    p_vlow = round(p_atm * 0.6, 1)   # 180 相当
    p_xlow = round(p_atm * 0.47, 1)  # 140 相当
    p_vhigh = round(p_atm * 1.17, 1) # 350 相当
    p_xhigh = round(p_atm * 0.83, 1) # 250 相当
    p_med = round(p_atm * 0.73, 1)   # 220 相当
    p_sml = round(p_atm * 0.5, 1)    # 150 相当
    p_xsml = round(p_atm * 0.4, 1)   # 120 相当
    p_xxhigh = round(p_atm * 0.8, 1) # 240 相当

    if preset == StrategyPreset.LONG_CALL:
        positions.append(
            OptionPosition("call", "long", k_atm, p_atm, 1, contract_size, r, sigma, T_short)
        )
    elif preset == StrategyPreset.SHORT_CALL:
        positions.append(
            OptionPosition("call", "short", k_atm, p_atm, 1, contract_size, r, sigma, T_short)
        )
    elif preset == StrategyPreset.LONG_PUT:
        positions.append(
            OptionPosition("put", "long", k_atm, p_atm, 1, contract_size, r, sigma, T_short)
        )
    elif preset == StrategyPreset.SHORT_PUT:
        positions.append(
            OptionPosition("put", "short", k_atm, p_atm, 1, contract_size, r, sigma, T_short)
        )
    elif preset == StrategyPreset.BULL_CALL_SPREAD:
        positions.append(
            OptionPosition("call", "long", k_low, p_high, 1, contract_size, r, sigma, T_short)
        )
        positions.append(
            OptionPosition("call", "short", k_high, p_low, 1, contract_size, r, sigma, T_short)
        )
    elif preset == StrategyPreset.BEAR_PUT_SPREAD:
        positions.append(
            OptionPosition("put", "long", k_high, p_high, 1, contract_size, r, sigma, T_short)
        )
        positions.append(
            OptionPosition("put", "short", k_low, p_low, 1, contract_size, r, sigma, T_short)
        )
    elif preset == StrategyPreset.BULL_PUT_SPREAD:
        positions.append(
            OptionPosition("put", "short", k_high, p_high, 1, contract_size, r, sigma, T_short)
        )
        positions.append(
            OptionPosition("put", "long", k_low, p_low, 1, contract_size, r, sigma, T_short)
        )
    elif preset == StrategyPreset.BEAR_CALL_SPREAD:
        positions.append(
            OptionPosition("call", "short", k_low, p_high, 1, contract_size, r, sigma, T_short)
        )
        positions.append(
            OptionPosition("call", "long", k_high, p_low, 1, contract_size, r, sigma, T_short)
        )
    elif preset == StrategyPreset.ATM_BULL_PUT_CREDIT_SPREAD:
        # atm ブル・プット・クレジット・スプレッド
        # 現在価格付近(ATM)のプットを売り、その1ストライク下のプットを買う
        positions.append(
            OptionPosition("put", "short", k_atm, p_mid, 1, contract_size, r, sigma, T_short)
        )
        positions.append(
            OptionPosition("put", "long", k_down1, p_xlow, 1, contract_size, r, sigma, T_short)
        )
    elif preset == StrategyPreset.ATM_BEAR_CALL_CREDIT_SPREAD:
        # atm ベア・コール・クレジット・スプレッド
        # 現在価格付近(ATM)のコールを売り、その1ストライク上のコールを買う
        positions.append(
            OptionPosition("call", "short", k_atm, p_mid, 1, contract_size, r, sigma, T_short)
        )
        positions.append(
            OptionPosition("call", "long", k_up1, p_xlow, 1, contract_size, r, sigma, T_short)
        )
    elif preset == StrategyPreset.LONG_STRADDLE:
        positions.append(
            OptionPosition("call", "long", k_atm, p_atm, 1, contract_size, r, sigma, T_short)
        )
        positions.append(
            OptionPosition("put", "long", k_atm, p_atm, 1, contract_size, r, sigma, T_short)
        )
    elif preset == StrategyPreset.SHORT_STRADDLE:
        positions.append(
            OptionPosition("call", "short", k_atm, p_atm, 1, contract_size, r, sigma, T_short)
        )
        positions.append(
            OptionPosition("put", "short", k_atm, p_atm, 1, contract_size, r, sigma, T_short)
        )
    elif preset == StrategyPreset.LONG_STRANGLE:
        positions.append(
            OptionPosition("call", "long", k_high, p_vlow, 1, contract_size, r, sigma, T_short)
        )
        positions.append(
            OptionPosition("put", "long", k_low, p_vlow, 1, contract_size, r, sigma, T_short)
        )
    elif preset == StrategyPreset.SHORT_STRANGLE:
        positions.append(
            OptionPosition("call", "short", k_high, p_xhigh, 1, contract_size, r, sigma, T_short)
        )
        positions.append(
            OptionPosition("put", "short", k_low, p_xhigh, 1, contract_size, r, sigma, T_short)
        )
    elif preset == StrategyPreset.IRON_CONDOR:
        positions.append(
            OptionPosition("put", "short", k_low, p_low, 1, contract_size, r, sigma, T_short)
        )
        positions.append(
            OptionPosition("put", "long", k_very_low, round(p_low * 0.5, 1), 1, contract_size, r, sigma, T_short)
        )
        positions.append(
            OptionPosition("call", "short", k_high, p_low, 1, contract_size, r, sigma, T_short)
        )
        positions.append(
            OptionPosition("call", "long", k_very_high, round(p_low * 0.5, 1), 1, contract_size, r, sigma, T_short)
        )
    elif preset == StrategyPreset.IRON_BUTTERFLY:
        positions.append(
            OptionPosition("call", "short", k_atm, p_atm, 1, contract_size, r, sigma, T_short)
        )
        positions.append(
            OptionPosition("put", "short", k_atm, p_atm, 1, contract_size, r, sigma, T_short)
        )
        positions.append(
            OptionPosition("call", "long", k_high, p_sml, 1, contract_size, r, sigma, T_short)
        )
        positions.append(
            OptionPosition("put", "long", k_low, p_sml, 1, contract_size, r, sigma, T_short)
        )
    elif preset == StrategyPreset.COVERED_CALL:
        positions.append(
            OptionPosition("call", "short", k_high, p_low, 1, contract_size, r, sigma, T_short)
        )
    elif preset == StrategyPreset.PROTECTIVE_PUT:
        positions.append(
            OptionPosition("put", "long", k_low, p_low, 1, contract_size, r, sigma, T_short)
        )
    elif preset == StrategyPreset.COLLAR:
        positions.append(
            OptionPosition("put", "long", k_low, p_sml, 1, contract_size, r, sigma, T_short)
        )
        positions.append(
            OptionPosition("call", "short", k_high, p_sml, 1, contract_size, r, sigma, T_short)
        )
    elif preset == StrategyPreset.CASH_SECURED_PUT:
        positions.append(
            OptionPosition("put", "short", k_low, p_xhigh, 1, contract_size, r, sigma, T_short)
        )
    elif preset == StrategyPreset.PUT_RATIO_SPREAD:
        positions.append(
            OptionPosition("put", "long", k_high, p_vhigh, 1, contract_size, r, sigma, T_short)
        )
        positions.append(
            OptionPosition("put", "short", k_low, p_med, 2, contract_size, r, sigma, T_short)
        )
    elif preset == StrategyPreset.CALENDAR_SPREAD:
        positions.append(
            OptionPosition("call", "short", k_atm, p_low, 1, contract_size, r, sigma, T_short)
        )
        positions.append(
            OptionPosition("call", "long", k_atm, p_xhigh, 1, contract_size, r, sigma, T_long)
        )
    elif preset == StrategyPreset.JADE_LIZARD:
        # シンプルなジェイド・リザード：
        # プット売り（k_low）＋ショート・コール・スプレッド（k_high〜k_very_high）
        positions.append(
            OptionPosition("put", "short", k_low, p_mid, 1, contract_size, r, sigma, T_short)
        )
        positions.append(
            OptionPosition("call", "short", k_high, p_med, 1, contract_size, r, sigma, T_short)
        )
        positions.append(
            OptionPosition("call", "long", k_very_high, p_xsml, 1, contract_size, r, sigma, T_short)
        )
    elif preset == StrategyPreset.SAFE_JADE_LIZARD:
        # 安全版ジェイド・リザード：
        # ジェイド・リザード＋遠いプット買いで下もスプレッド化（両サイド損失限定）
        positions.append(
            OptionPosition("put", "short", k_low, p_xxhigh, 1, contract_size, r, sigma, T_short)
        )
        positions.append(
            OptionPosition("put", "long", k_very_low, p_xsml, 1, contract_size, r, sigma, T_short)
        )
        positions.append(
            OptionPosition("call", "short", k_high, p_med, 1, contract_size, r, sigma, T_short)
        )
        positions.append(
            OptionPosition("call", "long", k_very_high, p_xsml, 1, contract_size, r, sigma, T_short)
        )

    for p in positions:
        p.premium = theo_premium(p.option_type, p.strike, spot_center, r, p.sigma, p.T * 365)

    return positions


async def main(page: ft.Page):
    page.title = "オプション戦略 損益グラフ（BS 付き）"
    page.scroll = "adaptive"
    page.padding = 20

    page.window.width = 1440
    page.window.height = 900
    page.window.update()

    await page.window.center()

    # 状態管理
    state = {
        "ticker_input": "^N225",   # ユーザーが入力したそのまま
        "ticker_yf": "^N225",      # yfinance に投げる用
        "ticker": "^N225",
        "spot_center": 32000.0,
        "spot_range": 4000.0,
        "spot_step": 100.0,
        "contract_size": 1000,
        "r": 0.0075,
        "preset": StrategyPreset.NONE,
        "n_legs": 2,
        "legs": {},
        "prev_ticker": None,
        "prev_spot_center": None,
        "prev_preset": None,
        # --- ここから What-if 用 ---
        "what_if_enabled": True,
        "what_if_price": 32000.0,
        "what_if_days": 30,
        "what_if_iv": 0.20,   # 20%
    }

    def fmt_spot(x: float) -> str:
        """原資産価格の表示用。XSP・米国株式は小数点第2位、それ以外は整数。"""
        if uses_decimals(state.get("ticker_input", "")):
            return f"{x:,.2f}"
        return f"{x:,.0f}"
    
    # UI要素の参照
    ticker_field = ft.TextField(
        label="原資産ティッカー（例: ^N225, 7203.T）",
        value=state["ticker"],
        width=270
    )

    auto_fetch_check = ft.Checkbox(
        label="ティッカーから中心価格を自動取得する",
        value=True
    )

    spot_center_field = ft.TextField(
        label="中心となる原資産価格",
        value=str(state["spot_center"]),
        keyboard_type=ft.KeyboardType.NUMBER,
        width=200,
        on_change=sanitize_numeric_input,
    )

    yfinance_info = ft.Text("")

    spot_range_field = ft.TextField(
        label="上下レンジ幅",
        value=str(state["spot_range"]),
        keyboard_type=ft.KeyboardType.NUMBER,
        width=200
    )

    spot_step_field = ft.TextField(
        label="刻み幅",
        value=str(state["spot_step"]),
        keyboard_type=ft.KeyboardType.NUMBER,
        width=200,
        on_change=sanitize_numeric_input,
    )

    contract_size_field = ft.TextField(
        label="取引単位",
        value=str(state["contract_size"]),
        keyboard_type=ft.KeyboardType.NUMBER,
        width=200,
        on_change=sanitize_numeric_input,
    )

    r_field = ft.TextField(
        label="無リスク金利 r（年率, 例: 0.0075 = 0.75%）",
        value=str(state["r"]),
        keyboard_type=ft.KeyboardType.NUMBER,
        width=200,
        on_change=sanitize_numeric_input,
    )

    legs_slider = ft.Slider(
        min=1,
        max=6,
        divisions=5,
        value=2,
        label="{value}",
        width=300
    )

    legs_container = ft.Column()
    strike_dials_container = ft.Column()  # ストライクダイヤル用
    chart_container = ft.Column()
    results_text = ft.Text("")

    # What-if UI
    what_if_enabled_check = ft.Checkbox(
        label="What-if（時間・IV）シナリオを表示",
        value=state["what_if_enabled"]
    )

    # 値表示用テキスト
    what_if_price_value_text = ft.Text("", width=110)
    what_if_days_value_text = ft.Text("", width=110)
    what_if_iv_value_text = ft.Text("", width=110)

    what_if_price_slider = ft.Slider(
        min=0.1,
        max=state["spot_center"] * 2,
        value=state["spot_center"],
        label="",           # ここは空でもOK（下でまとめて管理）
        expand=True,
    )

    what_if_days_slider = ft.Slider(
        min=0,
        max=365,
        value=state["what_if_days"],
        label="",
        expand=True,
    )

    what_if_iv_slider = ft.Slider(
        min=0.01,
        max=1.0,
        value=state["what_if_iv"],
        label="",
        expand=True,
    )

    what_if_pl_text = ft.Text("", size=16, weight=ft.FontWeight.BOLD)


    def update_what_if_value_labels():
        price = state.get("what_if_price", state.get("spot_center", 0.0))
        if uses_decimals(state.get("ticker_input", "")):
            what_if_price_slider.label = "{value:.2f}"
        else:
            what_if_price_slider.label = "{value:.0f}"
        what_if_price_value_text.value = fmt_spot(price)

        # 日数は「◯日」
        days = state.get("what_if_days", 0)
        what_if_days_slider.label = "{value}"
        what_if_days_value_text.value = f"{days} 日"

        # IV はパーセント表示（例 0.25 → 25.0 %）
        iv = state.get("what_if_iv", 0.2)
        what_if_iv_slider.label = "{value:.2f}"
        what_if_iv_value_text.value = f"{iv*100:.2f} %"


    def reset_what_if_from_legs():
        """
        現在のレッグ設定から、what-if の初期値を「いい感じ」に決め直す。
        - 価格: 現在の中心価格
        - 日数: レッグの T_days の平均
        - IV  : レッグの σ の平均
        """
        # 価格
        base_price = state.get("spot_center", 0.0)

        # 残存日数
        t_list = [safe_int(leg.get("T_days", 30), 30) for leg in state["legs"].values()]
        if t_list:
            base_days = int(round(sum(t_list) / len(t_list)))
        else:
            base_days = 30

        # インプライド・ボラティリティ
        s_list = [safe_float(leg.get("sigma", 0.2), 0.2) for leg in state["legs"].values()]
        if s_list:
            base_iv = float(sum(s_list) / len(s_list))
        else:
            base_iv = 0.2

        # 状態を更新
        state["what_if_price"] = base_price
        state["what_if_days"] = base_days
        state["what_if_iv"] = base_iv

        # スライダーにも反映
        what_if_price_slider.min = state["spot_center"] - state["spot_range"]
        what_if_price_slider.max = state["spot_center"] + state["spot_range"]
        what_if_price_slider.value = base_price

        what_if_days_slider.value = base_days
        what_if_iv_slider.value = base_iv

        update_what_if_label_texts()
        update_what_if_value_labels() 

    def update_what_if_label_texts():
        if uses_decimals(state.get("ticker_input", "")):
            what_if_price_slider.label = "{value:.2f}"
        else:
            what_if_price_slider.label = "{value:.0f}"
        what_if_days_slider.label = "{value}日"
        what_if_iv_slider.label = "{value:.2f}"

    update_what_if_label_texts()

    def on_what_if_enabled_change(e):
        state["what_if_enabled"] = bool(e.control.value)
        generate_chart()

    def on_what_if_price_change(e):
        val = safe_float(e.control.value, default=state["spot_center"])
        state["what_if_price"] = val
        update_what_if_value_labels()
        generate_chart()

    def on_what_if_days_change(e):
        val = safe_int(e.control.value, default=state["what_if_days"])
        state["what_if_days"] = max(0, val)
        update_what_if_value_labels()
        generate_chart()

    def on_what_if_iv_change(e):
        val = safe_float(e.control.value, default=state["what_if_iv"])
        state["what_if_iv"] = max(0.01, min(val, 3.0))
        update_what_if_value_labels()
        generate_chart()

    what_if_enabled_check.on_change = on_what_if_enabled_change
    what_if_price_slider.on_change = on_what_if_price_change
    what_if_days_slider.on_change = on_what_if_days_change
    what_if_iv_slider.on_change = on_what_if_iv_change

    def on_iv_crash(e):
        # 各レッグの現在σの平均から 50% 減らす例
        sigmas = [safe_float(leg.get("sigma", 0.2), 0.2) for leg in state["legs"].values()]
        if sigmas:
            avg_sigma = float(sum(sigmas) / len(sigmas))
        else:
            avg_sigma = 0.2
        crashed = max(avg_sigma * 0.5, 0.01)
        state["what_if_iv"] = crashed
        what_if_iv_slider.value = crashed
        update_what_if_label_texts()
        update_what_if_value_labels()
        generate_chart()

    iv_crash_button = ft.OutlinedButton(
        "IVクラッシュを再現",
        icon=ft.Icons.FLASH_ON,
        on_click=on_iv_crash,
    )

    def on_what_if_reset(e):
        reset_what_if_from_legs()
        generate_chart()

    what_if_reset_button = ft.OutlinedButton(
        "What-if をリセット",
        icon=ft.Icons.RESTART_ALT,
        on_click=on_what_if_reset,
    )


    what_if_card = ft.Card(
        content=ft.Container(
            content=ft.Column([
                ft.Row([
                    ft.Text("What-if：時間とボラティリティ", size=18, weight=ft.FontWeight.BOLD),
                    what_if_enabled_check,
                ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                ft.Text("ドラッグして満期前（現時点）の損益カーブの変化を確認できます。", size=12),

                # 価格
                ft.Row([
                    ft.Text("価格", width=80),
                    what_if_price_slider,
                    what_if_price_value_text,   # ← 現在値
                ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),

                # 満期までの日数
                ft.Row([
                    ft.Text("満期までの日数", width=120),
                    what_if_days_slider,
                    what_if_days_value_text,
                ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),

                # インプライド・ボラティリティ
                ft.Row([
                    ft.Text("インプライド・ボラティリティ", width=150),
                    what_if_iv_slider,
                    what_if_iv_value_text,
                ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),

                ft.Row([
                    iv_crash_button,
                    what_if_reset_button,
                    ft.Container(expand=True),
                    ft.Text("予想損益: "),
                    what_if_pl_text,
                ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
            ]),
            padding=15,
        )
    )


    update_what_if_value_labels()


    # 横軸範囲コントロール用UI
    range_info_text = ft.Text(
        f"表示範囲: {state['spot_center'] - state['spot_range']:,.0f} 〜 {state['spot_center'] + state['spot_range']:,.0f}（中心: {state['spot_center']:,.0f} ±{state['spot_range']:,.0f}）",
        size=14,
        weight=ft.FontWeight.BOLD,
        color=ft.Colors.BLUE_GREY_700,
    )

    range_slider = ft.Slider(
        min=10,
        max=50000,
        value=state["spot_range"],
        label="{value}",
        expand=True,
    )

    def update_range_slider_bounds(target_val: float):
        # 目標レンジ幅に合わせて min と max を動的に適正化
        cur_min = min(float(range_slider.min), target_val * 0.2, 50.0)
        cur_min = max(5.0, round(cur_min / 5.0) * 5.0)
        cur_max = max(float(range_slider.max), target_val * 2.5, 1000.0)
        range_slider.min = cur_min
        range_slider.max = cur_max
        range_slider.value = max(cur_min, min(target_val, cur_max))

    def update_range_display():
        center = state["spot_center"]
        rng = state["spot_range"]
        range_info_text.value = (
            f"表示範囲: {fmt_spot(center - rng)} 〜 {fmt_spot(center + rng)}"
            f"（中心: {fmt_spot(center)} ±{fmt_spot(rng)}）"
        )

    def set_auto_range(e=None):
        update_state_from_fields()
        strikes = []
        for leg in state["legs"].values():
            if "strike" in leg:
                try:
                    strikes.append(float(leg["strike"]))
                except (ValueError, TypeError):
                    pass
        center = state["spot_center"]
        # 元のユーザー入力を使用
        ticker_for_step = state.get("ticker_input", state["ticker"])
        strike_step = state["spot_step"]
        if strikes:
            max_dist = max([abs(k - center) for k in strikes] + [strike_step * 2])
            auto_range = float(round_to_step(max_dist * 1.5, strike_step))
            auto_range = max(auto_range, strike_step * 3)
        else:
            auto_range = max(float(round_to_step(center * 0.15, strike_step)), strike_step * 5)

        state["spot_range"] = auto_range
        spot_range_field.value = str(auto_range)
        update_range_slider_bounds(auto_range)
        update_range_display()
        build_strike_dials_ui()
        generate_chart()

    def set_quick_range(val: float):
        update_state_from_fields()
        state["spot_range"] = val
        spot_range_field.value = str(val)
        update_range_slider_bounds(val)
        update_range_display()
        build_strike_dials_ui()
        generate_chart()

    def set_quick_range_percent(pct: float):
        update_state_from_fields()
        center = state["spot_center"]
        # 元のユーザー入力を使用
        ticker_for_step = state.get("ticker_input", state["ticker"])
        strike_step = state["spot_step"]
        val = float(round_to_step(center * (pct / 100.0), strike_step))
        val = max(val, strike_step * 2)
        set_quick_range(val)

    def on_range_slider_change(e):
        val = float(e.control.value)
        state["spot_range"] = val
        spot_range_field.value = str(val)
        update_range_display()
        build_strike_dials_ui()
        generate_chart()

    range_slider.on_change = on_range_slider_change

    range_control_card = ft.Card(
        content=ft.Container(
            content=ft.Column([
                ft.Row([
                    ft.Row([
                        ft.Icon(ft.Icons.AUTO_GRAPH, color=ft.Colors.PRIMARY),
                        ft.Text("横軸表示範囲（原資産価格 S）", weight=ft.FontWeight.BOLD, size=16),
                    ]),
                    range_info_text,
                ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                ft.Row([
                    ft.Text("レンジ幅: ±"),
                    range_slider,
                ]),
                ft.Row([
                    ft.FilledTonalButton(
                        "全体表示（自動調整）",
                        icon=ft.Icons.FIT_SCREEN,
                        on_click=set_auto_range
                    ),
                    ft.OutlinedButton("±5%", on_click=lambda e: set_quick_range_percent(5)),
                    ft.OutlinedButton("±10%", on_click=lambda e: set_quick_range_percent(10)),
                    ft.OutlinedButton("±20%", on_click=lambda e: set_quick_range_percent(20)),
                    ft.OutlinedButton("±35%", on_click=lambda e: set_quick_range_percent(35)),
                ], wrap=True),
            ]),
            padding=15
        )
    )

    def update_state_from_fields():

        v = safe_float(spot_step_field.value, 0.0)
        if v > 0:
            state["spot_step"] = v

        # ティッカーは正規化して保存（update_ticker_info と同じロジック）
        raw = ticker_field.value or "^N225"
        state["ticker_input"] = raw
        norm = normalize_ticker_for_yfinance(raw)
        state["ticker_yf"] = norm
        state["ticker"] = norm
        
        try:
            state["spot_center"] = float(spot_center_field.value)
        except (ValueError, TypeError):
            pass
        try:
            state["spot_range"] = float(spot_range_field.value)
        except (ValueError, TypeError):
            pass
        try:
            v = safe_float(spot_step_field.value, 0.0)
            if v > 0:
                state["spot_step"] = v
        except (ValueError, TypeError):
            pass
        try:
            state["contract_size"] = int(float(contract_size_field.value))
        except (ValueError, TypeError):
            pass
        try:
            state["r"] = float(r_field.value)
        except (ValueError, TypeError):
            pass


    def make_default_leg(center, step):
        k = round_to_step(center, step)
        return {"type": "call", "side": "long", "strike": k,
                "premium": theo_premium("call", k, center, state["r"], 0.2, 30),
                "quantity": 1, "sigma": 0.2, "T_days": 30}


    def update_ticker_info(e=None, force_k_update=True, manual_center=None):
        raw = ticker_field.value or "^N225"
        ticker_changed = (raw != state.get("prev_ticker"))
        state["prev_ticker"] = raw
        state["ticker_input"] = raw
        norm = normalize_ticker_for_yfinance(raw)
        state["ticker_yf"] = norm
        state["ticker"] = norm

        old_center = state.get("spot_center", 32000.0)

        # 1. 中心価格
        if manual_center is None and auto_fetch_check.value:
            price, raw_close = get_underlying_price(raw)
            if price is not None:
                if is_xsp(raw):
                    new_center = round(price, 2)
                    yfinance_info.value = (f"SPX 終値: 約 {raw_close:,.2f} → "
                                        f"XSP換算(1/10): 約 {price:,.2f}")
                elif uses_decimals(raw):
                    new_center = round(price, 2)
                    yfinance_info.value = f"yfinance 終値: 約 {price:,.2f}"
                else:
                    new_center = float(round(price, -1) if price > 1000 else round(price, 1))
                    yfinance_info.value = f"yfinance 終値: 約 {price:,.2f}"
                state["spot_center"] = new_center
                spot_center_field.value = f"{new_center:.2f}" if uses_decimals(raw) else str(new_center)
            else:
                yfinance_info.value = "価格を取得できませんでした"
        else:
            state["spot_center"] = (manual_center if manual_center is not None
                                    else safe_float(spot_center_field.value, old_center))
        center = state["spot_center"]

        # 2. 取引単位・刻み(ティッカー変更時のみ推定)
        if ticker_changed:
            state["contract_size"] = guess_contract_size_from_ticker(raw)
            contract_size_field.value = str(state["contract_size"])
            state["spot_step"] = guess_strike_step_from_ticker(raw, center)
            spot_step_field.value = str(state["spot_step"])
        strike_step = state["spot_step"]

        # 3. レンジ幅(既存どおり、strike_step を使う)
        suggested_range = max(float(round_to_step(center * 0.15, strike_step)), strike_step * 5)
        if state["spot_range"] >= center or abs(center - old_center) > old_center * 0.1:
            state["spot_range"] = suggested_range
            spot_range_field.value = str(suggested_range)
            update_range_slider_bounds(suggested_range)
        else:
            update_range_slider_bounds(state["spot_range"])

        # 4. レッグ再構築
        if force_k_update:
            if state.get("preset", StrategyPreset.NONE) != StrategyPreset.NONE:
                preset_positions = build_positions_from_preset(
                    preset=state["preset"],
                    spot_center=center,
                    contract_size=state["contract_size"],
                    r=state["r"],
                    ticker=state["ticker_input"],
                    step=state["spot_step"],
                )
                state["n_legs"] = len(preset_positions)
                legs_slider.value = state["n_legs"]
                state["legs"] = {
                    i: {"type": p.option_type, "side": p.side, "strike": p.strike,
                        "premium": p.premium, "quantity": p.quantity,
                        "sigma": p.sigma, "T_days": int(round(p.T * 365))}
                    for i, p in enumerate(preset_positions, start=1)}
            else:
                ratio = center / old_center if old_center > 0 else 1.0
                rescale = abs(ratio - 1.0) > 0.03
                for i in range(1, state["n_legs"] + 1):
                    leg = state["legs"].get(i)
                    if leg is None:
                        state["legs"][i] = make_default_leg(center, strike_step)
                        continue
                    if rescale:
                        leg["strike"] = round_to_step(leg.get("strike", old_center) * ratio, strike_step)
                        leg["premium"] = theo_premium(leg["type"], leg["strike"], center,
                                                    state["r"], leg["sigma"], leg["T_days"])
                    elif "strike" not in leg:
                        leg["strike"] = round_to_step(center, strike_step)

        # 5. 再構築・再描画(既存どおり)
        build_legs_ui()
        update_range_display()
        reset_what_if_from_legs()
        generate_chart()


    def on_preset_change(e=None):
        val = None
        if e is not None:
            if hasattr(e, "control") and e.control and hasattr(e.control, "value") and e.control.value:
                val = e.control.value
            elif hasattr(e, "data") and e.data:
                val = e.data
        if not val and "preset_dropdown" in locals():
            val = preset_dropdown.value

        selected_preset = resolve_strategy_preset(val)
        state["preset"] = selected_preset
        
        if selected_preset != StrategyPreset.NONE:
            update_state_from_fields()
            preset_positions = build_positions_from_preset(
                preset=selected_preset,
                spot_center=state["spot_center"],
                contract_size=state["contract_size"],
                r=state["r"],
                ticker=state["ticker_input"],
                step=state["spot_step"],
            )
            state["n_legs"] = len(preset_positions)
            legs_slider.value = state["n_legs"]
            
            # レッグデータを初期化
            state["legs"] = {}
            for i, pos in enumerate(preset_positions, start=1):
                state["legs"][i] = {
                    "type": pos.option_type,
                    "side": pos.side,
                    "strike": pos.strike,
                    "premium": pos.premium,
                    "quantity": pos.quantity,
                    "sigma": pos.sigma,
                    "T_days": int(round(pos.T * 365))
                }
            
            # UIを再構築して全体が収まる自動レンジで再描画
            build_legs_ui()
            set_auto_range()
        else:
            page.update()

    preset_dropdown = ft.Dropdown(
        label="戦略プリセット",
        options=[
            ft.dropdown.Option(key=p.name, text=p.value)
            for p in StrategyPreset
        ],
        value=StrategyPreset.NONE.name,
        width=400,
    )

    def on_legs_change(e):
        state["n_legs"] = int(legs_slider.value)
        build_legs_ui()
        page.update()

    def build_legs_ui():
        legs_container.controls.clear()
        
        for i in range(1, state["n_legs"] + 1):
            if i not in state["legs"]:
                state["legs"][i] = make_default_leg(state["spot_center"], state["spot_step"])
            
            leg_data = state["legs"][i]
            
            type_dropdown = ft.Dropdown(
                label="種類",
                options=[
                    ft.dropdown.Option("call", "Call"),
                    ft.dropdown.Option("put", "Put"),
                ],
                value=leg_data["type"],
                width=150,
            )
            type_handler = lambda e, idx=i: update_leg(idx, "type", e.control.value if (e and hasattr(e, "control") and e.control) else type_dropdown.value)
            for ev in ["on_select", "on_change", "on_selected"]:
                if hasattr(type_dropdown, ev):
                    setattr(type_dropdown, ev, type_handler)
            
            side_dropdown = ft.Dropdown(
                label="売買",
                options=[
                    ft.dropdown.Option("long", "Long"),
                    ft.dropdown.Option("short", "Short"),
                ],
                value=leg_data["side"],
                width=150,
            )
            side_handler = lambda e, idx=i: update_leg(idx, "side", e.control.value if (e and hasattr(e, "control") and e.control) else side_dropdown.value)
            for ev in ["on_select", "on_change", "on_selected"]:
                if hasattr(side_dropdown, ev):
                    setattr(side_dropdown, ev, side_handler)
            
            def sync_dials(e):
                build_strike_dials_ui()
                generate_chart()
            
            strike_field = ft.TextField(
                label="行使価格 K",
                value=fmt_k(leg_data["strike"], state["ticker_input"]),
                keyboard_type=ft.KeyboardType.NUMBER,
                width=150,
                on_change=lambda e, idx=i: update_leg(
                    idx, "strike", safe_float(sanitize_numeric_input(e), default=state["legs"].get(idx, {}).get("strike", 0.0))
                ),
                on_blur=sync_dials,
                on_submit=sync_dials,
            )
            
            premium_field = ft.TextField(
                label="プレミアム",
                value=f"{leg_data['premium']:.2f}",  # 小数点2桁で表示
                keyboard_type=ft.KeyboardType.NUMBER,
                width=150,
                on_change=lambda e, idx=i: update_leg(
                    idx, "premium", safe_float(sanitize_numeric_input(e), default=state["legs"].get(idx, {}).get("premium", 0.0))
                )
            )
            
            quantity_field = ft.TextField(
                label="枚数",
                value=str(leg_data["quantity"]),
                keyboard_type=ft.KeyboardType.NUMBER,
                width=100,
                on_change=lambda e, idx=i: update_leg(
                    idx, "quantity", safe_int(sanitize_numeric_input(e), default=state["legs"].get(idx, {}).get("quantity", 1))
                )
            )
            
            sigma_field = ft.TextField(
                label="ボラティリティ σ",
                value=str(leg_data["sigma"]),
                keyboard_type=ft.KeyboardType.NUMBER,
                width=150,
                on_change=lambda e, idx=i: update_leg(
                    idx, "sigma", safe_float(sanitize_numeric_input(e), default=state["legs"].get(idx, {}).get("sigma", 0.2))
                )
            )
            
            t_days_field = ft.TextField(
                label="残存日数（日）",
                value=str(leg_data["T_days"]),
                keyboard_type=ft.KeyboardType.NUMBER,
                width=150,
                on_change=lambda e, idx=i: update_leg(
                    idx, "T_days", safe_int(sanitize_numeric_input(e), default=state["legs"].get(idx, {}).get("T_days", 30))
                )
            )
            
            leg_card = ft.Card(
                content=ft.Container(
                    content=ft.Column([
                        ft.Text(f"レッグ {i}", weight=ft.FontWeight.BOLD),
                        ft.Row([type_dropdown, side_dropdown, strike_field]),
                        ft.Row([premium_field, quantity_field, sigma_field]),
                        ft.Row([t_days_field]),
                    ]),
                    padding=15
                )
            )
            
            legs_container.controls.append(leg_card)

        build_strike_dials_ui()


    def build_strike_dials_ui():
        strike_dials_container.controls.clear()
        if not state["legs"]:
            return
        step = state["spot_step"]
        lo = math.ceil((state["spot_center"] - state["spot_range"]) / step - 1e-9) * step
        hi = math.floor((state["spot_center"] + state["spot_range"]) / step + 1e-9) * step
        if hi <= lo:
            hi = lo + step
        divisions = max(1, round((hi - lo) / step))

        for i in range(1, state["n_legs"] + 1):
            leg = state["legs"].get(i)
            if not leg:
                continue
            k = float(leg["strike"])

            def on_dial_change(e, idx=i, st=step):
                new_k = round_to_step(safe_float(e.control.value, state["legs"][idx]["strike"]), st)
                state["legs"][idx]["strike"] = round(new_k, 4)
                generate_chart()

            def on_dial_end(e):
                build_legs_ui()
                page.update()

            dial = ft.Slider(
                min=lo, max=hi,
                value=round_to_step(max(lo, min(k, hi)), step),
                divisions=divisions,
                label="{value:.1f}",
                on_change=on_dial_change,
                on_change_end=on_dial_end,
                expand=True,
            )
            label_text = f"Leg {i}: {leg['type'][0].upper()}/{leg['side'][0].upper()}  K={fmt_k(k, state['ticker_input'])}"
            strike_dials_container.controls.append(
                ft.Column([ft.Text(label_text, size=12), dial])
            )

    def update_leg(leg_idx, field, value):
        if leg_idx not in state["legs"]:
            return
        state["legs"][leg_idx][field] = value

        # 種類・売買が変わったらダイヤルのラベルを作り直し、グラフも更新する
        if field in ("type", "side"):
            build_strike_dials_ui()
            generate_chart()      # 内部で page.update() まで行う

    def generate_chart(e=None):
        update_state_from_fields()

        # ポジション構築
        positions = []
        for i in range(1, state["n_legs"] + 1):
            if i in state["legs"]:
                leg = state["legs"][i]
                positions.append(
                    OptionPosition(
                        option_type=leg["type"],
                        side=leg["side"],
                        strike=leg["strike"],
                        premium=leg["premium"],
                        quantity=leg["quantity"],
                        contract_size=state["contract_size"],
                        r=state["r"],
                        sigma=leg["sigma"],
                        T=leg["T_days"] / 365.0
                    )
                )

        if not positions:
            results_text.value = "少なくとも 1 つポジションを入力してください"
            chart_container.controls.clear()
            page.update()
            return

        # 計算
        spot_min = state["spot_center"] - state["spot_range"]
        spot_max = state["spot_center"] + state["spot_range"]
        
        # なめらかな曲線を生成（最低 500 点）
        n_points = max(int((spot_max - spot_min) / 50.0) + 1, 500)
        spots = np.linspace(spot_min, spot_max, n_points)

        total_now = total_pl_now_bs(positions, spots)
        total_maturity = total_payoff_maturity(positions, spots)

        # --- What-if 用計算 ---
        what_if_curve = None
        what_if_pl_value = None
        what_if_max = None
        what_if_min = None

        if state.get("what_if_enabled", False):
            T_w = max(state["what_if_days"], 0) / 365.0
            sigma_w = max(state["what_if_iv"], 0.01)

            positions_w = []
            for p in positions:
                positions_w.append(
                    OptionPosition(
                        option_type=p.option_type,
                        side=p.side,
                        strike=p.strike,
                        premium=p.premium,
                        quantity=p.quantity,
                        contract_size=p.contract_size,
                        r=p.r,
                        sigma=sigma_w,
                        T=T_w,
                    )
                )
            what_if_curve = total_pl_now_bs(positions_w, spots)

            # What-if 価格スライダー位置での合計損益
            S_w = state["what_if_price"]
            S_w_clamped = max(spot_min, min(S_w, spot_max))
            what_if_pl_value = float(
                total_pl_now_bs(positions_w, np.array([S_w_clamped]))[0]
            )

            # What-if 曲線の最大・最小
            what_if_max = float(np.max(what_if_curve))
            what_if_min = float(np.min(what_if_curve))

            # 予想損益テキスト更新
            if what_if_pl_text is not None:
                # 元のユーザー入力を使用
                ticker_for_currency = state.get("ticker_input", state["ticker"])
                what_if_pl_text.value = format_currency(what_if_pl_value, ticker_for_currency)
        else:
            if what_if_pl_text is not None:
                what_if_pl_text.value = ""

        max_now = float(np.max(total_now))
        min_now = float(np.min(total_now))
        max_mat = float(np.max(total_maturity))
        min_mat = float(np.min(total_maturity))

        # チャート作成（matplotlib）
        system = platform.system()
        if system == 'Darwin':  # macOS
            plt.rcParams['font.family'] = ['Hiragino Sans', 'Hiragino Maru Gothic Pro', 'AppleGothic']
        elif system == 'Windows':
            plt.rcParams['font.family'] = ['MS Gothic', 'Yu Gothic', 'Meiryo']
        else:  # Linux等
            plt.rcParams['font.family'] = ['Noto Sans CJK JP', 'IPAGothic', 'DejaVu Sans']

        plt.rcParams['axes.unicode_minus'] = False  # マイナス記号の文字化け対策

        fig, ax = plt.subplots(figsize=(10, 5.5))
        ax.plot(spots, total_now, label='期中損益 (BS 理論価格)', color='#ff7f0e', linewidth=2.5)
        ax.plot(spots, total_maturity, label='満期損益', color='#1f77b4', linewidth=2, linestyle='--')

        # --- What-if カーブを追加 ---
        if what_if_curve is not None:
            ax.plot(
                spots,
                what_if_curve,
                label='What-if 期中損益',
                color='#2ca02c',
                linewidth=2,
                linestyle='-.'
            )
            # What-if 価格位置に縦線マーカー
            S_w = state["what_if_price"]
            if spot_min <= S_w <= spot_max:
                ax.axvline(x=S_w, color='#2ca02c', linewidth=1.0, linestyle=':', alpha=0.8)

        ax.axhline(y=0, color='gray', linewidth=1)
        # 中心価格の表示（0.5刻みなら小数点表示）
        c = state["spot_center"]
        center_label = f'中心価格 ({fmt_spot(c)})'
        ax.axvline(x=state["spot_center"], color='green', linewidth=1.2, linestyle=':', label=center_label)

        # 各レッグの行使価格（K）にマーカー垂直線を引く
        # strikes = sorted(list({p.strike for p in positions}))
        # for k in strikes:
        #     if spot_min <= k <= spot_max:
        #         ax.axvline(x=k, color='#9467bd', linewidth=0.8, linestyle=':', alpha=0.6)


        # 各レッグの行使価格（K）にタグ＋縦線を引く（theta PROFITS 風）
        # まず y 範囲を取得して、タグを載せる高さを決める
        fig.canvas.draw()
        y_min_ax, y_max_ax = ax.get_ylim()
        tag_y = y_max_ax * 0.9  # グラフ上部 90% あたり

        for pos in positions:
            k = pos.strike
            if not (spot_min <= k <= spot_max):
                continue

            # ラベル文字列 例: 63250C, 62000P（0.5刻みなら 735.5C など）
            label = f"{fmt_k(k, state['ticker_input'])}{'C' if pos.option_type == 'call' else 'P'}"

            facecolor, edgecolor, textcolor = get_tag_style_for_position(pos)

            # タグ本体
            ax.text(
                k, tag_y, label,
                ha="center", va="center",
                color=textcolor,
                fontsize=8,
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor=facecolor,
                    edgecolor=edgecolor,
                    linewidth=1.0,
                )
            )

            # 縦のガイドライン
            ax.axvline(x=k, color=edgecolor, linewidth=0.8, linestyle=":", alpha=0.6)

        # タグが切れないように少しだけ上に余裕を足す
        ax.set_ylim(y_min_ax, y_max_ax * 1.15)


        # 最大利益・損失ライン
        ax.axhline(y=max_now, color='#ff7f0e', linewidth=0.8, linestyle=':', alpha=0.6)
        ax.axhline(y=min_now, color='#ff7f0e', linewidth=0.8, linestyle=':', alpha=0.6)
        ax.axhline(y=max_mat, color='#1f77b4', linewidth=0.8, linestyle=':', alpha=0.6)
        ax.axhline(y=min_mat, color='#1f77b4', linewidth=0.8, linestyle=':', alpha=0.6)

        # 横軸・縦軸のフォーマット改善（3桁カンマ区切り、詳細すぎる目盛りの整理）
        ax.set_xlim(spot_min, spot_max)
        
        # X軸フォーマット：刻み幅が1未満なら小数点表示
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(
            lambda x, p: fmt_spot(x)))
        
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, p: f"{y:,.0f}"))

        # 目盛りの数を多すぎず（6〜8本程度に）キリの良い数値で自動調整
        # 0.5刻みにも対応
        ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=8, steps=[1, 2, 2.5, 5, 10]))
        ax.tick_params(axis='x', rotation=15, labelsize=9)
        ax.tick_params(axis='y', labelsize=9)

        # 通貨記号を取得（元のユーザー入力を使用）
        ticker_for_currency = state.get("ticker_input", state["ticker"])
        currency = get_currency_symbol(ticker_for_currency)
        
        ax.set_xlabel('原資産価格 S', fontsize=11)
        ax.set_ylabel(f'損益 ({currency})', fontsize=11)
        # タイトルには元のユーザー入力を表示
        ticker_for_title = state.get("ticker_input", state["ticker"])
        ax.set_title(f'オプション戦略 損益グラフ（原資産: {ticker_for_title}）', fontsize=13, weight='bold')
        ax.legend(loc='best', framealpha=0.85)
        ax.grid(True, alpha=0.3, linestyle='--')
        fig.tight_layout()

        # メモリ上にPNG画像として出力し、ft.Image に渡す
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        plt.close(fig)
        img_bytes = buf.getvalue()

        chart_container.controls.clear()
        chart_container.controls.append(
            ft.Image(src=img_bytes, width=800, height=440)
        )

        # 通貨フォーマット用のティッカー取得（元のユーザー入力を使用）
        current_ticker = state.get("ticker_input", state["ticker"])
        
        base_text = (
            f"最大利益（期中）: {format_currency(max_now, current_ticker)}\n"
            f"最大損失（期中）: {format_currency(min_now, current_ticker)}\n"
            f"最大利益（満期）: {format_currency(max_mat, current_ticker)}\n"
            f"最大損失（満期）: {format_currency(min_mat, current_ticker)}"
        )

        if what_if_curve is not None and what_if_max is not None and what_if_min is not None:
            base_text += (
                f"\n\n"
                f"【What-if シナリオ】\n"
                f"最大利益（期中）: {format_currency(what_if_max, current_ticker)}\n"
                f"最大損失（期中）: {format_currency(what_if_min, current_ticker)}"
            )

        results_text.value = base_text

        update_range_display()

        page.update()

    def on_spot_range_field_change(e):
        sanitize_numeric_input(e)
        try:
            val = float(spot_range_field.value)
            if val > 0:
                state["spot_range"] = val
                update_range_slider_bounds(val)
                update_range_display()
                build_strike_dials_ui()
        except (ValueError, TypeError):
            pass


    def on_ticker_commit(e):
        # ティッカーが変わったときだけ再取得(blur のたびにレッグが初期化されるのを防ぐ)
        if (ticker_field.value or "") != state.get("prev_ticker"):
            update_ticker_info(force_k_update=True)

    ticker_field.on_submit = on_ticker_commit
    ticker_field.on_blur = on_ticker_commit

    def on_spot_center_field_change(e):
        sanitize_numeric_input(e)
        val = safe_float(spot_center_field.value, 0.0)
        if val > 0 and abs(val - state["spot_center"]) > 0.01:
            auto_fetch_check.value = False                 # 手入力を優先
            update_ticker_info(force_k_update=True, manual_center=val)
            # ※ state["spot_center"] を先に書き換えない(old_center が潰れるため)

    def on_step_change(e):
        sanitize_numeric_input(e)
        v = safe_float(spot_step_field.value, 0.0)
        if v > 0:
            state["spot_step"] = v
            build_strike_dials_ui()
            page.update()

    spot_step_field.on_change = on_step_change


    # イベントハンドラー設定
    ticker_field.on_submit = lambda e: update_ticker_info(force_k_update=True)
    ticker_field.on_blur = lambda e: update_ticker_info(force_k_update=True)
    auto_fetch_check.on_change = lambda e: update_ticker_info(force_k_update=True)
    legs_slider.on_change = on_legs_change
    spot_range_field.on_change = on_spot_range_field_change
    spot_center_field.on_submit = on_spot_center_field_change
    for ev in ["on_select", "on_change", "on_selected"]:
        if hasattr(preset_dropdown, ev):
            setattr(preset_dropdown, ev, on_preset_change)

    # 初期レッグUI構築
    build_legs_ui()

    # レイアウト
    page.add(
        ft.Row([
            # サイドバー（左側）
            ft.Container(
                content=ft.Column([
                    ft.Text("原資産レンジ設定", size=20, weight=ft.FontWeight.BOLD),
                    ft.Row([
                        ticker_field,
                        ft.IconButton(
                            icon=ft.Icons.SEARCH,
                            tooltip="価格を取得して行使価格Kを更新",
                            on_click=lambda e: update_ticker_info(force_k_update=True),
                        )
                    ]),
                    auto_fetch_check,
                    spot_center_field,
                    yfinance_info,
                    spot_range_field,
                    spot_step_field,
                    ft.Divider(),
                    ft.Text("共通パラメータ", size=20, weight=ft.FontWeight.BOLD),
                    contract_size_field,
                    r_field,
                    ft.Divider(),
                    ft.Text("戦略プリセット", size=20, weight=ft.FontWeight.BOLD),
                    preset_dropdown,
                    ft.Divider(),
                    ft.Text("レッグ数", size=20, weight=ft.FontWeight.BOLD),
                    legs_slider,
                ], scroll="adaptive"),
                width=400,
                padding=20
            ),
            
            # メインエリア（右側）
            ft.Container(
                content=ft.Column([
                    ft.Text("ポジション入力", size=24, weight=ft.FontWeight.BOLD),
                    legs_container,
                    ft.FilledButton(
                        "グラフを更新",
                        on_click=generate_chart,
                        icon=ft.Icons.REFRESH
                    ),
                    ft.Divider(),
                    what_if_card,
                    range_control_card,
                    strike_dials_container,
                    chart_container,
                    results_text,
                ], scroll="adaptive"),
                expand=True,
                padding=20
            )
        ], expand=True)
    )
    
    # 初期情報更新 & 初回グラフ描画
    update_ticker_info()


if __name__ == "__main__":
    import flet.app as fapp
    fapp.run(main)

