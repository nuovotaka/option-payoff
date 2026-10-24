# app-flet.py
# オプション戦略 損益グラフ（BS 付き）＋ yfinance 原資産
# Flet版

import math
import numpy as np
import flet as ft
from dataclasses import dataclass
from typing import List, Literal
from scipy.stats import norm
import yfinance as yf
from enum import Enum
import os
import tempfile

OptionType = Literal["call", "put"]
SideType = Literal["long", "short"]


def get_last_close(ticker: str) -> float | None:
    try:
        data = yf.Ticker(ticker).history(period="5d")
        if data.empty:
            return None
        return float(data["Close"].iloc[-1])
    except Exception:
        return None


def guess_contract_size_from_ticker(ticker: str) -> int:
    t = ticker.upper()
    if "N225" in t:
        if "MICRO" in t:
            return 10
        if "MINI" in t:
            return 100
        return 1000
    return 100


def guess_strike_step_from_ticker(ticker: str) -> float:
    t = ticker.upper()
    if "TOPIX" in t or "TPX" in t:
        return 25.0
    if "N225" in t:
        return 125.0
    return 100.0


def round_to_step(x: float, step: float) -> float:
    if step <= 0:
        return x
    return round(x / step) * step


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


def build_positions_from_preset(
    preset: StrategyPreset,
    spot_center: float,
    contract_size: int,
    r: float,
    ticker: str,
) -> List[OptionPosition]:
    positions: List[OptionPosition] = []

    sigma = 0.2
    T_short = 30 / 365
    T_long = 90 / 365

    step = guess_strike_step_from_ticker(ticker)
    k_atm = round_to_step(spot_center, step)
    k_low = round_to_step(spot_center * 0.95, step)
    k_high = round_to_step(spot_center * 1.05, step)
    k_very_low = round_to_step(spot_center * 0.9, step)
    k_very_high = round_to_step(spot_center * 1.1, step)

    if preset == StrategyPreset.LONG_CALL:
        positions.append(
            OptionPosition("call", "long", k_atm, 300.0, 1, contract_size, r, sigma, T_short)
        )
    elif preset == StrategyPreset.SHORT_CALL:
        positions.append(
            OptionPosition("call", "short", k_atm, 300.0, 1, contract_size, r, sigma, T_short)
        )
    elif preset == StrategyPreset.LONG_PUT:
        positions.append(
            OptionPosition("put", "long", k_atm, 300.0, 1, contract_size, r, sigma, T_short)
        )
    elif preset == StrategyPreset.SHORT_PUT:
        positions.append(
            OptionPosition("put", "short", k_atm, 300.0, 1, contract_size, r, sigma, T_short)
        )
    elif preset == StrategyPreset.BULL_CALL_SPREAD:
        positions.append(
            OptionPosition("call", "long", k_low, 400.0, 1, contract_size, r, sigma, T_short)
        )
        positions.append(
            OptionPosition("call", "short", k_high, 200.0, 1, contract_size, r, sigma, T_short)
        )
    elif preset == StrategyPreset.BEAR_PUT_SPREAD:
        positions.append(
            OptionPosition("put", "long", k_high, 400.0, 1, contract_size, r, sigma, T_short)
        )
        positions.append(
            OptionPosition("put", "short", k_low, 200.0, 1, contract_size, r, sigma, T_short)
        )
    elif preset == StrategyPreset.BULL_PUT_SPREAD:
        positions.append(
            OptionPosition("put", "short", k_high, 400.0, 1, contract_size, r, sigma, T_short)
        )
        positions.append(
            OptionPosition("put", "long", k_low, 200.0, 1, contract_size, r, sigma, T_short)
        )
    elif preset == StrategyPreset.BEAR_CALL_SPREAD:
        positions.append(
            OptionPosition("call", "short", k_low, 400.0, 1, contract_size, r, sigma, T_short)
        )
        positions.append(
            OptionPosition("call", "long", k_high, 200.0, 1, contract_size, r, sigma, T_short)
        )
    elif preset == StrategyPreset.LONG_STRADDLE:
        positions.append(
            OptionPosition("call", "long", k_atm, 300.0, 1, contract_size, r, sigma, T_short)
        )
        positions.append(
            OptionPosition("put", "long", k_atm, 300.0, 1, contract_size, r, sigma, T_short)
        )
    elif preset == StrategyPreset.SHORT_STRADDLE:
        positions.append(
            OptionPosition("call", "short", k_atm, 300.0, 1, contract_size, r, sigma, T_short)
        )
        positions.append(
            OptionPosition("put", "short", k_atm, 300.0, 1, contract_size, r, sigma, T_short)
        )
    elif preset == StrategyPreset.LONG_STRANGLE:
        positions.append(
            OptionPosition("call", "long", k_high, 180.0, 1, contract_size, r, sigma, T_short)
        )
        positions.append(
            OptionPosition("put", "long", k_low, 180.0, 1, contract_size, r, sigma, T_short)
        )
    elif preset == StrategyPreset.SHORT_STRANGLE:
        positions.append(
            OptionPosition("call", "short", k_high, 250.0, 1, contract_size, r, sigma, T_short)
        )
        positions.append(
            OptionPosition("put", "short", k_low, 250.0, 1, contract_size, r, sigma, T_short)
        )
    elif preset == StrategyPreset.IRON_CONDOR:
        positions.append(
            OptionPosition("put", "short", k_low, 200.0, 1, contract_size, r, sigma, T_short)
        )
        positions.append(
            OptionPosition("put", "long", k_very_low, 100.0, 1, contract_size, r, sigma, T_short)
        )
        positions.append(
            OptionPosition("call", "short", k_high, 200.0, 1, contract_size, r, sigma, T_short)
        )
        positions.append(
            OptionPosition("call", "long", k_very_high, 100.0, 1, contract_size, r, sigma, T_short)
        )
    elif preset == StrategyPreset.IRON_BUTTERFLY:
        positions.append(
            OptionPosition("call", "short", k_atm, 300.0, 1, contract_size, r, sigma, T_short)
        )
        positions.append(
            OptionPosition("put", "short", k_atm, 300.0, 1, contract_size, r, sigma, T_short)
        )
        positions.append(
            OptionPosition("call", "long", k_high, 150.0, 1, contract_size, r, sigma, T_short)
        )
        positions.append(
            OptionPosition("put", "long", k_low, 150.0, 1, contract_size, r, sigma, T_short)
        )
    elif preset == StrategyPreset.COVERED_CALL:
        positions.append(
            OptionPosition("call", "short", k_high, 200.0, 1, contract_size, r, sigma, T_short)
        )
    elif preset == StrategyPreset.PROTECTIVE_PUT:
        positions.append(
            OptionPosition("put", "long", k_low, 200.0, 1, contract_size, r, sigma, T_short)
        )
    elif preset == StrategyPreset.COLLAR:
        positions.append(
            OptionPosition("put", "long", k_low, 150.0, 1, contract_size, r, sigma, T_short)
        )
        positions.append(
            OptionPosition("call", "short", k_high, 150.0, 1, contract_size, r, sigma, T_short)
        )
    elif preset == StrategyPreset.CASH_SECURED_PUT:
        positions.append(
            OptionPosition("put", "short", k_low, 250.0, 1, contract_size, r, sigma, T_short)
        )
    elif preset == StrategyPreset.PUT_RATIO_SPREAD:
        positions.append(
            OptionPosition("put", "long", k_high, 350.0, 1, contract_size, r, sigma, T_short)
        )
        positions.append(
            OptionPosition("put", "short", k_low, 220.0, 2, contract_size, r, sigma, T_short)
        )
    elif preset == StrategyPreset.CALENDAR_SPREAD:
        positions.append(
            OptionPosition("call", "short", k_atm, 200.0, 1, contract_size, r, sigma, T_short)
        )
        positions.append(
            OptionPosition("call", "long", k_atm, 250.0, 1, contract_size, r, sigma, T_long)
        )

    return positions


def main(page: ft.Page):
    page.title = "オプション戦略 損益グラフ（BS 付き）"
    page.scroll = "adaptive"
    page.padding = 20

    # 状態管理
    state = {
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
    }

    # UI要素の参照
    ticker_field = ft.TextField(
        label="原資産ティッカー（例: ^N225, 7203.T）",
        value=state["ticker"],
        width=300
    )

    auto_fetch_check = ft.Checkbox(
        label="ティッカーから中心価格を自動取得する",
        value=True
    )

    spot_center_field = ft.TextField(
        label="中心となる原資産価格",
        value=str(state["spot_center"]),
        keyboard_type=ft.KeyboardType.NUMBER,
        width=200
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
        width=200
    )

    contract_size_field = ft.TextField(
        label="取引単位",
        value=str(state["contract_size"]),
        keyboard_type=ft.KeyboardType.NUMBER,
        width=200
    )

    r_field = ft.TextField(
        label="無リスク金利 r（年率）",
        value=str(state["r"]),
        keyboard_type=ft.KeyboardType.NUMBER,
        width=200
    )

    preset_dropdown = ft.Dropdown(
      label="戦略プリセット",
      options=[
          ft.DropdownOption(key=p.name, text=p.value)   # key: Enum 名, text: 日本語ラベル
          for p in StrategyPreset
      ],
      value=StrategyPreset.NONE.name,
      width=400
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
    chart_container = ft.Column()
    results_text = ft.Text("")



    def on_preset_change(e: ft.Event[ft.Dropdown]):
      print("on_preset_change fired")
      print("Dropdown raw value:", e.control.value)

      preset_name = e.control.value  # = preset_dropdown.value と同じ
      state["preset"] = StrategyPreset[preset_name]

      if state["preset"] != StrategyPreset.NONE:
          update_state_from_fields()
          preset_positions = build_positions_from_preset(
              preset=state["preset"],
              spot_center=state["spot_center"],
              contract_size=state["contract_size"],
              r=state["r"],
              ticker=state["ticker"],
          )
          state["n_legs"] = len(preset_positions)
          legs_slider.value = state["n_legs"]

          state["legs"] = {}
          for i, pos in enumerate(preset_positions, start=1):
              state["legs"][i] = {
                  "type": pos.option_type,
                  "side": pos.side,
                  "strike": pos.strike,
                  "premium": pos.premium,
                  "quantity": pos.quantity,
                  "sigma": pos.sigma,
                  "T_days": int(pos.T * 365),
              }

          build_legs_ui()

      page.update()




    def update_ticker_info(e=None):
        state["ticker"] = ticker_field.value
        
        if auto_fetch_check.value:
            last_close = get_last_close(state["ticker"])
            if last_close is not None:
                state["spot_center"] = float(round(last_close, -1))
                spot_center_field.value = str(state["spot_center"])
                yfinance_info.value = f"yfinance 終値: 約 {last_close:.2f}"
            else:
                yfinance_info.value = "価格を取得できませんでした"
        
        state["contract_size"] = guess_contract_size_from_ticker(state["ticker"])
        contract_size_field.value = str(state["contract_size"])
        
        page.update()

    def update_state_from_fields():
        state["ticker"] = ticker_field.value
        state["spot_center"] = float(spot_center_field.value)
        state["spot_range"] = float(spot_range_field.value)
        state["spot_step"] = float(spot_step_field.value)
        state["contract_size"] = int(contract_size_field.value)
        state["r"] = float(r_field.value)

    def on_preset_change(e):
        preset_name = preset_dropdown.value
        state["preset"] = StrategyPreset[preset_name]
        
        print(f"プリセット選択: {preset_name}")  # デバッグ用
        
        if state["preset"] != StrategyPreset.NONE:
            update_state_from_fields()
            preset_positions = build_positions_from_preset(
                preset=state["preset"],
                spot_center=state["spot_center"],
                contract_size=state["contract_size"],
                r=state["r"],
                ticker=state["ticker"],
            )
            state["n_legs"] = len(preset_positions)
            legs_slider.value = state["n_legs"]
            
            print(f"レッグ数: {state['n_legs']}")  # デバッグ用
            
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
                    "T_days": int(pos.T * 365)
                }
                print(f"レッグ {i}: {state['legs'][i]}")  # デバッグ用
            
            # UIを再構築
            build_legs_ui()
        
        page.update()

    def on_legs_change(e):
        state["n_legs"] = int(legs_slider.value)
        build_legs_ui()
        page.update()

    def build_legs_ui():
        print("build_legs_ui called; n_legs =", state["n_legs"])
        print("state['legs'] before:", state["legs"])

        legs_container.controls.clear()
        
        for i in range(1, state["n_legs"] + 1):
            if i not in state["legs"]:
                strike_step = guess_strike_step_from_ticker(state["ticker"])
                state["legs"][i] = {
                    "type": "call",
                    "side": "long",
                    "strike": round_to_step(state["spot_center"], strike_step),
                    "premium": 300.0,
                    "quantity": 1,
                    "sigma": 0.2,
                    "T_days": 30
                }
            
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
            type_dropdown.on_change = lambda e, idx=i: update_leg(idx, "type", e.control.value)
            
            side_dropdown = ft.Dropdown(
                label="売買",
                options=[
                    ft.dropdown.Option("long", "Long"),
                    ft.dropdown.Option("short", "Short"),
                ],
                value=leg_data["side"],
                width=150,
            )
            side_dropdown.on_change = lambda e, idx=i: update_leg(idx, "side", e.control.value)
            
            strike_field = ft.TextField(
                label="行使価格 K",
                value=str(leg_data["strike"]),
                keyboard_type=ft.KeyboardType.NUMBER,
                width=150,
                on_change=lambda e, idx=i: update_leg(idx, "strike", float(e.control.value) if e.control.value else 0)
            )
            
            premium_field = ft.TextField(
                label="プレミアム",
                value=str(leg_data["premium"]),
                keyboard_type=ft.KeyboardType.NUMBER,
                width=150,
                on_change=lambda e, idx=i: update_leg(idx, "premium", float(e.control.value) if e.control.value else 0)
            )
            
            quantity_field = ft.TextField(
                label="枚数",
                value=str(leg_data["quantity"]),
                keyboard_type=ft.KeyboardType.NUMBER,
                width=100,
                on_change=lambda e, idx=i: update_leg(idx, "quantity", int(float(e.control.value)) if e.control.value else 1)
            )
            
            sigma_field = ft.TextField(
                label="ボラティリティ σ",
                value=str(leg_data["sigma"]),
                keyboard_type=ft.KeyboardType.NUMBER,
                width=150,
                on_change=lambda e, idx=i: update_leg(idx, "sigma", float(e.control.value) if e.control.value else 0)
            )
            
            t_days_field = ft.TextField(
                label="残存日数（日）",
                value=str(leg_data["T_days"]),
                keyboard_type=ft.KeyboardType.NUMBER,
                width=150,
                on_change=lambda e, idx=i: update_leg(idx, "T_days", int(float(e.control.value)) if e.control.value else 0)
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

    def update_leg(leg_idx, field, value):
        if leg_idx in state["legs"]:
            state["legs"][leg_idx][field] = value

    def generate_chart(e):
        update_state_from_fields()
        # ... spots, total_now, total_maturity などの計算までは今のままでOK ...

        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import platform

        system = platform.system()
        if system == 'Darwin':
            plt.rcParams['font.family'] = ['Hiragino Sans', 'Hiragino Maru Gothic Pro', 'AppleGothic']
        elif system == 'Windows':
            plt.rcParams['font.family'] = ['MS Gothic', 'Yu Gothic', 'Meiryo']
        else:
            plt.rcParams['font.family'] = ['Noto Sans CJK JP', 'IPAGothic', 'DejaVu Sans']

        plt.rcParams['axes.unicode_minus'] = False

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(spots, total_now, label='Total P/L Now (BS)', color='orange', linewidth=2)
        ax.plot(spots, total_maturity, label='Total P/L at Maturity', color='blue', linewidth=2, linestyle='--')
        ax.axhline(y=0, color='gray', linewidth=1)
        ax.axvline(x=state["spot_center"], color='green', linewidth=1, linestyle=':')
        ax.set_xlabel('原資産価格 S')
        ax.set_ylabel('損益')
        ax.set_title(f'オプション戦略 損益グラフ（原資産: {state["ticker"]}）')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # ここから「一時ファイル方式」
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
        fig.savefig(temp_file.name, dpi=100, bbox_inches='tight')
        plt.close(fig)

        print("graph saved to:", temp_file.name)  # デバッグログ

        chart_container.controls.clear()
        chart_container.controls.append(
            ft.Image(src=temp_file.name, width=800, height=480)
        )

        results_text.value = (
            f"最大利益（期中）: {max_now:,.0f}\n"
            f"最大損失（期中）: {min_now:,.0f}\n"
            f"最大利益（満期）: {max_mat:,.0f}\n"
            f"最大損失（満期）: {min_mat:,.0f}"
        )

        page.update()

        # ★絶対に削除しない★
        # os.unlink(temp_file.name)
        
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
            page.update()
            return
        
        # 計算
        spot_min = state["spot_center"] - state["spot_range"]
        spot_max = state["spot_center"] + state["spot_range"]
        spots = np.arange(spot_min, spot_max + state["spot_step"], state["spot_step"])
        
        total_now = total_pl_now_bs(positions, spots)
        total_maturity = total_payoff_maturity(positions, spots)
        
        max_now = float(np.max(total_now))
        min_now = float(np.min(total_now))
        max_mat = float(np.max(total_maturity))
        min_mat = float(np.min(total_maturity))
        
        # チャート作成（Fletではmatplotlibを使用）
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import platform
        
        # 日本語フォント設定（OS別）
        system = platform.system()
        if system == 'Darwin':  # macOS
            plt.rcParams['font.family'] = ['Hiragino Sans', 'Hiragino Maru Gothic Pro', 'AppleGothic']
        elif system == 'Windows':
            plt.rcParams['font.family'] = ['MS Gothic', 'Yu Gothic', 'Meiryo']
        else:  # Linux等
            plt.rcParams['font.family'] = ['Noto Sans CJK JP', 'IPAGothic', 'DejaVu Sans']
        
        plt.rcParams['axes.unicode_minus'] = False  # マイナス記号の文字化け対策
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(spots, total_now, label='Total P/L Now (BS)', color='orange', linewidth=2)
        ax.plot(spots, total_maturity, label='Total P/L at Maturity', color='blue', linewidth=2, linestyle='--')
        ax.axhline(y=0, color='gray', linewidth=1)
        ax.axvline(x=state["spot_center"], color='green', linewidth=1, linestyle=':')
        ax.set_xlabel('原資産価格 S')
        ax.set_ylabel('損益')
        ax.set_title(f'オプション戦略 損益グラフ（原資産: {state["ticker"]}）')
        ax.legend()
        ax.grid(True, alpha=0.3)
        

        # 画像としてメモリに保存
        import io
        import base64
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        def generate_chart(e):
            ...
            # fig を作り終わったあと:

            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
            plt.close(fig)
            buf.seek(0)

            img_base64 = base64.b64encode(buf.read()).decode("utf-8")
            data_url = f"data:image/png;base64,{img_base64}"

            chart_container.controls.clear()
            chart_container.controls.append(
                ft.Image(
                    src=data_url,      # ← src_base64 ではなく src に data URL を渡す
                    width=800,
                    height=480,
                )
            )

            page.update()


        
        results_text.value = (
            f"最大利益（期中）: {max_now:,.0f}\n"
            f"最大損失（期中）: {min_now:,.0f}\n"
            f"最大利益（満期）: {max_mat:,.0f}\n"
            f"最大損失（満期）: {min_mat:,.0f}"
        )
        
        page.update()
        
        # 一時ファイルを削除
        # os.unlink(temp_file.name)

    # イベントハンドラー設定
    ticker_field.on_change = update_ticker_info
    preset_dropdown.on_select = on_preset_change
    legs_slider.on_change = on_legs_change

    # 初期レッグUI構築
    build_legs_ui()

    # レイアウト
    page.add(
        ft.Row([
            # サイドバー（左側）
            ft.Container(
                content=ft.Column([
                    ft.Text("原資産レンジ設定", size=20, weight=ft.FontWeight.BOLD),
                    ticker_field,
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
                    chart_container,
                    results_text,
                ], scroll="adaptive"),
                expand=True,
                padding=20
            )
        ], expand=True)
    )
    
    # 初期情報更新
    update_ticker_info()


if __name__ == "__main__":
    import flet.app as fapp
    fapp.run(main)

