# オプション戦略 損益グラフツール

このツールは、複数レッグのオプションポジションを組み合わせたときの

**満期時損益曲線** と **ブラック＝ショールズ（BS）理論価格ベースの期中損益曲線** を可視化する Streamlit アプリです。

日経225や日本株（`XXXX.T`）などのティッカーを指定し、代表的なオプション戦略プリセットを選ぶことで、損益イメージを直感的に確認できます。

---

## 必要なもの（インストール）

Python 3.15 以上を推奨します。

### リポジトリの取得

```bash

git clone <このリポジトリのURL>

cd option-payoff
```

依存パッケージのインストール（pip）

`⁠requirements.txt`⁠ がある場合は次で一括インストールできます。

```
pip install -r requirements.txt
```

`requirements.txt⁠` がない場合は、以下を個別にインストールしてください。

```
pip install streamlit plotly yfinance scipy numpy
```

アプリの起動方法（基本）

ローカル開発環境での起動は以下の通りです。

```
cd option-payoff
streamlit run app.py

```

実行後、ターミナルに表示される URL（通常は `⁠http://localhost:8501⁠）`をブラウザで開くとアプリが表示されます。

### Mac でのデスクトップアプリ化

[streamlit-desktop-app](https://zenn.dev/ohtaman/articles/streamlit-desktop-app)を利用して
作成します。
詳細についてはリンク先をご確認ください。

### Mac でのデスクトップアプリ化（簡易）

ここでは「ネイティブアプリ」にするのではなく、Streamlit アプリをワンクリックで起動できる 簡易デスクトップアプリ風 の運用方法を説明します。

仮想環境の作成（任意）

```
python3 -m venv venv

source venv/bin/activate

pip install streamlit plotly yfinance scipy numpy
```

起動用シェルスクリプトの作成

プロジェクト直下に `⁠run_app.sh`⁠ を作成します。

```
#!/bin/bash

# 仮想環境を使う場合は以下のコメントアウトを外す

# source venv/bin/activate

cd "$(dirname "$0")"
```

```
streamlit run app.py
```

スクリプトに実行権限を付与します。

```
chmod +x run_app.sh
```

Automator アプリとしてラップ

1. Automator を起動する

2. 「新規書類」→「アプリケーション」を選択する

3. 左のライブラリから「ユーティリティ」→「シェルスクリプトを実行」を追加する

4. 「シェルスクリプトを実行」のテキスト欄に以下を記述する

```
cd /path/to/option-payoff

./run_app.sh
```

`/path/to/option-payoff⁠` は実際のパスに置き換えてください。

5. 「ファイル」→「保存」で `OptionPayoff.app⁠` などの名前を付けて保存する

以後は、この `.app⁠ `をダブルクリックするだけでターミナルが立ち上がり、ブラウザでアプリが自動起動します。

### Windows でのデスクトップアプリ化（簡易）

こちらも「ワンクリックで起動する」ためのバッチファイルを用いた簡易的な方法です。

仮想環境の作成（任意）

```
python -m venv venv

venv\Scripts\activate

pip install streamlit plotly yfinance scipy numpy
```

起動用バッチファイルの作成

プロジェクト直下に `run_app.bat`⁠ を作成します。

```
@echo off

REM 仮想環境を使う場合は以下のコメントアウトを外す

REM call venv\Scripts\activate

cd /d "%~dp0"

streamlit run app.py

pause
```

この `⁠run_app.bat`⁠ をダブルクリックすると、コマンドプロンプトが開き、Streamlit アプリが起動してブラウザが立ち上がります。
`⁠run_app.bat⁠` のショートカットをデスクトップなどに置いておけば、デスクトップアプリとほぼ同じ感覚で利用できます。

## 株価などの取得方法

このアプリでは、`⁠yfinance`⁠ ライブラリを使って Yahoo! Finance から株価データを取得しています。

### ティッカーの指定方法

サイドバーの「原資産ティッカー」入力欄に、Yahoo! Finance 形式のティッカーを指定します。

- 日経平均: `^N225⁠`

- TOPIX: `^TOPX⁠` など

- 日本株個別銘柄: `7203.T⁠（トヨタ自動車）`、`6758.T⁠（ソニーグループ）`など

yfinance による終値取得ロジック

`app.py`⁠ には以下のような関数があります。

```
import yfinance as yf
```

```
def get_last_close(ticker: str) -> float | None:

    try:

        data = yf.Ticker(ticker).history(period="5d")

        if data.empty:

            return None

        return float(data["Close"].iloc[-1])

    except Exception:

        return None
```

- `⁠yf.Ticker(ticker).history(period="5d")⁠` で直近 5 営業日の株価履歴を取得

- `⁠data["Close"].iloc[-1]⁠` で最新の終値を取得

- データが取れない場合は `None`⁠ を返し、アプリ側で手動入力を促します

取得した終値は、サイドバーの「中心となる原資産価格」の初期値として使用されます。

## グラフ用の価格レンジ生成

サイドバーで指定した中心価格・レンジ幅・刻み幅に基づき、以下のように
損益グラフの横軸となる原資産価格の配列を生成します。

```
spot_min = spot_center - spot_range

spot_max = spot_center + spot_range

spots = np.arange(spot_min, spot_max + spot_step, spot_step)
```

この `spots⁠` に対して各レッグの損益を計算し、合算結果を Plotly で描画します。

## 戦略プリセットとカスタマイズ

サイドバーの「戦略プリセット」では、代表的なオプション戦略を選択できます。
例として以下のようなカテゴリが含まれます。

- 基本 4 ポジション（ロングコール / ショートコール / ロングプット / ショートプット）

- バーティカル・スプレッド（ブル・コール / ベア・プット / ブル・プット / ベア・コール など）

- ボラティリティ戦略（ロング・ストラドル / ショート・ストラドル / ロング・ストラングル / ショート・ストラングル など）

- レンジ相場向け戦略（アイアン・コンドル / アイアン・バタフライ など）

- 現物組み合わせ戦略（カバード・コール / プロテクティブ・プット / カラー / キャッシュ・セキュアード・プットなど）

- 上級者向け戦略（プット・レシオ・スプレッド / カレンダー・スプレッド など）

プリセットを選ぶと、各レッグ（種類・売買・行使価格・プレミアム・枚数・ボラティリティ・残存日数）が自動でセットされます。
その上で、メイン画面の「レッグ n の設定」から個別に微調整することも可能です。

## 注意事項・免責

- 本ツールは学習・研究・シミュレーション用途を想定しており、投資助言を行うものではありません。

- 実際の取引では、約定価格・スリッページ・手数料・証拠金・金利・ボラティリティ変動などにより、本ツールが示す理論損益と実際の損益は大きく異なる可能性があります。

- 相場の急変、取引停止、配当・権利落ちなど、モデルで考慮していない要因も多数存在します。

本ツールの利用によって生じたいかなる損失・損害についても、開発者は責任を負いません。
ご利用は自己責任でお願いします。
