# オプション戦略 損益グラフツール

このツールは、複数レッグのオプションポジションを組み合わせたときの

**満期時損益曲線** と **ブラック＝ショールズ（BS）理論価格ベースの期中損益曲線** を可視化するアプリです。

日経 225 や日本株（`XXXX.T`）などのティッカーを指定し、代表的なオプション戦略プリセットを選ぶことで、損益イメージを直感的に確認できます。

**2 つのバージョンを提供しています：**

- **Streamlit 版（`app.py`）**: Web ブラウザベースの簡易版
- **Flet 版（`app-flet.py`）**: デスクトップ/Web/モバイル対応のクロスプラットフォーム版

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

**Streamlit 版の場合：**

```
pip install streamlit plotly yfinance scipy numpy
```

**Flet 版の場合：**

```
pip install flet matplotlib yfinance scipy numpy
```

## アプリの起動方法

### Streamlit 版の起動

ローカル開発環境での起動は以下の通りです。

```bash
cd option-payoff
streamlit run app.py
```

実行後、ターミナルに表示される URL（通常は `⁠http://localhost:8501⁠）`をブラウザで開くとアプリが表示されます。

### Flet 版の起動

デスクトップアプリケーションとして起動します。

```bash
cd option-payoff
python app-flet.py
```

実行後、自動的にデスクトップアプリケーションウィンドウが開きます。

**Flet 版の特徴：**

- ネイティブデスクトップアプリとして動作
- Web 版としても実行可能（コード内で設定変更）
- iOS/Android アプリにもビルド可能
- インターネット接続不要（yfinance 使用時を除く）

### Flet 版の Web モードでの起動

デスクトップアプリではなく、Web ブラウザで開きたい場合は、`app-flet.py` の最後の部分を以下のように変更してください。

**変更前：**

```python
if __name__ == "__main__":
    ft.app(target=main)
```

**変更後：**

```python
if __name__ == "__main__":
    ft.app(target=main, view=ft.WEB_BROWSER)
```

これにより、実行時にデフォルトの Web ブラウザで自動的に開きます。

### Flet 版のモバイルアプリ化（iOS/Android）

Flet は Flutter をベースにしているため、iOS および Android のネイティブアプリとしてビルドできます。

#### 前提条件

**iOS の場合：**

- macOS が必要
- Xcode のインストール（App Store から入手）
- Apple Developer アカウント（実機テストやリリースの場合）

**Android の場合：**

- Android Studio のインストール
- Android SDK と NDK の設定

#### モバイルアプリのビルド手順

1. **Flet CLI のインストール**

```bash
pip install flet
```

2. **プロジェクトの準備**

モバイルビルド用のメタデータを追加するため、プロジェクトディレクトリに `pyproject.toml` を作成します。

```toml
[project]
name = "option-payoff"
version = "1.0.0"
description = "オプション戦略損益グラフツール"

[tool.flet]
app_name = "OptionPayoff"
org = "com.yourcompany"
```

3. **Android APK のビルド**

```bash
flet build apk --project app-flet.py
```

ビルドが完了すると、`build/apk` ディレクトリに `.apk` ファイルが生成されます。

4. **Android AAB のビルド（Google Play 配布用）**

```bash
flet build aab --project app-flet.py
```

5. **iOS IPA のビルド（macOS のみ）**

```bash
flet build ipa --project app-flet.py
```

6. **生成されたアプリのインストール**

- **Android**: 生成された `.apk` ファイルを Android デバイスに転送してインストール
- **iOS**: Xcode を使って実機にインストール、または TestFlight/App Store 経由で配布

#### モバイルビルド時の注意事項

**依存関係の調整**

モバイル環境では一部のライブラリが動作しない場合があります。特に以下に注意が必要です：

- **yfinance**: インターネット接続が必要です。モバイル環境でも動作しますが、ネットワークパーミッションが必要です
- **matplotlib**: モバイルでも動作しますが、画像の一時保存パスに注意が必要です
- **numpy/scipy**: ARM アーキテクチャでのコンパイルに時間がかかる場合があります

**UI の調整**

現在の `app-flet.py` はデスクトップ向けのレイアウトになっています。モバイル向けに最適化するには：

1. 画面サイズの検出と応答的なレイアウト
2. タッチ操作に適したボタンサイズ
3. スクロール可能なコンテナの活用

モバイル対応版を作成する場合は、`page.width` や `page.platform` を使って動的にレイアウトを調整することをお勧めします。

**パーミッション設定**

インターネットアクセス（yfinance 用）やファイルアクセス（チャート画像保存用）が必要な場合は、Android の `AndroidManifest.xml` や iOS の `Info.plist` にパーミッションを追加する必要があります。

詳細は [Flet 公式ドキュメント](https://flet.dev/docs/guides/python/mobile) を参照してください。

### Mac/Linux でのデスクトップアプリ化（Streamlit 版）

[streamlit-desktop-app](https://zenn.dev/ohtaman/articles/streamlit-desktop-app)を利用して
作成します。
詳細についてはリンク先をご確認ください。

### Mac でのワンクリック起動（Streamlit 版）

ここでは「ネイティブアプリ」にするのではなく、Streamlit アプリをワンクリックで起動できる 簡易デスクトップアプリ風 の運用方法を説明します。

**注意：Flet 版（`app-flet.py`）を使用する場合は、この手順は不要です。`python app-flet.py` を実行するだけでネイティブアプリとして動作します。**

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

### Windows でのワンクリック起動（Streamlit 版）

こちらも「ワンクリックで起動する」ためのバッチファイルを用いた簡易的な方法です。

**注意：Flet 版（`app-flet.py`）を使用する場合は、この手順は不要です。`python app-flet.py` を実行するだけでネイティブアプリとして動作します。**

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
