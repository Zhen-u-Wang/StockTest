# 台股訊號看板（twstock-signal）

以籌碼為主的 LightGBM 橫斷面排名模型，每天盤後對股票池打分、產生「強烈買進／買進／觀望」建議，
五個交易日後自動回填結果、計算命中率與貝氏信賴區間；累積到門檻才把模型標記為「已啟用」。

```
GitHub Actions（每個交易日 17:30 自動跑）
  fetch ──▶ features ──▶ train/predict ──▶ score ──▶ export_json
   │ FinMind                                           │
   ▼                                                   ▼
 data/*.parquet（Actions cache）              docs/data/*.json ──▶ GitHub Pages 顯示 docs/index.html
                                                       │
                                        （選用）同步到 Supabase ──▶ 即時報價 Edge Function
```

## 路線 A：零安裝，只用 GitHub（建議）

你的電腦什麼都不用裝。整個流程在 GitHub Actions 跑，結果 commit 回 repo，GitHub Pages 直接顯示。

1. **建 repo**：到 github.com → New repository → 選 **Public**（Public 的 Actions 分鐘數免費不限）→ Create。
2. **把這個資料夾放進去**，三種方式擇一：
   - 把 repo 名稱給 Claude，由它直接推上去（最快）；
   - 用 [GitHub Desktop](https://desktop.github.com)：File → Add local repository → 選這個資料夾 → Publish；
   - 有裝 git 的話：`git init && git add -A && git commit -m init && git branch -M main && git remote add origin <repo url> && git push -u origin main`。
   
   注意 `.github/` 是隱藏資料夾，用網頁拖拉上傳可能會漏掉，所以不建議用網頁上傳。
3. **開 Pages**：repo Settings → Pages → Source 選 `Deploy from a branch` → Branch `main`、資料夾 `/docs` → Save。
4. **（建議）FinMind token**：到 [finmindtrade.com](https://finmindtrade.com) 免費註冊拿 token → repo Settings → Secrets and variables → Actions → New repository secret → 名稱 `FINMIND_TOKEN`。沒有也能跑，但匿名配額低，第一次抓資料可能要等配額重置、多花一小時。
5. **跑第一次**：Actions 分頁 → `daily-signal` → Run workflow（`retrain` 預設已勾）→ 約 20~40 分鐘。跑完後打開 Pages 的網址（Settings → Pages 會顯示），看板就有資料了。

之後每個交易日 17:30 自動更新、每月 1 日自動重訓；`output/predictions_log.csv` 每天 commit，git 歷史就是你的觀察紀錄。回測報表在 `output/backtest_report.md`。

## 路線 B：本機一鍵執行

想在自己電腦跑（例如不想把東西放 GitHub）：

- **Windows**：雙擊 `setup.bat`
- **macOS / Linux**：`bash setup.sh`

腳本會自動建虛擬環境、裝套件、問你要不要填 FinMind token、抓資料、訓練、產生建議，最後在 http://localhost:8765 開看板。
之後每天再執行一次就會更新。需要先裝 Python 3.11 以上（Windows 安裝時記得勾 "Add python.exe to PATH"）。

手動分步跑也可以：

```bash
pip install -r requirements.txt
export FINMIND_TOKEN=...          # 選填
python -m pipeline.fetch          # 第一次全量，10~20 分鐘
python -m pipeline.train          # walk-forward 回測 → output/backtest_report.md
python -m pipeline.predict        # 今日建議 → output/predictions_log.csv
python -m pipeline.score          # 到期回填（每天跑）
python -m pipeline.export_json    # 更新 docs/data/ 給看板
python -m http.server 8765 --directory docs   # 看板
```

沒有網路或想先確認程式能跑：`python scripts/make_synthetic.py` 會產生合成資料，之後直接跑 train / predict / score。
合成資料裡刻意植入了「投信買超 → 五天後上漲」的關係，所以回測會好看，那不是真實績效。

## 路線 C：加上 Supabase（進階，選用）

A、B 都不需要資料庫。只有在你想要「即時報價」、「帳號／自選股」或「多人共用同一份資料」時才需要：

1. supabase.com 建專案（免費）→ SQL Editor 貼上 `supabase/schema.sql` 執行。
2. Settings → API 取 `Project URL`、`anon public` key、`service_role` key。
3. `docs/index.html` 最上方填入 `SUPABASE_URL` 與 `SUPABASE_ANON_KEY`，頁面就改讀 Supabase。
4. repo Secrets 加 `SUPABASE_URL`、`SUPABASE_SERVICE_KEY`，排程會同步寫入。`service_role` key 只給排程用，不要放進前端。

RLS 已設定成 anon 只能讀。

## 專案結構

```
pipeline/
  config.py        所有參數（股票池、門檻、成本、訓練設定、Supabase）
  fetch.py         FinMind 抓行情、法人、融資券、月營收、加權指數（增量、配額用盡會等待）
  features.py      特徵工程 + 跨股票百分位標準化 + 大盤環境
  labels.py        未來 5 日超額報酬 → 前 30% / 後 30% 標籤
  train.py         Walk-forward 回測 + 最終模型
  predict.py       每日打分、產生建議、寫入日誌（與 Supabase，若有設定）
  score.py         到期回填結果、命中率、貝氏區間、策略開關
  export_json.py   匯出 docs/data/*.json 給看板
  supabase_io.py   Supabase 讀寫（沒設定時自動略過）
docs/index.html                    前端看板（自動偵測：有 Supabase 設定就讀 Supabase，否則讀 docs/data/*.json）
docs/data/                         排程產生的 JSON 結果
.github/workflows/daily.yml        每日排程（GitHub Actions）
setup.bat / setup.sh               本機一鍵執行
supabase/schema.sql                資料表與 RLS（路線 C 用）
supabase/functions/fugle-quote/    即時報價中繼（路線 C 用）
scripts/make_synthetic.py          離線合成資料，測試用
universe.csv                       股票池（可自行增減）
output/predictions_log.csv         預測日誌正本（已存在的列永不覆寫；commit 進 git 即不可竄改）
output/backtest_report.md          回測報表
```

## 即時報價（選用）

需要路線 C（Supabase）。前端不能放富果的 API key，所以用 Supabase Edge Function 當中繼：

```bash
npm i -g supabase
supabase login
supabase link --project-ref <你的 project ref>
supabase secrets set FUGLE_API_KEY=<富果行情 API key>
supabase functions deploy fugle-quote --no-verify-jwt
```

部署後前端會每 15 秒更新「今日建議」的即時價與漲跌（`QUOTE_REFRESH_SEC` 可調）。
富果免費方案有速率限制，10 檔 × 每 15 秒一次在限制內；要更頻繁請升級方案或改用永豐 Shioaji。

## 建議與門檻（config.py）

| 參數 | 預設 | 意思 |
|---|---|---|
| `HORIZON` | 5 | 持有幾個交易日 |
| `TOP_N` | 10 | 每天最多幾檔建議 |
| `SCORE_STRONG` / `SCORE_BUY` | 0.65 / 0.55 | 強烈買進／買進的分數門檻 |
| `MARKET_MA` / `MARKET_MOM` | 60 / 20 | 大盤需站上季線且 20 日報酬為正才出手 |
| `COST` | 0.006 | 來回成本（手續費 0.1425%×2 + 證交稅 0.3%） |
| `SLIPPAGE` | 0.002 | 來回滑價假設，與成本一起從績效扣除 |
| `GATE_MIN_N` | 30 | 回測通過，且至少 30 個不重疊持有期、每期扣成本超額的 95% 信賴下界 > 0 才「啟用」 |
| `FIRST_TEST_YEAR` | 2021 | walk-forward 從哪年開始測 |

回測通過的定義：每期扣成本後超額報酬平均為正，而且拿掉最好的一年仍為正。同一期的股票會一起漲跌，所以門檻用持有期數計算，不用單筆筆數。

## 幾件要知道的事

- **進出場價。** 回測與結算都用隔日開盤進場、第 5 個交易日收盤出場，沒有排除開盤漲停買不到的情況。
- **倖存者偏差尚未修正。** `universe.csv` 是固定名單，回測結果仍可能偏樂觀。
- **籌碼資料要等盤後公布。** 排程設在 17:30 是為了等 FinMind 更新完當日法人與融資券；若發現籌碼特徵常是前一天的，把 cron 往後調。
- **FinMind 配額。** 免費註冊 600 次/小時，第一次全量抓 110 檔約 450 次請求，剛好夠；股票池超過 140 檔就會卡，分兩次跑或付費。
- **回測若命中率 > 70%，先懷疑資料洩漏**，不要高興。
- **`predictions_log.csv` 只增不改。** `predict.py` 會跳過已存在的 (日期, 股票)，Supabase 也用 ignore_duplicates；`score.py` 只填結果欄位。
- 本專案所有輸出皆為統計結果，不構成投資建議。

## 下一步可以加的

- 外資台指期淨未平倉、台積電 ADR 隔夜報酬（FinMind 有 `TaiwanFuturesInstitutionalInvestors`、美股可用 yfinance）
- 元標籤（meta-labeling）：第二層模型判斷每個訊號該不該信，提高 precision
- 法說會／月營收公布後的事件型訊號
- Notion 鏡像：把 predictions 另外寫一份到 Notion Database 方便手機翻
