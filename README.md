# Hawkes Process Simulator

Hawkes過程の挙動を体験できるインタラクティブな可視化ツールです。様々なカーネル関数を選択して、自己励起過程（Self-Exciting Process）の特性を探索できます。

## このアプリでできること

### **リアルタイムシミュレーション**
パラメータを調整して、Hawkesプロセスの挙動をその場で確認できます：
- イベント発生のタイミングを可視化
- 強度関数λ(t)の時間変化をプロット
- イベント間の相互作用を観察

### **5種類のカーネル関数**
異なる励起パターンをモデル化できる、多様なカーネル関数に対応：

| カーネル | 特徴 | 適用例 |
|---------|------|--------|
| **指数減衰** | 短時間記憶、最も一般的 | 金融市場の取引、SNSの投稿 |
| **べき乗則 (Omori)** | 長時間記憶、ゆっくり減衰 | 地震の余震、ニュース拡散 |
| **複数指数** | 複数の時間スケール | 短期と長期の両方の影響 |
| **ガウス** | 遅延ピーク、滑らかな励起 | 遅延を伴う反応パターン |
| **矩形** | 一定期間の一定励起 | 固定期間の影響モデル |

## 参考文献

Hawkesプロセスの理論的背景については、以下を参照してください：

- Hawkes, A. G. (1971). "Spectra of some self-exciting and mutually exciting point processes." Biometrika.
- Ogata, Y. (1981). "On Lewis' simulation method for point processes." IEEE Transactions on Information Theory.
- Omori, F. (1894). "On the after-shocks of earthquakes." Journal of the College of Science.
