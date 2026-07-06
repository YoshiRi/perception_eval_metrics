# 指標リファレンス（解釈ガイド）

各指標が「何を測っているか」「単位」「集計方法」「注意点」を一箇所にまとめる。
AIも人間も、この文書に書かれた定義以外の解釈で指標を語らないこと。

---

## 1. Grid Safety Miss Rate

**実装**: `occupancy_metrics/compute.py` の `compute_occupancy_risk` / `_summarise_grid_totals`

```
safety_risk_cells = |{ cell : GT Occupied(cell) かつ Pred Free(cell) }|   (評価範囲内、全フレーム合算)
gt_occupied_cells = |{ cell : GT Occupied(cell) }|                       (評価範囲内、全フレーム合算)

Grid Safety Miss Rate = safety_risk_cells / gt_occupied_cells
```

- **単位**: 割合（0〜1、%表示）
- **集計方法**: 全フレームの該当セル数を先に合算してから割る（フレームごとの比率の平均ではない）。
- **感度・注意点**:
  - Grid解像度（`resolution_m`）に依存する。解像度を変えると同じミスでもセル数が変わり、値が変動する。
  - 物体サイズに敏感。バスのような大きい物体の見逃しは、歩行者の見逃しよりcell数への寄与が大きい＝同じ「1物体の見逃し」でもクラスによって値への影響度が違う。
  - このため **単体で経時比較・バージョン比較の主指標にはしない**。解像度・評価範囲・クラス構成が同じ条件下でのみ相対比較に使う補助指標という位置づけ。

---

## 2. Polar Safety Mean

**実装**: `occupancy_metrics/polar.py` の `compute_polar_risk`、`compute.py` の `run_all` で加重平均

各ray（方向）θについて、GTとPredそれぞれの最近傍占有距離 `r_gt(θ)`, `r_pred(θ)` を求める。

```
safety_per_ray(θ) = max(0, r_pred(θ) - r_gt(θ))
```

これは「Predが実際（GT）より遠くに障害物があると認識している量」＝**近い障害物を見逃して、より遠くの障害物しか見えていない/何も見えていない、という危険側の誤差**を表す。逆方向（Predが近くに誤検知している量）は Availability 側の指標であり、Safety Meanには含まれない。

```
Polar Safety Mean = 全フレーム・全valid ray の safety_per_ray の平均
                   （両方とも無占有のrayは分母から除外。フレームごとのvalid ray数で加重）
```

- **単位**: メートル（m）
- **意味**: 値が大きいほど「本来近くにあるはずの障害物を、より遠くに（＝安全側に寄せて）誤認識している」度合いが大きい。
- **注意点**:
  - bboxの角部で最近傍距離が不安定になりやすく（doc内「最近傍距離評価の問題」参照）、単発の外れ値に弱い。
  - 平均値なので、少数の大きな外れ値（`safety_max`）に引っ張られやすい。`safety_max`も併記して外れ値の有無を確認すること。
  - 「前方だけ」を見ているわけではなく、全方位のrayが対象（後述のRay-based Forward Corridorとは評価範囲が異なる）。

---

## 3. 衝突リスク (False Free) = Polar Occupancy Interval の `false_free_m`

**実装**: `occupancy_metrics/polar.py` の `compute_polar_interval_risk`（`_interval_diff_length`）

各rayについて、GTの占有区間とPredの占有区間（マージ済み区間）を比較する。

```
false_free_m = Σ_ray ( GTの占有区間のうち、Predの占有区間でカバーされていない長さ )
             （全ray・全フレームを単純合算した累積値）
```

- **単位**: メートル（長さの合計）。**フレーム数で正規化されていない生の累積値**。
- **なぜこの指標が推奨されるか**: 最近傍距離の単純比較（Polar Safety Mean）はbbox角部で不安定になりやすいが、区間（interval）の被覆比較はこの不連続に強い。より安定した「見逃しの総量」の指標。
- **重要な注意点**:
  - **フレーム数（`n_timestamps`）が異なるデータセット間では絶対値を直接比較できない。** 今回のようにバージョン間で比較する場合、同一データセット・同一フレーム数で評価されていることが前提。
  - 値が大きいほど「GTが占有と認識している空間を、Predが自由空間だと誤認識している総延長」が大きい＝見逃しの総量が大きい。
  - 「Availability（誤って占有と誤認識した量）」は対になる `false_occupied_m` であり、`false_free_m` には含まれない。

---

## 4. Ray-based Forward Corridor TP/FP/FN + Critical Zone Hard-Fail（今回追加分）

**実装**: `occupancy_metrics/ray_collision.py`

上記1〜3が「セル数」「距離差」「区間の被覆長」という **量的な誤差** を測るのに対し、この指標は自車前方の衝突回避に直結する物体に限定して **TP/FP/FNというイベント単位** でカウントし、precision/recallという別の切り口で解釈する。

- **評価範囲**: 自車前方に伸びる、距離によらず一定幅の帯（Corridor）のみ。範囲外（隣接車線・後方）は評価対象外。
- **Corridor幅はクラス別**: car/truck/bus は中心線から±1.75m（1車線幅相当）、pedestrianは±1.0m（脇にずれた歩行者の誤検知は許容する設計）。
- **Critical Zone**: Corridorよりさらに狭く・近い範囲（±1.0m、15m以内）でのFN/FPは、クラスを問わず「無条件で許容できない」hard-fail件数として、通常のprecision/recallとは別枠で常時表示する。
- 設計の詳細・数値根拠は `docs/occupancy_eval_direction.md` の「Forward Corridor Safety Zone」章を参照。

---

## 5. Roadside VRU Focus（路肩の歩行者・自転車のロスト/誤検知）

**実装**: `occupancy_metrics/ray_collision.py` の `compute_roadside_records`

上記4（Forward Corridor）が自車前方一定幅の**内側**のみを評価するのに対し、こちらはCorridorの**外側**（歩道・路肩に相当する側方領域）を評価対象とし、対象クラスも歩行者・自転車など（既定: pedestrian, bicycle）に限定する。TP/FP/FNの判定・precision/recallの計算方法はForward Corridorと同じ枠組みを再利用している。

```
longitudinal = r * cos(θ)   # 前方成分（>0 が前方）
lateral      = r * sin(θ)   # 中心線からの横距離

longitudinal > 0  かつ  lateral_min <= |lateral| <= lateral_max
```

を満たし、かつラベルが対象クラス（`roadside_classes`）に含まれる場合のみ評価対象とする。

- **評価範囲**: 既定で中心線から横距離1.0m〜4.5mの帯（Corridorの境界1.75mより外側から、標準的な歩道幅程度まで）。Corridorのような「以内」ではなく、内側・外側の両方に境界がある帯である点に注意。
- **対象クラス**: 既定 pedestrian, bicycle。車両などその他クラスは、この帯の中にあっても対象外（Corridor側の評価に委ねる）。
- **Critical Zoneのような hard-fail 別枠は無い**: Forward Corridorと対等な「focus/track」用の指標として、per-class precision/recallのみを報告する。
- 設計の詳細は `docs/occupancy_eval_direction.md` の「Roadside VRU Focus」章を参照。
- **既知の注意点**: `evaluate_timestamp` はPolar変換時に `cfg.classes ∪ cfg.roadside_classes` でフィルタするため、`roadside_classes` に `cfg.classes` に含まれないクラス（例: bicycle）を指定してもこの指標には正しく現れる。ただしGrid系の指標（Safety/Availability、per-classブレークダウン）は従来通り `cfg.classes` のみを見ているため、`roadside_classes` にしか無いクラスはGrid系には一切出てこない（設計通り）。

---

## どの指標をいつ見るか（早見表）

| 知りたいこと | 見る指標 |
|---|---|
| 大まかな検知漏れの傾向（解像度・データセット固定での相対比較） | Grid Safety Miss Rate（補助指標として） |
| 方向ごとの見逃し距離の大きさ・外れ値の有無 | Polar Safety Mean + Safety Max |
| 見逃しの総量（bbox角部の不安定性に強い、より信頼できる量的指標） | 衝突リスク (False Free) = interval `false_free_m` |
| 前方の衝突対象物体を、TP/FP/FNというイベント単位・クラス別precision/recallで見たい | Ray-based Forward Corridor TP/FP/FN |
| 「自車の目の前」だけは絶対に見逃し・誤検知してほしくない、という運用要件のチェック | Critical Zone Hard-Fail件数（0であるべき） |
| 路肩の歩行者・自転車のロスト/誤検知 | Roadside VRU Focus TP/FP/FN |
