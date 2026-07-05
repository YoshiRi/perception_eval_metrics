# 自動運転向け Perception 安全評価の方向性メモ

## 背景

現在の評価は物体単位（bbox単位）の Precision / Recall / mAP が中心である。

しかしこの評価では、

- 密集した群衆の中の1人を見逃す
- 車両直前の1人を見逃す

が同じ False Negative として扱われる。

また、

- 存在しない障害物を検出して誤停止する
- 実際の障害物を見逃して衝突リスクを生む

も十分に区別できない。

Perceptionの品質を安全性や運転性能と結び付けるためには、Object-centric評価から Occupancy-centric評価へ移行する必要がある。

---

## 基本方針

評価の基本単位を物体ではなく空間占有（Occupancy）とする。

Canonical representation:

```
Occupancy O(x, y, t)
Velocity  V(x, y, t)
```

bboxや点群は入力表現の一つとして扱い、評価は Occupancy 上で実施する。

これにより、

- bbox検出
- Occupancy Network
- Stixel
- Point Cloud
- Unknown Obstacle
- Prediction

を共通の枠組みで評価できる。

---

## 評価軸

評価は以下の2軸に分離する。

### Safety Risk

本来占有されている空間を自由空間と誤認するケース。

```
GT Occupied ∩ Pred Free
```

例：

- 歩行者見逃し
- 車両見逃し
- 落下物見逃し

安全リスクに直接つながる。

---

### Availability Risk

本来自由空間である場所を占有と誤認するケース。

```
GT Free ∩ Pred Occupied
```

例：

- Phantom obstacle
- 誤停止
- 過剰減速

可用性低下につながる。

---

## なぜ Object 評価では不十分か

Object Recall では、

```
GT: 10人
Pred: 9人
```

は常に FN=1 となる。

しかし、

- 群衆中の1人欠落
- 車両前方3mの1人欠落

は安全性への影響が全く異なる。

Occupancy評価では、

- 実際に自由空間が増えたか
- 危険な空間が自由空間と誤認されたか

を評価できる。

---

## Phase 1: Occupancyベース評価

まずは時間軸なしで評価する。

### Safety

```
Risk_safety =
Σ false_free_cells
```

### Availability

```
Risk_availability =
Σ false_occupied_cells
```

さらに近距離を重く評価する。

```
w_dist = 1 / r²
```

として

```
Risk =
Σ w_dist(cell) × error(cell)
```

を用いる。

---

## Stixel / Polar指標

デバッグ用途として有効。

各方向θについて

```
r_gt(θ)
r_pred(θ)
```

を定義する。

### Safety

```
max(0, r_pred - r_gt)
```

障害物を遠く見ている量。

### Availability

```
max(0, r_gt - r_pred)
```

障害物を近く見ている量。

---

## 最近傍距離評価の問題

bbox角部では微小な位置ずれで距離が大きく変化する。

そのため単純な最近傍距離比較は不安定。

推奨方法：

### Polar Occupancy Interval

各rayについて

```
occupied interval
```

を比較する。

例：

```
GT   : [5m, 7m]
Pred : [5.2m, 6.8m]
```

のような占有区間比較を行う。

これは角部の不連続に強い。

---

## Forward Corridor Safety Zone（自車前方の重点評価）

### 課題：角度ベースの前方windowは遠方で破綻する

Ray-based TP/FP/FN評価では、当初「自車+x軸を中心に±N°」という角度windowで前方領域を定義していた。

しかし角度windowの横方向カバー範囲は

```
lateral_extent(r) = r * tan(half_angle)
```

であり、距離rに比例して無限に広がる。例えば±60°のwindowは、40m先では左右に約69mもの範囲を「前方」として扱ってしまい、実質的に隣接車線・歩道まで含んでしまう。

現場の運用感覚としては、

- 自車の目の前（1車線幅程度）のFP/FNは絶対に見逃せない
- 少し脇にずれた歩行者のFPは、車両のFPほど重大ではない

という距離に依存しない・クラスに依存する感覚があり、角度windowはこれを表現できない。

### 解決：一定横距離のCorridor

角度windowではなく、中心線からの横距離（lateral offset）で前方領域を定義する。

各rayの最近傍占有点（距離r, 角度θ）について、

```
longitudinal = r * cos(θ)   # 前方成分（>0 が前方）
lateral      = r * sin(θ)   # 中心線からの横距離
```

を計算し、

```
longitudinal > 0  かつ  |lateral| <= half_width
```

を満たす場合のみ評価対象とする。half_widthは車線幅相当（片側1.75m程度）を基準値とし、距離によらず一定幅の「車線」を評価領域とする。

### クラス別Corridor幅

同じ横位置でも、物体の種類によって運用上の重要度が異なる（例：脇にずれた歩行者のFPは、同じ位置の車両のFPほど気にならない）。そのため corridor の half_width はクラスごとに設定できる。

| クラス | half_width (m) | 備考 |
|--------|----------------|------|
| car / truck / bus | 1.75 | 標準的な1車線幅の半分 |
| pedestrian | 1.0 | 車両より狭いcorridorで評価（脇のFPは許容） |
| (未設定クラス) | 1.75 (default) | |

GT側・EST側それぞれ、自分自身のクラスのcorridor幅で独立に判定する（不一致ペアでも各々の妥当性で評価する）。

### Critical Zone：至近距離のFP/FNは無条件でhard-fail

「目の前の誤検知・未検知は全て論外」という要求は、precision/recallの重み付けではなく、別枠のカウントとして扱う。

Corridorよりさらに狭く・近い範囲（half_width 1.0m, range 15m を既定値とする）を Critical Zone と定義し、そこに含まれるFN/FPはクラスを問わず

```
ray_collision_critical: { "FN": n, "FP": n }
```

として常に別表示する。これは per-class precision/recall には混ぜ込まず、「0であるべき」指標として単独で監視する（1件でもあれば hard-fail 扱い）。

### 実装

`occupancy_metrics/ray_collision.py` の `compute_ray_collision_records` が上記ロジックを実装する。設定は `OccupancyConfig` の

```
ray_collision_corridor_half_width_m          # クラス別 half_width (dict)
ray_collision_corridor_default_half_width_m  # 未知クラスの既定 half_width
ray_collision_critical_half_width_m          # Critical Zone half_width
ray_collision_critical_range_m               # Critical Zone range
```

で調整可能。

---

## Phase 2: Reachable Set評価

Pathが存在しない場合でも、

```
Reachable(T)
```

を利用する。

現在速度・操舵制約から、

T秒以内に到達可能な領域を計算する。

評価対象を

```
Occupancy ∩ Reachable(T)
```

に限定する。

これにより、

- 後方
- 到達不能領域

の誤差を過剰評価しない。

---

## Phase 3: Velocity付きOccupancy

Occupancy binに速度がある場合、

静的占有誤差から動的リスク評価へ拡張できる。

例：

```
relative position
relative velocity
```

から

```
TTC
Closest Approach
Required Deceleration
```

を計算する。

安全リスクは

```
false free occupancy
```

に対して

```
f(TTC)
```

で重み付けする。

---

## Phase 4: Candidate Path評価

Pathが利用可能になった段階。

候補経路の Swept Volume を利用する。

```
P(t)
```

に対して

### Safety

```
P(t) ∩ GT Occupied ∩ Pred Free
```

### Availability

```
P(t) ∩ GT Free ∩ Pred Occupied
```

を評価する。

これにより

- 実際に衝突しうる見逃し
- 実際に走行可能なのに阻害されたケース

を直接評価できる。

---

## Phase 5: Prediction-aware評価

最終形。

評価対象を

```
O(x,y,t)
```

へ拡張する。

必要な情報：

```
GT occupancy forecast
Pred occupancy forecast
```

評価例：

```
future collision risk
future false stop risk
```

Prediction品質まで含めたPlanner近接評価が可能になる。

---

## 推奨ロードマップ

| Phase | 内容 | 主要指標 |
|-------|------|----------|
| Phase 1 | Occupancyベース | false free / false occupied |
| Phase 2 | Reachable Set導入 | reachable occupancy error |
| Phase 3 | Velocity導入 | TTC-weighted risk |
| Phase 4 | Candidate Path導入 | path-conditioned risk |
| Phase 5 | Prediction導入 | space-time occupancy risk |

---

## 設計原則

評価APIは以下の形を目標とする。

```python
metric(
  gt_occupancy,
  pred_occupancy,
  ego_state,
  optional_velocity=None,
  optional_path_set=None,
  optional_prediction=None
)
```

PathやPredictionが無い場合でも評価可能であり、将来的にPlanner-awareな評価へ自然に拡張できる構造を維持する。

最終的には「物体を正しく検出したか」ではなく、

- **安全上重要な空間を正しく認識できたか**
- **可用性を損なう誤認識をどれだけ起こしたか**

を評価することを目的とする。
