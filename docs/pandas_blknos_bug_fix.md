# rpc_feed ParquetWriter `_blknos` 崩溃问题分析与修复

> 日期: 2026-09-01
> 作者: AI 辅助排查
> 状态: ✅ 已加固并验证通过（防御性修复；原始错误为偶发，无法确定性复现）

---

## 1. 现象

执行 `scripts/pipeline.py`（`fund.graphml`，数据集 `202608`）时，consumer worker 在写入 parquet 前的分区列构造阶段崩溃，**三级连锁异常**：

```
1. KeyError: 'date'                                  ← 正常：新列走 insert 分支
2. IndexError: tuple index out of range              ← 真正的错误
   pandas/core/internals/managers.py, in _insert_update_mgr_locs
       blk = self.blocks[blkno]
3. IndexError: tuple index out of range（第二次!）    ← 连错误日志都打不出来
   打印 f"Error in consumer worker {item}" 时对损坏 DataFrame 做 repr
   pandas/core/internals/concat.py, in _get_block_for_concat_plan
       blk = mgr.blocks[blkno]
```

调用栈关键位置：

```
async_consume_worker (to_graph.py:138)
  → ParquetWriter.next (writer.py:282)  await asyncio.to_thread(self._write_parquet, meta)
    → _write_parquet (writer.py:258)    meta = self._make_partition(meta)
      → _make_partition (writer.py:248) meta["date"] = dt.dt.strftime("%Y%m")   ← 崩溃点
```

第 3 层异常从 `print` 里逃逸，沿 `asyncio.gather(*consumers)` 一路抛出，**整个 pipeline 进程退出**。

---

## 2. 根因分析

### 2.1 错误本质：BlockManager 缓存与 blocks 不同步

pandas 2.1.x 的 `DataFrame` 内部由 `BlockManager` 管理：

```
DataFrame
  └── _mgr: BlockManager
        ├── blocks: tuple[Block, ...]     # 实际按 dtype 分组存储的 2D 数组
        ├── axes[0]: Index                # 列名
        ├── _blknos:  ndarray[intp]       # 缓存: 第 i 列 → blocks 下标
        └── _blklocs: ndarray[intp]       # 缓存: 第 i 列 → 所在 block 内的位置
```

`_blknos` 是**惰性构建、不校验失效**的缓存：首次访问时从 `blocks` 重建，之后除非显式更新，不会再被重新验证。崩溃两处（`insert` 与 `repr`→`concat`）都在 `self.blocks[self._blknos[i]]` 上炸出 `IndexError`，说明该 DataFrame 的 `_blknos` 数组内容已经与 `blocks` 元组**长度/内容不一致**——即 manager 内部状态损坏，属于 pandas 2.1.x block-manager 的已知 bug 类别（2.2 系列重写了这部分实现）。

### 2.2 触发条件（代码侧的踩雷写法）

崩溃点的旧代码（commit `1552efa` 引入的向量化版本）：

```python
def _make_partition(self, meta: pd.DataFrame) -> pd.DataFrame:
    dt = meta["datetime"]                # ← ① 提前持有来自 meta 的列引用
    meta["year"] = dt.dt.year.astype(str)      # ← ② 原地 insert 新列
    meta["quarter"] = "Q" + dt.dt.quarter.astype(str)  # ← ③ 再 insert
    meta["sid"] = meta["sid"].str.replace(...) # ← ④ 原地 setitem 既有列（_iset_item）
    meta["date"] = dt.dt.strftime("%Y%m")      # ← ⑤ 再 insert（此列的 value 仍是
    meta["datetime"] = dt.dt.tz_localize(...)  #    从 ① 的引用派生、共享 Block 引用）
    return meta
```

风险组合：

1. **跨多次原地 `insert`/`setitem` 持有列引用**（`dt = meta["datetime"]`）：该 Series 与父 DataFrame 共享 Block 及其引用计数（`refs`），pandas 2.1.x 的 `_iset_item`/`_insert_update_*` 在块拆分、合并、删除时对 `_blknos`/`refs` 的维护存在已知缺陷；
2. **插入的 value 本身派生自被修改的 DataFrame**（共享内存/引用），走了 `new_block_2d(refs=...)` 的引用计数路径；
3. **并发执行**：`CONSUMER_WORKERS=4` 个 consumer 协程通过 `asyncio.to_thread` 在共享线程池里同时跑 `_write_parquet`，向量化后 writer 速度提升约两个数量级，线程交错时序完全改变。

### 2.3 重要结论：偶发、非数据相关

系统性复现尝试**全部失败**（即当前代码 + 当前数据无法确定性触发）：

| 实验 | 规模 | 结果 |
|------|------|------|
| 新代码单线程 `_make_partition`（真实 `202608` 数据 + pickle 往返模拟跨进程） | 全量 51971 个文件 | 0 失败 |
| 4 线程并发（模拟 4 个 consumer + `asyncio.to_thread`） | 400 个真实帧 | 0 失败 |
| 完整真实 pipeline（loky 多进程 + 4 消费线程 + 真实 ParquetWriter） | 全量 fund 文件（sh51/sz15/sz16） | 成功，0 错误 |
| 旧代码（`1552efa` 之前的 apply 版本）对照 | 10% 采样 | 0 失败 |
| pandas 版本核对（`poetry.lock` git 历史） | — | 一直为 2.1.4，未变 |

因此定性为：**pandas 2.1.4 BlockManager 缓存损坏的运行时偶发问题**（线程时序/内存压力相关），`1552efa` 的向量化改动改变了 writer 线程的执行时序，最可能只是触发者而非确定性根因。`.01` 数据本身已验证无问题。

---

## 3. 修复方案

### 3.1 `rpc_feed/core/graph/node/writer.py::_make_partition`（防御性加固）

原则：**不在多次原地 `insert`/`setitem` 期间持有来自同一 DataFrame 的列引用；插入的 value 先计算为独立 Series；用 `assign` 写入副本，不原地修改传入数据**。

```python
def _make_partition(self, meta: pd.DataFrame) -> pd.DataFrame:
    # 注意：不要在多次原地 insert/setitem 期间持有来自 meta 的列引用 ——
    # pandas 2.1.x 的 BlockManager 在此场景下可能出现 _blknos 缓存与 blocks
    # 不同步（IndexError: tuple index out of range，且后续 repr 也会崩溃）。
    # 这里先把所有分区列的值计算为独立 Series，再一次性 assign 到副本上。
    dt = meta["datetime"]
    year = dt.dt.year.astype(str)  # quarter == (month-1)//3 + 1
    quarter = "Q" + dt.dt.quarter.astype(str)
    sid = meta["sid"].str.replace(r'^[a-zA-Z]+\.|\.[a-zA-Z]+$', '', regex=True)
    date = dt.dt.strftime("%Y%m")

    datetime_utc = (
        dt.dt.tz_localize("Asia/Shanghai")
        .dt.tz_convert("UTC")
        .dt.tz_localize(None)
    )
    meta = meta.assign(
        year=year, quarter=quarter, sid=sid, date=date, datetime=datetime_utc
    )
    return meta
```

要点：

- 所有新列值**先完整计算成独立 Series**（与 `meta` 无 Block 共享），再写入；
- `assign` 返回**副本**，原 DataFrame 不被原地修改（对上游数据也更安全）；
- 既有列（`sid`/`datetime`）通过 `assign` 覆盖，新列（`year`/`quarter`/`date`）追加，列顺序与旧实现一致，写入 parquet 的结果不变。

### 3.2 `rpc_feed/core/graph/to_graph.py::async_consume_worker`（防二次崩溃）

原实现的 `except` 分支把整个 `item` 塞进 f-string 做 `repr`——当 `item` 本身就是损坏的 DataFrame 时，`repr` 再次抛 `IndexError` 且逃出异常处理，直接炸掉整个 consumer。修复为只打印类型与异常：

```python
except Exception as e:
    # item 可能已处于损坏状态（repr DataFrame 本身可能再次抛异常），
    # 只打印类型与异常信息，避免二次异常导致整个 consumer 崩溃
    print(f"Error in consumer worker on {type(item).__name__} encounter: {e!r}")
```

这样即使未来再出现类似偶发问题，也只会**丢掉单条 item 并留下可读日志**，而不是终止整个 pipeline。

---

## 4. 验证

| 验证项 | 结果 |
|--------|------|
| 修复后 `_make_partition` 单元验证（真实 202608 数据 50 个文件，断言全部分区列存在） | ✅ 通过 |
| 完整 pipeline 端到端（`fund.graphml`，输出隔离到 `/tmp`，全量 fund 文件，loky 多进程 + 4 消费线程） | ✅ `pipeline 完成`，1448 个 parquet，0 错误 |
| 新旧实现输出一致性 | ✅ 分区列值、列顺序、parquet 写入行为一致 |

---

## 5. 编码约定（防止同类问题）

在 pandas 2.1.x 上操作 DataFrame 时：

1. **不要缓存列引用后跨多次写操作使用**。需要多次派生时，每步重新取列，或先把所有派生值算完再一次性写入。
2. **避免 `df["new"] = <派生自 df 自身的 Series>` 连续多次原地插入**；优先 `df = df.assign(...)` 或 `pd.concat(axis=1)` 一次性构造。
3. **异常处理中不要 repr 整个可能已损坏的 DataFrame**；打印 `type(item).__name__`、`item.shape` 之外的轻量信息，或用 try 包裹日志本身。
4. 长期方案：**升级 pandas 到 2.2.x**（BlockManager 重写，修复了大量此类 `IndexError: tuple index out of range`），升级前需回归验证 provider/rpc 侧的 Arrow 交互。
