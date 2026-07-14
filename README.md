# HBF MFU公式仿真器

本项目依据《端侧HBF云端应用MFU计算任务书》中的计算流与公式，构建M12-24B模型的基础端到端公式仿真器。

当前版本坚持两个原则：

- 只实现任务书能够明确展开的部分。
- 对量纲错误和数据依赖错误做最小修正，不使用结果拟合或隐藏假设。

## 项目结构

```text
知存项目/
├─ 01_任务书/       原始任务书
├─ 02_输入模板/     用户填写的硬件指标Excel模板
├─ 03_仿真器源码/   仿真器与公式测试
└─ 04_汇报材料/     PPT、讲稿和截图素材
```

## 输入

硬件指标从 `02_输入模板/硬件指标填写模板.xlsx` 读取，包括：

- 芯片数量
- 单芯片Matrix算力与有效利用率
- HBF读写带宽及访问延迟
- DDR带宽
- PCIe有效带宽及固定延迟

单次仿真还需提供：

- `mode`：`prefill` 或 `decode`
- `batch-size`
- `input-seqlen`
- `history-seqlen`
- `tail-kv-tokens`：历史长度大于0时必填

## 运行

Prefill示例：

```powershell
python .\03_仿真器源码\HBF_MFU公式仿真器.py `
  --hardware-file .\02_输入模板\硬件指标填写模板.xlsx `
  --mode prefill `
  --batch-size 1 `
  --input-seqlen 1024 `
  --history-seqlen 0 `
  --csv .\仿真汇总.csv `
  --detail-csv .\逐层时序.csv
```

Decode示例：

```powershell
python .\03_仿真器源码\HBF_MFU公式仿真器.py `
  --hardware-file .\02_输入模板\硬件指标填写模板.xlsx `
  --mode decode `
  --batch-size 1 `
  --input-seqlen 1 `
  --history-seqlen 1024 `
  --tail-kv-tokens 16 `
  --csv .\仿真汇总.csv `
  --detail-csv .\逐层时序.csv
```

## 输出

- `MFU (%)`
- `HBF read util (%)`
- `HBF write util (%)`
- `E2E latency(ms)`
- 可选的48层逐层时序明细

## 测试

```powershell
cd .\03_仿真器源码
python -m unittest -v .\测试_HBF_MFU公式仿真器.py
```

## 当前边界

- 已支持单场景、非AFD的prefill/decode基础计算链路。
- AFD缺少完整时序公式，当前不输出AFD指标。
- `tail_kv_tokens`在任务书中没有定义，独立decode场景需要用户明确填写。
- top-k延迟、vector/softmax、SRAM tiling以及通信计算重叠尚未详细建模。
- 当前结果代表公式模型输出，不代表已经完成真实硬件校准。
