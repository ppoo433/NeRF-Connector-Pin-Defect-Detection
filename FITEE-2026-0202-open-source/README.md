# Geometry-Induced 3D Defect Detection for Automotive Connector Pins Using Neural Radiance Fields

FITEE-2026-0202 论文最终主线的源码整理包。

处理流程：COLMAP / Nerfacto 重建 → MAMR 尺度恢复 → PointNet++ MSG 分割与 OBB 修正 → BAR 基座配准 → 中心线提取 → 位置＋方向容差特征 → TGAD LightGBM 判决。

## 内容

| 目录 | 功能与入口 |
| --- | --- |
| `scale_recovery/` | MAMR 标记尺度恢复，CLI：`colmap-scaler` |
| `segmentation/` | PointNet++ MSG 训练、推理和最终 quantile＋OBB 修正 |
| `detection/jiance/registration/` | BAR：`register_overall_then_base_v1.py` |
| `detection/jiance/pin_cloud2center_line.py` | graph-slab 中心线提取 |
| `detection/jiance/tolerance_band_v1/` | 用自备金样本构建位置／方向容差模型 |
| `detection/jiance/tolerance_band_train_manualgt_dir_v1/` | 论文实际使用的方向特征生成、GT 接口和特征表导出 |
| `detection/modeling/` | 最终 LightGBM 拟合及预测；配置在 `detection/configs/tgad_selected.json` |

未收录对照方法的独立程序、历史版本、超参搜索队列、CT 验证支线和实验结果。必要的共享模块保留内部辅助函数。

## 数据与环境

**数据集不公开。** 本目录不含原始图像、点云、CT、标注、数据划分表、训练／测试特征表、预测缓存或模型权重。金样本生成的容差模型、参考中心线及 OBB 坐标也不提供，使用者需用自己的数据生成或配置。

各阶段分开配置环境：MAMR 要求 Python >=3.12，依赖见其 `pyproject.toml`；几何／检测使用 NumPy、SciPy、Open3D 0.19、pandas、scikit-learn、matplotlib、LightGBM；分割另需相互匹配的 PyTorch、PyG、torch-cluster 等扩展。依赖清单是功能要求，未作为跨平台锁定环境验证。

点云坐标采用 **米**；几何参数后缀 `_mm` 使用毫米，`UNIT_SCALE=0.001`。COLMAP 和 Nerfacto 使用其上游实现，本包不重复分发。

## 使用示例

以下占位路径均需替换为自备输入。Windows Conda 环境通过 `conda run` 执行，例如：

```powershell
& "C:\ProgramData\miniconda3\Scripts\conda.exe" run -n cloud --no-capture-output python detection/jiance/pin_cloud2center_line.py --help
```

尺度恢复见 [`scale_recovery/README.md`](scale_recovery/README.md)。分割入口默认采用 RGB＋density、k=4、正类权重 0.75、DropBlock=0.05、4096 点块、50 epochs：

```text
python segmentation/run_mainline.py train --data-dir YOUR_LABELLED_PLY_DIR --split-csv YOUR_SPLIT.csv
python segmentation/run_mainline.py infer --data-dir YOUR_LABELLED_PLY_DIR --split-csv YOUR_SPLIT.csv --train-run-id YOUR_TRAIN_RUN_ID
```

原实验脚本要求 `<sample_id>.ply` 包含 XYZ、RGB、`scalar_pin` / `scalar_base` 标注；划分 CSV 含 `sample_id,role,cv_fold`，role 为 `trainval` 或 `test`。这些文件均需自备。此入口保留论文实验接口，未封装为通用无标注推理服务。训练产物写入 `segmentation/debug_out/`。

OBB 修正只保留最终 quantile 阈值 `rho=0.425` 与配准后基座 OBB 约束。复制 `segmentation/configs/obb_template.json`，填入自己的参考点云路径和三个 OBB，再执行：

```text
python segmentation/postprocess.py --cloud YOUR_CLOUD.ply --prediction YOUR_PREDICTION.npz --config YOUR_OBB_CONFIG.json --output NEW_SEGMENT_DIR
```

输入点云应已完成 SOR 清理（50 邻点、std_ratio=1.5），并与推理概率的点序一致。输出 `pins.ply` / `base.ply`。

容差构建与特征生成保留原程序接口。进入 `detection/`，使用自己的金样本设置 `DAIMA_TOL_INPUT_DIR` 后运行 `jiance/tolerance_band_v1/build_centerline_tolerance.py`；批处理检测用 `--manifest_csv` 指定自己的样本清单，并用 `DAIMA_TOLERANCE_JSON_PATH` 指向生成的容差模型。参考／待检点云和 GT 缓存均需自备。后续通过 `build_feature_lab_v1.py`、`export_research_modeling_pack_v1.py` 构建训练／预测特征表；各脚本的 `--help` 列出输入参数。

最终 TGAD 可直接对自备特征表拟合并预测：

```text
python detection/modeling/tgad.py --train-csv YOUR_TRAIN_FEATURES.csv --predict-csv YOUR_PRESENT_PIN_FEATURES.csv --output NEW_TGAD_OUTPUT
```

训练表需有 `gt_abnormal` 和配置中列出的特征列；预测表不需要标签，需先限定为实际存在的针脚。追加 `--evaluate` 才读取预测表中的评估标签及 `test_gt_geo_included`。测试标签不参与训练或调参。最终模型为 `lgbm_regularized__position_direction_selected`，固定阈值 **0.5755495578059794**，不使用样本级 top-k 后处理。

## 整理与验证边界

源码以论文冻结定义、2026-05-04 TGAD 配置及相应程序快照筛选；整理仅移除旧判决后处理和未采用的独立路线，并适配路径及运行入口。共享的 `baseline/common.py` 在当前源目录中缺失，本包从 **2026-04-06 运行归档**恢复其所需工具函数；尚无法逐字节确认它与论文最后一次运行版本一致。

本包用于源码公开准备。语法、依赖和合成数据检查不能替代论文数据集复现；未随包提供数据、权重与容差模型，也未声称重新得到论文指标。文件来源及清理范围见 `SOURCE_MANIFEST.csv`。

第三方许可见 [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)。作者自有代码的开源许可证尚待作者选定，本次未代为授予许可。
