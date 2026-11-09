C/D PRM 小规模配对试验 — 2026-10-21 GPU 版本

授权范围
只做 C/D 的小规模离线 PRM 适配和评分、奖励、优势检查。不更新策略模型，不启动 C/D 完整 RL。不停止、重启或修改 B。所有试验文件位于独立 experiments/cd_prm_pilot_20261021/，不写 training_b/、正式 proposal 或 protocol_simple.json。

规模与控制
16 道 MATH 训练题，由原始 Qwen2.5-Math-1.5B 采样，每题 4 条，2048 completion token 上限。只收完整且可判分的解答。
128 个公共步骤锚点，正负各 64；来自 H4/prm800k-trl-dedup 的固定版本 fd8cba17d8214bfbabf6e7e10f29da4e6614dd6c。原始文件 SHA-256 校验通过后使用。
锚点限原始 MATH train，并排除验证、测试、全部 ProcessBench 和本次 B 检查题的归一化题目文本；初始策略适配题也排除这些检查题。未知原 PRM 的训练暴露和改写重复仍不能排除。
原 H4 PRM，LoRA rank 8、alpha 16、dropout 0；原模型及原分类头冻结。两组相同 seed 17、相同初始化、数据、批次顺序、AdamW、学习率 2e-5、32 次更新。每次 1 条失败、1 条成功、1 个正锚点、1 个负锚点。
C = L_fail + L_success + L_step。
D = L_fail + 0 * L_success + L_step。
每个分量分别均值，删除成功项后不重新平均剩余项。采用 detach 的 softmax 加权错误概率池化，温度 0.2。

检查
1. 27 个 B-6M 原始八回答组，共 216 条：三个结果类型各哈希选 8 组，加 3 组已知错误例子。已知错误是事后选出的示例，分开报告，不作错误率估计。
2. 固定 96 条 ProcessBench 解答：答对且过程有效、答对但过程有错、答错且过程有错各 32 条。报告错误定位、有效推导接受率、调和平均和答对内部有效/无效 AUROC，以及配对题目 bootstrap 区间。此为分层小样本诊断，不是官方完整 ProcessBench 分数。
3. B/C/D 均用同一 GPU FP32 后端，TF32 关闭重新评分。对原始八回答组保留相同结果奖励、无效输出处理、min 聚合、0.2 系数、group 标准化和 1e-4 epsilon，直接重算实际优势。
4. 检查原高分错答的分数、有效回答内排名及实际优势；分数下降不等于学习信号改善。所有同结果组仍受原有标准化结构影响。
5. 技术门槛包括实际模型的梯度差等于成功监督梯度、仅 LoRA 参数更新、保存加载一致、C/D 初始适配器哈希及批次一致、梯度有限、冻结权重不变、重复评分一致。

运行方式与边界
执行中发现 B 在约 14.24M token 自行因优势核对断言退出（最大差 2.947e-5，容差 2e-5）；未对 B 发出停止信号或修改 B 源码。停止自己尚未完成的 CPU 采样及等待控制器，保留原记录；在 B 释放的 GPU0 上继续本次 pilot。使用 GPU FP32、关闭 TF32，8 CPU 线程、nice 15。16 道题、128 锚点、检查集与损失预算完全不变；C/D 均从同一原 PRM 重新初始化。没有依据诊断结果选模型或更改超参。B 源码哈希在各阶段核对。
prepare.py -> preflight_model.py -> adapt.py -> check.py，由 run_pilot.py 一次性顺序执行；检查完成退出。失败保留现场并停止后续阶段。禁止重复启动或覆盖已有结果。
采样使用 Transformers generate 的 FP32 推理，正式策略 RL 使用 vLLM/BF16；两组共用同一份采样结果，后端差异会限制向正式训练外推。
本次是明确缩小规模的 pilot，不替代正式 512 x 4 缓冲区、2048 锚点和 64 x 8 验证 probe，也不执行正式重复次数资源门槛。不按这次诊断的胜负挑 checkpoint、改学习率或追加更新。
不训练 S，因此 C/B 的改变不能拆成锚点和失败监督各自的贡献。D/C 只隔离成功弱监督系数。
原始策略采样是推理；技术 smoke 适配器仅用于实现验证，绝不会加载进 C/D 或用于结果评分。

远程目录
/path/to/ece285_math_rl_project/experiments/cd_prm_pilot_20261021_gpu

关键文件（完成后产生）
pipeline_status.json：一次性流水线状态与源码哈希。
artifacts/buffer_manifest.json、anchor_manifest.json、batch_manifest.json：数据、批次与来源。
artifacts/C/completion.json、D/completion.json：各组 32 步完成记录。
artifacts/paired_wiring_check.json：实际首步梯度对照。
artifacts/scores_B.jsonl、scores_C.jsonl、scores_D.jsonl：同样本评分。
artifacts/group_advantages.json：逐组奖励和优势，可复算。
artifacts/summary.json：结果、区间、案例与限制。没有此文件就不能声称检查完成。

原始来源
https://huggingface.co/HuggingFaceH4/Qwen2.5-Math-1.5B-Instruct-PRM-0.2
https://huggingface.co/datasets/HuggingFaceH4/prm800k-trl-dedup
https://github.com/openai/prm800k
