ECE285 四模型统一评测与完整报告

范围
固定初始Qwen2.5-Math-1.5B，以及同一轮约1M/seed17的A、C-mix、C-bounded最终LoRA。
共同MATH500500题、GSM8K1319题；每题一次greedy，cap2048，seed285，原始prepared prompt与reward-repair-v3验答。
评测不加载PRM，不改checkpoint，不投票、不选最佳测试checkpoint，不新增训练。

当前项目路径绑定的可复现快照（不是通用安装包）
本地源码：../../experiments/final_four_eval_20261029/
远端项目：/srv/research/ece285_math_rl_project
远端评测源码：experiments/final_four_eval_20261029
远端结果：runs/Final_four_eval_seed17_1M_20261029
唯一派发：dispatch.json（已有任务，不要重新运行dispatch.py或prepare_remote.py）

本地命令
查看状态：python3 inspect_remote.py
更新本地报告：python3 build_report.py
完成后一次性取回：python3 retrieve_results.py（已有archive时禁止重复覆盖）
源代码和控制器均有独立锁、source release和contract；有已提交batch时只允许在核验后恢复，不能从头重算。

完成后的交付文件
review.html：完整中文报告与三张图
REPORT.txt：核心数值与局限
accuracy.csv / comparison.json：结构化结果
qualitative_review.json：选定真实回答的助手数学审读（非人工金标准）
signal_case_analysis.json：实际混合组奖励到优势的复算及明确标注的数学反例
training_reward_diagnostics.json：既有训练回答的PRM分数/组类型/优势诊断
retrieval_acceptance.json：本地预测/原始token/正确数/adapter文件hash验收
completed_evidence.tar.gz / evidence/：远端评测代码、固定配置、批次、8份预测、CPU验收和三个LoRA adapter副本

模型与版本
Base: Qwen/Qwen2.5-Math-1.5B, revision4a83ca6e4526a4f2da3aa259ec36c259f66b2ab2。
torch2.8.0 / vLLM0.10.2 / transformers4.56.2 / peft0.17.1 / TRL0.23.1 / math-verify0.8.0。
基础模型约3.1GB未重复下载，远端现有文件按原assets receipt逐个hash核验；最终交付包含三个LoRA的权重与配置。
原始MATH500/GSM8K数据版本与排重记录在evidence/data/prepared/manifest.json；推理协议、数据文件hash、模型/adapter/source hash在contract.json；单位元数据与验答源码随包保留。

复现与解释要求
在新目录复现实验前先调整common_eval.py中的ROOT/PILOT/RUN/AUDIT，重新生成release并CPU预检。不可覆盖本次完整结果。
完整训练的来源是既有c_bounded_pilot_20261029_v2及其冻结依赖；本轮只做该checkpoint的推理。
训练只有一个seed；题目配对区间不代表跨训练seed稳定性。不把验证集成绩混为测试集，不把结构合规或PRM分数当作数学过程金标准。
共享GPU的耗时和采样显存仅表示本轮成本，不能证明算法独占卡速度。
生成的代码和声称的执行结果只作为文本阅读，从未执行。
