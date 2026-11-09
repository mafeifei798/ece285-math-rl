旧6M A按四模型统一协议重评

本次仅新增历史A checkpoint-284的MATH500/GSM8K greedy预测，共1819条。
初始模型与当前1M A复用execution/final_four_eval_20261029的已验收预测。
无训练、PRM打分、提示或验答修改、checkpoint选择、额外seed或自动任务。

远端项目：/path/to/ece285_math_rl_project
源代码：experiments/a6m_protocol_recheck_20261029
运行：runs/A6M_protocol_recheck_20261029
审计：audits/A6M_protocol_recheck_20261029
唯一派发：dispatch.json；已派发后禁止重跑prepare_remote.py和dispatch.py。

推理：同一Qwen2.5-Math-1.5B base及历史LoRA，adapter ID1，BF16，seed285，
temperature0/top_p1/top_k-1，2048回答上限，4096上下文，batch64，max_num_seqs64，
max_num_batched_tokens4096，max_loras1/max_cpu_loras3/rank16，eager，GPU1利用率参数.15。
使用相同prepared prompt、数据与reward-repair-v3验答及单位元数据。
唯一调度区别：本次只生成一个模型，不执行上一轮四模型交替。不能据greedy假设逐位确定。

只读状态：python3 inspect_remote.py
已完成后一次取回：python3 retrieve_results.py（拒绝覆盖已有archive/evidence）
本地报告：python3 build_report.py
无定时跟进；wait_for_completion.py仅为当前控制器的有限只读等待。

交付：review.html、REPORT.txt、accuracy.csv、comparison.json、accuracy_comparison.png、
retrieval_acceptance.json、completed_evidence.tar.gz、evidence/。
四个定向数学/答案框样例是助手审读，不是人工金标准，不估计总体比例，也没有执行模型生成的代码。

证据包保存本次完整源/配置/批次/预测/日志、8份新旧参考预测、两个LoRA副本、数据和验答源码。
这是绑定当前项目目录的证据快照，不是通用安装包；基础模型不重复打包，来源及hash在assets/contract。
独立重现前需准备相同base/依赖、参考目录和预检所需历史receipt，并修改新run路径，不能覆盖本次结果。

解释：旧6M A与当前1M A同base、数据和LoRA初始化，但不是同一条不间断训练轨迹。
旧训练vLLM利用率参数.3，当前.15，重启历史也不同。不能把准确率差全部归因token数量，
也不能由纯A的较长训练结果判断过程奖励的较长训练效果。只有一个训练seed。
