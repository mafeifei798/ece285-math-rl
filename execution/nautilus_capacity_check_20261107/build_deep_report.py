"""Assemble read-only NRP evidence and non-persisting dry-run results."""
from pathlib import Path
from html import escape
import hashlib
import json

HERE=Path(__file__).resolve().parent
def read(name):return json.loads((HERE/name).read_text())
def table(headers,rows):return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+escape(str(x))+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+escape(str(x))+'</td>' for x in r)+'</tr>' for r in rows)+'</tbody></table></div>'

def main():
    previous=read('summary.json');deep=read('deep_status.json');admit=read('admission_dryrun.json');available=read('available_gpu_snapshot.json');cross=read('candidate_crosscheck.json')
    assert deep['create_jobs']==deep['create_pvc']=='yes'
    assert all(r['passed'] and r['dry_run_server'] and not r['persisted'] for r in admit['results'])
    assert all(v['returncode']==0 and not v['stdout'] for v in admit['non_persistence_check'].values())
    candidates=[r for r in available['A100_nodes_details'] if r['meets_nominal_requests']]
    candidate_rows=[]
    for c in candidates:
        candidate_rows.append([c['Name'],c['GPUType'],c['GPUAvailable'],c['CPUAvailableFloat'],f"{int(c['MemoryAvailable'])/2**30:.1f}GiB",c['Region']])
    sources={
        '实时资源仪表盘':'https://nrp.ai/viz/resources/',
        'GPU资源申请规则':'https://nrp.ai/documentation/userdocs/running/gpu-pods/',
        '优先级与可抢占资源':'https://nrp.ai/documentation/userdocs/running/priority-classes/',
        '存储类别与CephFS规则':'https://nrp.ai/documentation/userdocs/storage/ceph/',
        '批处理Job':'https://nrp.ai/documentation/userdocs/running/jobs/',
    }
    facts=[
        '账号可在ecepxie创建Job和PVC。普通优先级A100配额仍为12，当前配额计数使用0；这是课题组共享额度，不是个人保留12张卡。',
        '已对100GiB rook-cephfs持久卷及1张80GB A100/8CPU/64GiB内存/60GiB临时盘的Job声明执行server-side dry-run，两项通过，并GET确认相应名称对象不存在。没有创建任何Pod、Job或PVC。',
        'Job dry-run验证声明与当前Job API/admission接受性；它不会创建子Pod，不能替代子Pod策略检查、真实调度、镜像拉取、PVC绑定和挂载读写测试。',
        '官方实时资源表中有一个通过保守筛选的候选节点：80GB A100至少空闲1张、CPU至少8核、RAM至少64GiB、临时盘至少60GiB、Kubernetes Ready且无NoSchedule/NoExecute硬taint。当前80GB A100节点标记sharing-strategy=none。',
        '进一步读取全群其他namespace的Pod请求以交叉计算GPU占用被RBAC拒绝；没有重试绕过权限。因此空闲数量来自官方公开仪表盘快照，不是我们独立汇总所有作业得到的保证。',
        '没有查到namespace总requests.storage硬配额；100GiB申请策略验证通过，但这不是无限空间承诺，也不是100GiB已分配。已有其他项目卷的大小不能算成本项目空闲空间。',
        '常用西区rook-cephfs与rook-ceph-block都支持卷扩容。只读数据和不同Job各自的checkpoint目录适合放在100GiB CephFS卷；环境依赖放容器镜像。',
        '官方禁止在CephFS上pip/conda安装。不能把原服务器项目中的.venv原样搬过去当作已配置好；需镜像固定torch/transformers/vLLM/PEFT版本，编译缓存使用本地scratch。',
        'H100/H200/GH200普通优先级配额为0，但官方另有opportunistic路径，可绕过普通GPU配额，代价是可能随时被抢占。首轮仍推荐A100，避免把抢占与跨硬件差异引入小规模比较。',
        '10–14小时训练应作为执行完自行退出的批处理Job运行。不要把一次交互Pod当作可稳定持续十几小时的训练环境。',
        '原服务器/path/to/project_parent不会自动出现在Nautilus。新持久卷可以挂到相同容器路径，但需要迁移项目、必要权重、数据和checkpoint；不使用其他项目已有卷。',
    ]
    recommendation='可以把Nautilus作为新蒸馏方案的优先试运行平台：先准备100GiB项目持久卷和固定环境镜像，再以1张80GB A100、8CPU、64GiB主机内存做有限的老师/OPD吞吐pilot。当前只是配置建议与只读验证，尚未实际申请。'
    limits=[
        '所有空闲数随时变化；没有预留节点。',
        '未实际创建PVC，未测绑定、读写速度和文件额度。',
        '未启动GPU容器，未核实镜像、FlashAttention/vLLM运行、模型下载速度或真实OPD峰值。',
        '保留原服务器代码和结果；当前未迁移、未加载模型、未启动训练。',
    ]
    summary={'snapshot_time':available['checked_at'],'namespace':'ecepxie','main_recommendation':recommendation,'facts':facts,'candidate_rows':candidate_rows,'candidate_count_conservative':len(candidates),'dry_run_100Gi_PVC_passed':True,'dry_run_A10080_Job_passed':True,'dry_run_does_not_test_child_pod_or_scheduling':True,'no_created_resources_verified':True,'cluster_pod_crosscheck_forbidden':bool(cross.get('error')),'limits':limits,'sources':sources,'input_sha256':{n:hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in ['summary.json','deep_status.json','admission_dryrun.json','available_gpu_snapshot.json','candidate_crosscheck.json']}}
    (HERE/'deep_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    text=['Nautilus ecepxie：深入资源核查',available['checked_at'],'',recommendation,'','已核实',*facts,'','候选节点快照（非预留）','节点 | GPU | GPU Free | CPU Free | RAM Free | Region']+[' | '.join(map(str,r)) for r in candidate_rows]+['','未核实',*limits,'','参考']+[k+': '+v for k,v in sources.items()]
    (HERE/'DEEP_REPORT.txt').write_text('\n'.join(text)+'\n')
    css='body{font:15px/1.75 system-ui;margin:0;background:#f4f6fa;color:#203047}main{max-width:1050px;margin:auto;padding:40px 22px}section{background:white;border:1px solid #dce3ed;border-radius:12px;padding:24px;margin:20px 0}h1{font-size:28px}h2{font-size:20px}p{margin:11px 0}table{width:100%;border-collapse:collapse;font-size:14px}td,th{text-align:left;vertical-align:top;padding:10px;border-bottom:1px solid #e3e8ef}.scroll{overflow-x:auto}a{color:#3154a3}.muted{color:#61728a}'
    page='<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Nautilus深入核查</title><style>'+css+'</style></head><body><main><h1>Nautilus：蒸馏训练资源核查</h1><p class="muted">ecepxie · '+escape(available['checked_at'])+' · 未创建资源</p><section><h2>建议</h2><p>'+escape(recommendation)+'</p></section><section><h2>已核实</h2>'+''.join('<p>'+escape(s)+'</p>' for s in facts)+'</section><section><h2>候选快照</h2>'+table(['节点','GPU','GPU Free','CPU Free','系统RAM Free','Region'],candidate_rows)+'<p>Free是官方仪表盘的资源分配快照，不是作业已调度或实际运行速度保证。</p></section><section><h2>仍需实际pilot验证</h2>'+''.join('<p>'+escape(s)+'</p>' for s in limits)+'</section><section><h2>证据与来源</h2>'+''.join('<p><a href="'+escape(v)+'">'+escape(k)+'</a></p>' for k,v in sources.items())+'<p><a href="deep_summary.json">结构化核查结论</a> · <a href="admission_dryrun.json">Dry-run记录</a> · <a href="DEEP_REPORT.txt">文本报告</a></p></section></main></body></html>'
    (HERE/'review.html').write_text(page)
    print(json.dumps({'report':str(HERE/'review.html'),'nominal_candidates':len(candidates),'dry_run_passed':True,'new_resources_created':False},ensure_ascii=False))

if __name__=='__main__':main()
