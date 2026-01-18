# 需求分析文档：基于图编排的层级化病理诊断Agent
## 1. 项目背景与目标
设计并实现一个能够模拟病理医生读片逻辑的智能 Agent。该 Agent 需处理高分辨率病理切片（WSI），通过“全局扫描 - 局部聚焦 - 语义描述 - 反思校准 - 定量诊断”的闭环流程，完成对 ROI（感兴趣区域）的亚型分类及浸润深度（肌层、浆膜层）的自动化评估。系统通过模拟病理医生由粗及密的读片逻辑，利用多模态大模型（MLLM）解析形态学语义，并协同下游专家模型完成定量诊断任务（如亚型分类与浸润深度测量）。
## 2. 核心架构范式
编排范式： LangGraph (Graph-based State Machine)。支持有环迭代与状态回溯。

运行模式： Plan-and-Execute 结合 Self-Reflection。

计算底座： 私域 AI Infra，涉及 NPU 算子加速（本地推理）与 API/GPU 集群（大模型调度）。
## 3. 系统逻辑流程 (Workflow)
### A. 导航与采样 (Navigator & Sampler)
1. 全局扫描： 在低分辨率（如 $2.5 \times$ 或 $5 \times$）下识别潜在 ROI（如肿瘤细胞密集区）。
2. 坐标映射： 维护一套坐标转换逻辑，将低倍率下的像素坐标映射为全片（WSI）物理坐标。
3. 多尺度采样： 针对 ROI，提取高倍率（$20 \times$ 或 $40 \times$）截图，并进行小分辨率重采样以适配 MLLM 的输入限制。
### B. 多尺度描述迭代 (Describer & Reflector)
1. 高倍采样 (Sampler)： 根据 ROI 坐标抓取 40x 图像，并进行小分辨率（如 1024x1024）重采样。
2. 特征描述 (Describer)： * Prompt： “作为病理医生，描述该 Patch 的核分裂象及间质反应。”
3. 反思校验 (Reflector)： 
* 检查 1： 语义完整性（是否漏掉基底膜描述？）。
* 检查 2： 视觉可辨度（若 MLLM 反馈“模糊”，则触发 40x 以上更高精度采样）。
* 决策： 若不合格，重置采样参数返回 Sampler；若合格，进入下一步。

### C. 下游专家诊断 (Specialist)
1. 任务分发： 根据 MLLM 描述，触发 NPU 加速的专用模型。
2. 模型清单：
* 亚型分类模型： 如基于 ViT 或 CLAM 的子类判定。
* 浸润深度模型 
* 神经侵犯模型
* 脉管侵犯模型
* 核分裂象模型

### D. 报告生成 (Final Node)
汇总所有证据链条，产出包含形态学描述、定量指标及分期建议的报告。
## 4. 技术规范与数据结构
### A. 全局状态定义 (State Schema)
~~~
class PathologyState(TypedDict):
    wsi_path: str                 # 原始切片路径
    roi_queue: List[Dict]         # 待处理 ROI 坐标及倍率
    observations: List[Dict]      # MLLM 产生的形态学描述
    reflection_log: List[str]     # 反思反馈记录
    diagnostics: Dict             # 下游模型产出的定量结论
    current_iteration: int        # 迭代次数控制
    final_report: str             # 最终生成的病理报告
~~~
### B. 关键工程约束
* 多尺度管理： 必须处理跨尺度截图的图像张量转换。
* NPU 异步调用： 下游专家模型的推理需支持非阻塞调用，以适配高性能计算节点。
* 低延迟保障： 在 MLLM 描述阶段，需优化 Prompt 以减少 Token 生成开销。
## 5. 提示词工程 (Prompt Engineering) 要求
1. 角色设定： 资深病理专科医生。
2. 结构化输出： 强制要求描述包含 [细胞特征]、[组织结构]、[间质改变]、[基底膜连续性]。
3. 反思规则： 定义清晰的“描述召回”逻辑（例如：若未提及“间质”，则判定为 False 并触发 Re-scan）。
## 6. 全量模型清单
|角色|模型类别|推荐模型 (2026 栈)|核心功能|
| :--- | :---: | :---: |---: |
|感知导航器 (Navigator)|目标检测/分割|YOLOv11-Med / SAM 2|在低倍率（2.5x/5x）图中识别 ROI（如细胞簇、结构异常区）|
|语义解析员 (Describer)|多模态大模型 (MLLM)| Gemini/ MedGemma 1.5 / Claude 3.5 Sonnet|对高倍（20x/40x）截图进行形态学描述（核浆比、间质、基底膜）|
|审核审查员 (Reflector)|大语言模型 (LLM)|GPT-4o (Reasoning) / Med-PaLM 3|审查描述质量与逻辑一致性，决定是否触发重采样或回溯。|
|下游专家组 (Specialist)|专用小模型 (Expert)| MIL / CellViT++|执行特定任务：亚型分类、细胞核分割、浸润深度。|
|报告生成器 (Aggregator)|生成式 LLM|Llama 3.3 (Fine-tuned) / Baichuan-M3| 汇总各节点数据，产出符合临床规范的结构化诊断报告。|

### 7. 交付任务 (To-be-executed by Claude)
1. 代码框架： 基于 LangGraph 定义上述所有节点（Nodes）和边（Edges）的逻辑结构。
2. 坐标转换逻辑： 编写一个用于处理 WSI 不同倍率切片转换的工具类（Utility）。
3. 反思逻辑实现： 设计一套可靠的规则判断函数，用于在 Reflector 节点中决定路径跳转。
4. 接口封装： 预留调用本地 NPU 推理引擎（C++ API 或 PyTorch 接口）的 Python 包装器。

### 8. 后续开发建议
1. Prompt 库建设： 需针对不同癌种（如胃癌、肺癌）建立结构化的描述 Prompt 模版。
2. 反思阈值设定： 需定义“描述合格”的量化指标，防止 Agent 进入无限采样死循环。
3. 多模态对齐： 验证 MLLM 的文本描述与专家模型的数值结论之间的一致性（Consistency Score）。