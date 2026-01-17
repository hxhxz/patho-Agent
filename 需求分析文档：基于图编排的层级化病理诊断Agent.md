# 需求分析文档：基于图编排的层级化病理诊断Agent
## 1. 项目背景与目标
设计并实现一个能够模拟病理医生读片逻辑的智能 Agent。该 Agent 需处理高分辨率病理切片（WSI），通过“全局扫描 - 局部聚焦 - 语义描述 - 反思校准 - 定量诊断”的闭环流程，完成对 ROI（感兴趣区域）的亚型分类及浸润深度（肌层、浆膜层）的自动化评估。
## 2. 核心架构范式
编排框架： LangGraph (基于有向有环图的状态机)。
设计范式： Plan-and-Execute 结合 Reflection (反思机制)。
部署环境： 私域 AI Infra，涉及 NPU 算子加速（如国产 NPU 上的 PyTorch-NPU/ACL 推理）。
## 3. 系统逻辑流程 (Workflow)
### A. 导航与采样 (Navigator & Sampler)
1. 全局扫描： 在低分辨率（如 $2.5 \times$ 或 $5 \times$）下识别潜在 ROI（如肿瘤细胞密集区）。
2. 坐标映射： 维护一套坐标转换逻辑，将低倍率下的像素坐标映射为全片（WSI）物理坐标。
3. 多尺度采样： 针对 ROI，提取高倍率（$20 \times$ 或 $40 \times$）截图，并进行小分辨率重采样以适配 MLLM 的输入限制。
### B. 语义解析与反思 (Describer & Reflector)
1. MLLM 描述： 调用多模态大模型（如 GPT-4o, Qwen-VL-Max）提取形态学特征（如核浆比、间质反应、基底膜状态）。
2. 反思节点 (Reflection)： 
* 审查描述是否包含关键病理要素。判断是否存在视觉模糊或信息不足。
* 若不合格，回溯至采样节点调整倍率或对比采样。
### C. 下游专家诊断 (Specialist)
1. 任务分发： 根据 MLLM 描述，触发 NPU 加速的专用模型。
2. 模型清单：
* 亚型分类模型： 如基于 ViT 或 CLAM 的子类判定。
* 浸润深度模型： 如基于 HoVer-Net 的细胞分割及物理深度测量。
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
## 6. 交付任务 (To-be-executed by Claude)
1. 代码框架： 基于 LangGraph 定义上述所有节点（Nodes）和边（Edges）的逻辑结构。
2. 坐标转换逻辑： 编写一个用于处理 WSI 不同倍率切片转换的工具类（Utility）。
3. 反思逻辑实现： 设计一套可靠的规则判断函数，用于在 Reflector 节点中决定路径跳转。
4. 接口封装： 预留调用本地 NPU 推理引擎（C++ API 或 PyTorch 接口）的 Python 包装器。