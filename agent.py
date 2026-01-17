"""
基于 LangGraph 的层级化病理诊断 Agent
核心图编排逻辑 - Plan-Execute-Reflect 范式
"""

from typing import TypedDict, List, Dict, Annotated, Literal
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
import operator


# ============= 1. 全局状态定义 =============
class PathologyState(TypedDict):
    """全局状态 Schema"""
    wsi_path: str  # WSI 切片路径
    roi_queue: Annotated[List[Dict], operator.add]  # ROI 队列 [{coord, mag, status}]
    observations: Annotated[List[Dict], operator.add]  # MLLM 形态学描述
    reflection_log: Annotated[List[str], operator.add]  # 反思日志
    diagnostics: Dict  # 下游模型结果 {subtype, invasion_depth}
    current_iteration: int  # 当前迭代次数
    max_iterations: int  # 最大迭代限制
    final_report: str  # 病理报告


# ============= 2. WSI 库集成 =============
import openslide
from openslide import OpenSlide
import numpy as np
from PIL import Image

class WSICoordinateMapper:
    """处理多尺度坐标映射 + WSI 实际读取"""

    def __init__(self, wsi_path: str):
        self.wsi = openslide.OpenSlide(wsi_path)
        self.level_count = self.wsi.level_count
        self.level_dimensions = self.wsi.level_dimensions
        self.level_downsamples = self.wsi.level_downsamples

        # 获取物理倍率（MPP - microns per pixel）
        try:
            self.mpp_x = float(self.wsi.properties.get(openslide.PROPERTY_NAME_MPP_X, 0.25))
            self.mpp_y = float(self.wsi.properties.get(openslide.PROPERTY_NAME_MPP_Y, 0.25))
        except:
            self.mpp_x = self.mpp_y = 0.25  # 默认值

    def get_best_level_for_mag(self, target_mag: float) -> int:
        """根据目标倍率选择最佳 level"""
        # 假设 level 0 是 40x 或通过 objective-power 属性获取
        base_mag = float(self.wsi.properties.get(openslide.PROPERTY_NAME_OBJECTIVE_POWER, 40))
        target_downsample = base_mag / target_mag

        # 找到最接近的 level
        best_level = 0
        min_diff = float('inf')
        for i, ds in enumerate(self.level_downsamples):
            diff = abs(ds - target_downsample)
            if diff < min_diff:
                min_diff = diff
                best_level = i
        return best_level

    def low_to_high_mag(self, x: int, y: int,
                        from_level: int, to_level: int) -> tuple:
        """不同 level 间的坐标转换"""
        scale = self.level_downsamples[from_level] / self.level_downsamples[to_level]
        return int(x * scale), int(y * scale)

    def extract_roi_patch(self, center_x: int, center_y: int,
                          mag: float, patch_size: int = 512) -> np.ndarray:
        """从 WSI 提取指定倍率的 ROI patch"""
        level = self.get_best_level_for_mag(mag)

        # 将中心坐标转换为 level 0 坐标（OpenSlide 要求）
        level0_x, level0_y = center_x, center_y

        # 计算左上角坐标（patch 以中心为准）
        half_size = patch_size // 2
        downsample = self.level_downsamples[level]
        top_left_x = int(level0_x - half_size * downsample)
        top_left_y = int(level0_y - half_size * downsample)

        # 读取区域
        region = self.wsi.read_region(
            (top_left_x, top_left_y),
            level,
            (patch_size, patch_size)
        )

        # 转换为 RGB（OpenSlide 返回 RGBA）
        region_rgb = region.convert('RGB')
        return np.array(region_rgb)

    def get_thumbnail(self, target_size: tuple = (1024, 1024)) -> np.ndarray:
        """获取全局缩略图用于导航"""
        thumbnail = self.wsi.get_thumbnail(target_size)
        return np.array(thumbnail)

    def close(self):
        """释放 WSI 文件句柄"""
        self.wsi.close()


# ============= 3. 节点函数定义 =============

def navigator_node(state: PathologyState) -> PathologyState:
    """导航节点：全局扫描识别 ROI"""
    print(f"🔍 [Navigator] 扫描 WSI: {state['wsi_path']}")

    # 模拟低倍率扫描逻辑
    detected_rois = [
        {"coord": (1024, 2048), "mag": 5.0, "confidence": 0.92, "status": "pending"},
        {"coord": (3072, 1536), "mag": 5.0, "confidence": 0.87, "status": "pending"}
    ]

    return {
        "roi_queue": detected_rois,
        "current_iteration": state.get("current_iteration", 0) + 1
    }


def sampler_node(state: PathologyState) -> PathologyState:
    """采样节点：多尺度截图采样"""
    print(f"📸 [Sampler] 处理 ROI 队列...")

    pending_rois = [r for r in state["roi_queue"] if r["status"] == "pending"]

    if not pending_rois:
        return {}

    # 取队首 ROI 进行高倍率采样
    roi = pending_rois[0]
    mapper = WSICoordinateMapper((10000, 10000))

    # 转换到 20x 倍率
    high_x, high_y = mapper.low_to_high_mag(
        roi["coord"][0], roi["coord"][1],
        from_mag=5.0, to_mag=20.0
    )

    # 模拟采样
    patch_path = mapper.extract_roi_patch(None, high_x, high_y, mag=20.0)

    # 更新 ROI 状态
    updated_queue = state["roi_queue"].copy()
    for r in updated_queue:
        if r["coord"] == roi["coord"]:
            r["status"] = "sampled"
            r["patch_path"] = patch_path
            break

    return {"roi_queue": updated_queue}


def describer_node(state: PathologyState) -> PathologyState:
    """MLLM 描述节点：提取形态学特征"""
    print(f"🔬 [Describer] 调用 MLLM 分析形态学...")

    # 构造结构化 Prompt
    prompt = """
    你是资深病理专科医生。请分析该病理切片，必须包含：
    1. [细胞特征]：核浆比、核分裂象
    2. [组织结构]：腺体排列、坏死情况
    3. [间质改变]：纤维化、炎性浸润
    4. [基底膜]：连续性、破坏程度
    """

    # 模拟 MLLM 响应
    observation = {
        "patch_id": "roi_0_mag20",
        "description": {
            "细胞特征": "核浆比增高(>1:2)，核分裂象 3-5/HPF",
            "组织结构": "腺体融合，背靠背排列，局部坏死",
            "间质改变": "间质纤维化伴淋巴细胞浸润",
            "基底膜": "基底膜局部中断"
        },
        "completeness_score": 0.95
    }

    return {"observations": [observation]}


def reflector_node(state: PathologyState) -> PathologyState:
    """反思节点：质量检查与反馈"""
    print(f"🤔 [Reflector] 审查描述质量...")

    if not state["observations"]:
        return {"reflection_log": ["ERROR: 无有效观察结果"]}

    latest_obs = state["observations"][-1]
    desc = latest_obs["description"]

    # 反思规则
    missing_fields = []
    required_fields = ["细胞特征", "组织结构", "间质改变", "基底膜"]

    for field in required_fields:
        if field not in desc or len(desc[field]) < 10:
            missing_fields.append(field)

    if missing_fields:
        feedback = f"描述不完整，缺失: {', '.join(missing_fields)}"
        return {
            "reflection_log": [feedback],
            "roi_queue": [{"coord": (0, 0), "mag": 40.0, "status": "pending"}]  # 触发重采样
        }

    return {"reflection_log": ["✓ 描述合格"]}


def specialist_node(state: PathologyState) -> PathologyState:
    """专家诊断节点：NPU 加速模型推理"""
    print(f"🧠 [Specialist] 调用下游专家模型...")

    # 模拟 NPU 推理调用
    def call_npu_model(model_name: str, input_data: dict):
        """预留 NPU 推理接口"""
        # 实际调用: npu_engine.infer(model_name, input_data)
        return {"result": f"{model_name}_output"}

    # 亚型分类
    subtype_result = call_npu_model("subtype_classifier", {
        "patch": state["observations"][-1]["patch_id"]
    })

    # 浸润深度评估
    invasion_result = call_npu_model("invasion_depth_model", {
        "features": state["observations"][-1]["description"]
    })

    diagnostics = {
        "subtype": "moderately_differentiated_adenocarcinoma",
        "invasion_depth": "muscularis_propria",
        "confidence": 0.89
    }

    return {"diagnostics": diagnostics}


def report_generator_node(state: PathologyState) -> PathologyState:
    """报告生成节点"""
    print(f"📄 [Reporter] 生成病理报告...")

    report = f"""
=== 病理诊断报告 ===
切片编号: {state['wsi_path']}
诊断结论:
  - 肿瘤亚型: {state['diagnostics'].get('subtype', 'N/A')}
  - 浸润深度: {state['diagnostics'].get('invasion_depth', 'N/A')}

形态学观察:
{state['observations'][-1]['description'] if state['observations'] else '无'}

质控日志:
{chr(10).join(state['reflection_log'])}
    """

    return {"final_report": report.strip()}


# ============= 4. 路由逻辑 =============

def should_continue_reflection(state: PathologyState) -> Literal["sampler", "specialist"]:
    """反思后的路由决策"""
    if state["reflection_log"] and "不完整" in state["reflection_log"][-1]:
        return "sampler"  # 重新采样
    return "specialist"  # 进入诊断


def should_iterate(state: PathologyState) -> Literal["navigator", "report"]:
    """是否继续迭代"""
    if state["current_iteration"] >= state.get("max_iterations", 3):
        return "report"

    pending = [r for r in state["roi_queue"] if r["status"] == "pending"]
    if pending:
        return "navigator"

    return "report"


# ============= 5. 构建图结构 =============

def build_pathology_graph():
    """构建完整的诊断图"""
    workflow = StateGraph(PathologyState)

    # 添加节点
    workflow.add_node("navigator", navigator_node)
    workflow.add_node("sampler", sampler_node)
    workflow.add_node("describer", describer_node)
    workflow.add_node("reflector", reflector_node)
    workflow.add_node("specialist", specialist_node)
    workflow.add_node("report", report_generator_node)

    # 定义边
    workflow.add_edge("navigator", "sampler")
    workflow.add_edge("sampler", "describer")
    workflow.add_edge("describer", "reflector")

    # 条件边
    workflow.add_conditional_edges(
        "reflector",
        should_continue_reflection,
        {
            "sampler": "sampler",  # 反思失败 -> 重采样
            "specialist": "specialist"  # 反思通过 -> 诊断
        }
    )

    workflow.add_conditional_edges(
        "specialist",
        should_iterate,
        {
            "navigator": "navigator",  # 继续处理下一个 ROI
            "report": "report"  # 生成报告
        }
    )

    workflow.add_edge("report", END)

    # 设置入口
    workflow.set_entry_point("navigator")

    # 编译图
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)

    return app


# ============= 6. 执行示例 =============

if __name__ == "__main__":
    # 初始化状态
    initial_state = {
        "wsi_path": "/data/slides/008682e22a74ac4a85b3b3628ef3b775.svs",
        "roi_queue": [],
        "observations": [],
        "reflection_log": [],
        "diagnostics": {},
        "current_iteration": 0,
        "max_iterations": 2,
        "final_report": ""
    }

    # 构建图
    graph = build_pathology_graph()

    # 执行
    config = {"configurable": {"thread_id": "pathology_001"}}
    final_state = graph.invoke(initial_state, config)

    print("\n" + "=" * 50)
    print(final_state["final_report"])
    print("=" * 50)