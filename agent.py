"""
病理诊断 Agent 执行逻辑 - 基于 LangGraph 的图编排
"""

from typing import TypedDict, List, Dict, Annotated, Literal, Optional
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
import operator
import numpy as np
import logging

# 导入模型管理模块
from model_registry import ModelRegistry
from utils import save_rois

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


# ============= 1. 全局状态定义 =============

class PathologyState(TypedDict):
    """全局状态 Schema"""
    wsi_path: str                                       # WSI 切片路径
    roi_queue: Annotated[List[Dict], operator.add]     # ROI 队列
    observations: Annotated[List[Dict], operator.add]  # MLLM 形态学描述
    reflection_log: Annotated[List[str], operator.add] # 反思日志
    diagnostics: Dict                                   # 诊断结果（来自数据库）
    current_iteration: int                              # 当前迭代次数
    max_iterations: int                                 # 最大迭代限制
    final_report: str                                   # 病理报告
    slide_id: str                                       # 切片编号


# ============= 2. WSI 工具类 =============
from openslide import OpenSlide




class WSIHandler:
    """WSI 读取和坐标管理"""

    def __init__(self, wsi_path: str):
        self.wsi_path = wsi_path
        try:
            self.wsi = OpenSlide(wsi_path)
            self.level_count = self.wsi.level_count
            self.level_dimensions = self.wsi.level_dimensions
            self.level_downsamples = self.wsi.level_downsamples
        except Exception as e:
            logger.error(f"❌ 无法打开 WSI: {e}")
            self.wsi = None

    def extract_roi_patch(self, center_x, center_y, patch_size) -> np.ndarray:
        """从 WSI 提取指定倍率的 ROI patch"""
        # 将中心坐标转换为 level 0 坐标（OpenSlide 要求）
        thumbnail = self.wsi.get_thumbnail((2048, 2048))
        img_array = np.array(thumbnail)
        arr1 = img_array[:]
        width, height, channel = arr1.shape
        original_width, original_hight = self.level_dimensions[0]
        top_left_x = center_x / width * original_width - patch_size // 2
        top_left_y = center_y / width * original_hight - patch_size // 2

        # 读取区域
        region = self.wsi.read_region(
            (int(top_left_x), int(top_left_y)),
            0,
            (int(patch_size), int(patch_size))
        )

        # 转换为 RGB（OpenSlide 返回 RGBA）
        region_rgb = region.convert('RGB')
        return np.array(region_rgb)


    def get_thumbnail(self, size=(2048, 2048)) -> np.ndarray:
        """获取缩略图用于导航"""
        if self.wsi:
            thumbnail = self.wsi.get_thumbnail(size)
            thumbnail_rgb = thumbnail.convert("RGB")
            thumbnail_rgb.save('./roi_region/thumbnail.png', "PNG")
            return np.array(thumbnail)

    def close(self):
        """释放 WSI 文件句柄"""
        if self.wsi:
            self.wsi.close()


# ============= 3. LangGraph 节点定义 =============

class PathologyAgent:
    """病理诊断 Agent 主类"""

    def __init__(self, model_registry: ModelRegistry):
        self.models = model_registry

    # ----------- 节点 1: Navigator -----------
    def navigator_node(self, state: PathologyState) -> PathologyState:
        """导航节点：ROI 检测"""
        logger.info(f"\n{'='*70}")
        logger.info(f"🔍 [Navigator] 第 {state.get('current_iteration', 0) + 1} 轮导航")
        logger.info(f"{'='*70}")

        # 只在第一次迭代执行检测
        if state.get("current_iteration", 0) == 0:
            wsi = WSIHandler(state['wsi_path'])
            thumbnail = wsi.get_thumbnail()

            logger.info(f"缩略图尺寸是 {thumbnail.shape}")



            # 调用统一 API：感知导航器 (Gemini 3 Pro)
            rois = self.models.detect_rois(thumbnail)

            # 转换为标准格式
            roi_queue = [
                {
                    "coord": (roi["center_x"], roi["center_y"]),
                    "bbox": roi["bbox"],
                    "mag": 20.0,
                    "confidence": roi["confidence"],
                    "status": "pending",
                    "roi_type": roi["class"]
                }
                for roi in rois
            ]

            # 保存roi区域的缩略图
            save_rois(thumbnail, rois)

            wsi.close()

            logger.info(f"✅ 检测到 {len(roi_queue)} 个候选 ROI")

            return {
                "roi_queue": roi_queue,
                "current_iteration": 1
            }
        else:
            # 后续迭代只增加计数
            return {"current_iteration": state.get("current_iteration", 0) + 1}

    # ----------- 节点 2: Sampler -----------
    def sampler_node(self, state: PathologyState) -> PathologyState:
        """采样节点：提取高倍率 Patch"""
        logger.info(f"\n📸 [Sampler] 采样高倍率 Patch...")

        pending = [r for r in state["roi_queue"] if r["status"] == "pending"]
        if not pending:
            logger.warning("⚠️ 队列中无待处理 ROI")
            return {}

        roi = pending[0]
        logger.info(f"   处理 ROI: {roi['coord']} (类型: {roi.get('roi_type', 'unknown')})")

        wsi = WSIHandler(state['wsi_path'])

        # 提取 patch
        # todo : patches支持遍历
        patch = wsi.extract_roi_patch(roi["coord"][0], roi["coord"][1], patch_size=672)

        wsi.close()

        # 更新状态
        updated_queue = state["roi_queue"].copy()
        for r in updated_queue:
            if r["coord"] == roi["coord"] and r["status"] == "pending":
                r["status"] = "sampled"
                r["patch"] = patch
                break

        return {"roi_queue": updated_queue}

    # ----------- 节点 3: Describer -----------
    def describer_node(self, state: PathologyState) -> PathologyState:
        """描述节点：MLLM 形态学分析"""
        logger.info(f"\n🔬 [Describer] 生成形态学描述...")

        sampled = [r for r in state["roi_queue"] if r["status"] == "sampled"]
        if not sampled:
            logger.warning("⚠️ 无已采样的 ROI")
            return {}

        roi = sampled[-1]  # 取最新采样的
        patch = roi.get("patch")

        # 调用统一 API：语义解析员 (Gemini 3 Pro)
        description = self.models.describe_patch(patch)

        observation = {
            "roi_coord": roi["coord"],
            "roi_type": roi.get("roi_type"),
            "description": description,
            "timestamp": state.get("current_iteration")
        }
        logger.info(f"   形态学描述: {description.get('completeness_score', 0):.2f}")

        logger.info(f"   完整度评分: {description.get('completeness_score', 0):.2f}")

        return {"observations": [observation]}

    # ----------- 节点 4: Reflector -----------
    def reflector_node(self, state: PathologyState) -> PathologyState:
        """反思节点：质量检查"""
        logger.info(f"\n🤔 [Reflector] 审查描述质量...")

        if not state["observations"]:
            return {"reflection_log": ["ERROR: 无观察结果"]}

        latest_obs = state["observations"][-1]

        # 调用统一 API：审核审查员 (Baichuan)
        reflection = self.models.reflect_quality(
            latest_obs["description"],
            goal="subtype+invasion"
        )

        logger.info(f"   质量评分: {reflection.get('quality_score', 0):.2f}")
        logger.info(f"   决策: {reflection.get('action', 'UNKNOWN')}")

        if reflection.get("action") == "RE-SCAN":
            # 触发重采样
            last_roi_coord = latest_obs["roi_coord"]

            logger.warning(f"   ⚠️ {reflection.get('suggestions', '')}")

            return {
                "reflection_log": [f"⚠️ {reflection.get('suggestions', '')}"],
                "roi_queue": [{
                    "coord": last_roi_coord,
                    "mag": 40.0,
                    "status": "pending",
                    "reason": "reflection_rescan"
                }]
            }

        return {"reflection_log": [f"✓ {reflection.get('suggestions', '')}"]}

    # ----------- 节点 5: Diagnosis Query (替代 PFM + Specialist) -----------
    def diagnosis_query_node(self, state: PathologyState) -> PathologyState:
        """诊断查询节点：从离线数据库获取诊断结果"""
        logger.info(f"\n🗄️ [DiagnosisDB] 查询离线诊断结果...")

        if not state["observations"]:
            logger.warning("⚠️ 无观察结果")
            return {}

        latest_obs = state["observations"][-1]
        roi_coord = latest_obs["roi_coord"]

        # 调用数据库查询
        diagnosis_result = self.models.query_diagnosis(
            slide_id=state["slide_id"],
            roi_coord=roi_coord
        )

        diagnostics = {
            "subtype": diagnosis_result["subtype"],
            "subtype_confidence": diagnosis_result["subtype_confidence"],
            "invasion_layer": diagnosis_result["invasion_layer"],
            "depth_mm": diagnosis_result["depth_mm"],
            "invasion_confidence": diagnosis_result["invasion_confidence"],
            "model_version": diagnosis_result.get("model_version", "unknown")
        }

        logger.info(f"   📌 模型版本: {diagnostics['model_version']}")

        # 标记当前 ROI 完成
        updated_queue = state["roi_queue"].copy()
        for r in updated_queue:
            if r["status"] == "sampled":
                r["status"] = "diagnosed"
                break

        return {
            "diagnostics": diagnostics,
            "roi_queue": updated_queue
        }

    # ----------- 节点 6: Report Generator -----------
    def report_generator_node(self, state: PathologyState) -> PathologyState:
        """报告生成节点"""
        logger.info(f"\n📄 [Report Generator] 生成最终报告...")

        # 调用统一 API：报告生成器 (Baichuan)
        report = self.models.generate_report(
            state["observations"],
            state["diagnostics"],
            slide_id=state.get("slide_id", "UNKNOWN")
        )

        logger.info("   ✅ 报告生成完成")

        return {"final_report": report}


# ============= 4. 路由逻辑 =============

def should_process_roi(state: PathologyState) -> Literal["sampler", "report"]:
    """Navigator 后的路由"""
    pending = [r for r in state["roi_queue"] if r["status"] == "pending"]

    if pending:
        logger.info(f"📋 发现 {len(pending)} 个待处理 ROI，进入采样")
        return "sampler"
    else:
        logger.info("✅ 队列已空，直接生成报告")
        return "report"


def should_continue_reflection(state: PathologyState) -> Literal["sampler", "diagnosis_query"]:
    """Reflector 后的路由"""
    if state["reflection_log"] and "⚠️" in state["reflection_log"][-1]:
        logger.info("⚠️ 描述质量不足，重新采样")
        return "sampler"

    logger.info("✓ 描述合格，查询诊断数据库")
    return "diagnosis_query"


def should_iterate(state: PathologyState) -> Literal["navigator", "report"]:
    """Specialist 后的路由"""
    # 检查 1: 是否超过最大迭代次数
    if state["current_iteration"] >= state.get("max_iterations", 3):
        logger.info("⚠️ 达到最大迭代次数，生成报告")
        return "report"

    # 检查 2: 是否还有未处理的 ROI
    pending = [r for r in state["roi_queue"] if r["status"] == "pending"]

    if not pending:
        logger.info("✅ 所有 ROI 已处理完成，生成报告")
        return "report"

    logger.info(f"🔄 还有 {len(pending)} 个 ROI 待处理，继续迭代")
    return "navigator"


# ============= 5. 构建图 =============

def build_pathology_graph(model_registry: ModelRegistry) -> StateGraph:
    """构建完整的诊断图"""

    # 创建 Agent 实例
    agent = PathologyAgent(model_registry)

    # 构建图
    workflow = StateGraph(PathologyState)

    # 添加节点
    workflow.add_node("navigator", agent.navigator_node)
    workflow.add_node("sampler", agent.sampler_node)
    workflow.add_node("describer", agent.describer_node)
    workflow.add_node("reflector", agent.reflector_node)
    workflow.add_node("diagnosis_query", agent.diagnosis_query_node)  # 替代 pfm_extraction + specialist
    workflow.add_node("report", agent.report_generator_node)

    # 定义边
    workflow.add_conditional_edges(
        "navigator",
        should_process_roi,
        {
            "sampler": "sampler",
            "report": "report"
        }
    )

    workflow.add_edge("sampler", "describer")
    workflow.add_edge("describer", "reflector")

    workflow.add_conditional_edges(
        "reflector",
        should_continue_reflection,
        {
            "sampler": "sampler",
            "diagnosis_query": "diagnosis_query"  # 直接查询诊断数据库
        }
    )

    workflow.add_conditional_edges(
        "diagnosis_query",
        should_iterate,
        {
            "navigator": "navigator",
            "report": "report"
        }
    )

    workflow.add_edge("report", END)

    # 设置入口
    workflow.set_entry_point("navigator")

    # 编译图
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)


# ============= 6. 主执行入口 =============

def run_pathology_diagnosis(
    wsi_path: str,
    slide_id: str = "SLIDE-001",
    max_iterations: int = 2,
    model_config: Optional[Dict] = None
) -> Dict:
    """
    执行病理诊断流程

    Args:
        wsi_path: WSI 文件路径
        slide_id: 切片编号
        max_iterations: 最大迭代次数
        model_config: 模型配置字典

    Returns:
        Dict: 包含最终状态的字典
    """

    # 1. 初始化模型注册中心
    logger.info("\n" + "="*70)
    logger.info("🚀 初始化病理诊断系统")
    logger.info("="*70)

    model_registry = ModelRegistry(config=model_config)
    model_registry.load_all()

    # 2. 构建图
    graph = build_pathology_graph(model_registry)

    # 3. 初始化状态
    initial_state = {
        "wsi_path": "./data/slide/008682e22a74ac4a85b3b3628ef3b775.svs",
        "slide_id": "008682",
        "roi_queue": [],
        "observations": [],
        "reflection_log": [],
        "diagnostics": {},
        "current_iteration": 0,
        "max_iterations": max_iterations,
        "final_report": ""
    }

    # 4. 执行图
    logger.info("\n" + "="*70)
    logger.info("🏥 开始病理诊断流程")
    logger.info("="*70)

    config = {"configurable": {"thread_id": f"diagnosis_{slide_id}"}}

    try:
        final_state = graph.invoke(initial_state, config)

        # 5. 输出结果
        logger.info("\n" + "="*70)
        logger.info("📊 诊断完成")
        logger.info("="*70)
        logger.info(f"\n{final_state['final_report']}\n")
        logger.info("="*70)

        return final_state

    except Exception as e:
        logger.error(f"\n❌ 诊断流程出错: {e}\n")
        raise


# ============= 7. 命令行接口 =============

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="病理诊断 Agent V11")
    parser.add_argument("--wsi", type=str, required=False, help="WSI 文件路径")
    parser.add_argument("--slide-id", type=str, default="SLIDE-001", help="切片编号")
    parser.add_argument("--max-iter", type=int, default=25, help="最大迭代次数")
    parser.add_argument("--gemini-key", type=str, default=None,
                       help="Gemini API Key")
    parser.add_argument("--baichuan-key", type=str, default=None,
                       help="Baichuan API Key")
    parser.add_argument("--db-type", type=str, default="sqlite",
                       choices=["sqlite", "mongodb", "redis"],
                       help="诊断数据库类型")
    parser.add_argument("--db-path", type=str, default="pathology_diagnosis.db",
                       help="数据库路径（SQLite）或主机地址")

    args = parser.parse_args()

    # 构建配置
    config = {
        "api": {
            "gemini": {
                "api_key": args.gemini_key,
                "model": "gemini-3-pro-preview"
            },
            "baichuan": {
                "api_key": args.baichuan_ke,
                "model": "Baichuan-M3",
                "api_base": "https://api.baichuan-ai.com/v1"
            }
        },
        "database": {
            "type": args.db_type,
            "path": args.db_path if args.db_type == "sqlite" else None,
            "host": args.db_path if args.db_type in ["mongodb", "redis"] else None
        }
    }

    # 执行诊断
    run_pathology_diagnosis(
        wsi_path=args.wsi,
        slide_id=args.slide_id,
        max_iterations=args.max_iter,
        model_config=config
    )