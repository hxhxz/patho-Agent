"""
model_registry.py - 简化版
统一 API 调用接口 + 离线诊断数据库
"""

import numpy as np
from typing import Dict, List, Optional, Any
import logging
import json
import re
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

os.environ['HTTP_PROXY'] = 'http://127.0.0.1:56054'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:56054'


# ============= 统一 API 调用类 =============

class UnifiedModelAPI:
    """
    统一模型 API 调用接口
    支持 Gemini 和 Baichuan 两种后端
    """

    def __init__(self, config: Dict):
        """
        Args:
            config: {
                "gemini": {
                    "api_key": "...",
                    "model": "gemini-3-pro-vision"
                },
                "baichuan": {
                    "api_key": "...",
                    "model": "Baichuan4",
                    "api_base": "https://api.baichuan-ai.com/v1"
                }
            }
        """
        self.config = config
        self.gemini_client = None
        self.baichuan_session = None

        # 模块到后端的映射
        self.module_backend_map = {
            "locator": "gemini",      # 感知导航器 -> Gemini 3 Pro
            "describer": "gemini",    # 语义解析员 -> Gemini 3 Pro
            "reflector": "baichuan",  # 审核审查员 -> Baichuan
            "reporter": "baichuan"    # 报告生成器 -> Baichuan
        }

    def load(self):
        """初始化所有后端客户端"""
        logger.info("🔧 初始化统一 API 调用接口...")

        # 初始化 Gemini
        if "gemini" in self.config:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.config["gemini"]["api_key"])
                self.gemini_client = genai.GenerativeModel(
                    self.config["gemini"].get("model", "gemini-3-pro-preview")
                )
                logger.info("  ✅ Gemini 3 Pro 已加载")
            except Exception as e:
                logger.error(f"  ❌ Gemini 加载失败: {e}")

        # 初始化 Baichuan
        if "baichuan" in self.config:
            try:
                # TODO: 取消注释以使用真实 Baichuan
                import requests
                import json
                self.baichuan_session = requests.Session()
                self.baichuan_session.headers.update({
                    "Authorization": f"Bearer {self.config['baichuan']['api_key']}",
                    "Content-Type": "application/json"
                })
                logger.info("  ✅ Baichuan 3 已加载")
            except Exception as e:
                logger.error(f"  ❌ Baichuan 加载失败: {e}")

        logger.info("✅ 统一 API 接口初始化完成\n")

    def call(self,
             module: str,
             prompt: str,
             image: Optional[np.ndarray] = None,
             **kwargs) -> str:
        """
        统一调用接口

        Args:
            module: 模块名称 ("locator" | "describer" | "reflector" | "reporter")
            prompt: 文本提示
            image: 图像数据（可选，仅 Vision 模型需要）
            **kwargs: 其他参数（如 temperature, max_tokens）

        Returns:
            str: 模型响应文本
        """
        backend = self.module_backend_map.get(module)

        if backend == "gemini":
            return self._call_gemini(prompt, image, **kwargs)
        elif backend == "baichuan":
            return self._call_baichuan(prompt, **kwargs)
        else:
            raise ValueError(f"未知模块: {module}")

    def _call_gemini(self,
                     prompt: str,
                     image: Optional[np.ndarray] = None,
                     **kwargs) -> str:
        """调用 Gemini API"""

        # TODO: 实际 API 调用
        from PIL import Image

        if image is not None:
            img = Image.fromarray(image)
            response = self.gemini_client.generate_content([prompt, img])
        else:
            response = self.gemini_client.generate_content(prompt)

        return response.text

        # Mock 响应
        # logger.info("  🤖 [Gemini] 模拟调用...")
        # if "ROI" in prompt or "检测" in prompt:
        #     return json.dumps({
        #         "rois": [
        #             {"center_x": 5000, "center_y": 8000, "bbox": [4800, 7800, 5200, 8200],
        #              "confidence": 0.92, "class": "tumor_region"},
        #             {"center_x": 12000, "center_y": 6000, "bbox": [11800, 5800, 12200, 6200],
        #              "confidence": 0.87, "class": "dysplastic_area"}
        #         ]
        #     })
        # elif "形态学" in prompt or "描述" in prompt:
        #     return json.dumps({
        #         "细胞特征": "核浆比增高(>1:2)，核分裂象 3-5/HPF，核仁明显",
        #         "组织结构": "腺体融合排列，背靠背模式，局部坏死",
        #         "间质改变": "间质纤维化伴淋巴细胞浸润",
        #         "基底膜": "基底膜局部中断，侵犯粘膜下层",
        #         "completeness_score": 0.95
        #     })
        # else:
        #     return "Gemini mock response"

    def _call_baichuan(self, prompt: str, **kwargs) -> str:
        """调用 Baichuan API"""

        # TODO: 实际 API 调用
        response = self.baichuan_session.post(
            f"https://api.baichuan-ai.com/v1/chat/completions",
            json={
                "model": self.config['baichuan'].get('model', 'Baichuan-M3'),
                "messages": [{"role": "user", "content": prompt}],
                "temperature": kwargs.get("temperature", 0.7),
                "max_tokens": kwargs.get("max_tokens", 32000)
            }
        )
        result = response.json()
        return result["choices"][0]["message"]["content"]

        # Mock 响应
        # logger.info("  🤖 [Baichuan] 模拟调用...")
        # if "审查" in prompt or "质量" in prompt:
        #     return json.dumps({
        #         "quality_score": 0.95,
        #         "missing_fields": [],
        #         "action": "PROCEED",
        #         "suggestions": "描述完整，可进入诊断阶段"
        #     })
        # elif "报告" in prompt:
        #     return """
        #         === 病理诊断报告 ===
        #
        #         【标本信息】
        #         来源组织：胃窦粘膜活检
        #         染色方法：HE 染色
        #
        #         【镜下所见】
        #         腺体融合排列，背靠背模式，局部坏死
        #
        #         【诊断意见】
        #         肿瘤分型：中分化腺癌
        #         浸润深度：肌层 (2.3 mm)
        #
        #         【病理分期建议】
        #         T2 (侵犯肌层)
        #         """
        # else:
        #     return "Baichuan mock response"

    @staticmethod
    def parse_json_response(text: str) -> Dict:
        """从响应文本中提取 JSON"""
        # 尝试提取 JSON 代码块
        json_match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
        if json_match:
            text = json_match.group(1)

        # 尝试解析 JSON
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # 如果不是有效 JSON，返回原文本
            return {"raw_text": text}


# ============= Prompt 模板库 =============

class PromptTemplates:
    """预定义的 Prompt 模板"""

    LOCATOR_PROMPT = """
        你是病理学专家。请分析这张低倍率病理切片缩略图，识别所有可疑的肿瘤区域（ROI）。
        
        对每个 ROI，请以 JSON 格式输出：
        ```json
        {
          "rois": [
            {
              "center_x": 像素坐标,
              "center_y": 像素坐标,
              "bbox": [x1, y1, x2, y2],
              "confidence": 0-1之间的置信度,
              "class": "tumor_region" | "dysplastic_area" | "inflammatory_area"
            }
          ]
        }
        ```
        
        重点关注：细胞密集区、结构紊乱区、异型细胞聚集区。
        只输出 JSON，不要其他说明文字。
        """

    DESCRIBER_PROMPT = """
        你是资深病理专科医生。请分析该病理切片的高倍率图像，必须按以下结构输出：
        
        【细胞特征】：核浆比、核分裂象、核仁状态
        【组织结构】：腺体排列、坏死情况、分化程度
        【间质改变】：纤维化、炎性浸润、血管状态
        【基底膜】：连续性、破坏程度、侵犯深度
        
        要求：
        - 每项至少 15 字详细描述
        - 使用标准病理术语
        - 客观描述，避免诊断性结论
        
        请以 JSON 格式输出：
        ```json
        {
          "细胞特征": "...",
          "组织结构": "...",
          "间质改变": "...",
          "基底膜": "...",
          "completeness_score": 0.0-1.0
        }
        ```
        
        只输出 JSON，不要其他说明文字。
        """

    @staticmethod
    def reflector_prompt(description: Dict, diagnostic_goal: str = "subtype+invasion") -> str:
        return f"""
            你是病理质控专家。请审查以下形态学描述的质量。
            
            诊断目标：{diagnostic_goal}
            
            描述内容：
            {json.dumps(description, ensure_ascii=False, indent=2)}
            
            请评估：
            1. 是否包含所有必需字段（细胞特征、组织结构、间质改变、基底膜）
            2. 每个字段的描述是否足够详细（至少15字）
            3. 是否使用了标准病理术语
            
            以 JSON 格式输出：
            ```json
            {{
              "quality_score": 0-1之间的分数,
              "missing_fields": ["缺失或不完整的字段"],
              "action": "RE-SCAN" 或 "PROCEED",
              "suggestions": "具体改进建议"
            }}
            ```
            
            只输出 JSON，不要其他说明文字。
        """

    @staticmethod
    def reporter_prompt(observations: List[Dict], diagnostics: Dict, slide_id: str) -> str:
        return f"""
            你是病理报告撰写专家。请根据以下信息生成符合临床规范的病理诊断报告。
            
            切片编号：{slide_id}
            
            镜下观察结果：
            {json.dumps(observations, ensure_ascii=False, indent=2)}
            
            诊断结论：
            {json.dumps(diagnostics, ensure_ascii=False, indent=2)}
            
            要求：
            1. 使用规范的病理报告格式
            2. 包含：标本信息、镜下所见、诊断意见、分期建议
            3. 语言专业、准确、简洁
            4. 避免过度解读，客观描述事实
            
            请直接输出完整的病理报告，不要 JSON 格式。
            """


# ============= 离线诊断数据库 (Mock) =============

class DiagnosisDatabase:
    """离线诊断数据库接口（Mock 实现）"""

    def __init__(self, db_config: Optional[Dict] = None):
        self.db_config = db_config or {"type": "mock"}
        self.mock_data = {}  # 模拟数据存储

    def load(self):
        """初始化数据库连接"""
        logger.info("🔧 连接诊断数据库...")

        # 预填充一些 Mock 数据
        self.mock_data = {
            "SLIDE-001_(5000, 8000)": {
                "subtype": "moderately_differentiated_adenocarcinoma",
                "subtype_confidence": 0.89,
                "invasion_layer": "muscularis_propria",
                "depth_mm": 2.3,
                "invasion_confidence": 0.91,
                "model_version": "virchow2_atlas2_v2.1"
            },
            "SLIDE-001_(12000, 6000)": {
                "subtype": "well_differentiated_adenocarcinoma",
                "subtype_confidence": 0.92,
                "invasion_layer": "submucosa",
                "depth_mm": 1.2,
                "invasion_confidence": 0.88,
                "model_version": "virchow2_atlas2_v2.1"
            }
        }

        logger.info(f"  ✅ 数据库已连接 (类型: Mock, 缓存数: {len(self.mock_data)})\n")

    def query(self, slide_id: str, roi_coord: tuple) -> Optional[Dict]:
        """
        查询诊断结果

        Args:
            slide_id: 切片编号
            roi_coord: ROI 坐标 (x, y)

        Returns:
            Dict: 诊断结果或 None
        """
        key = f"{slide_id}_{roi_coord}"
        result = self.mock_data.get(key)

        if result:
            logger.info(f"  ✅ [数据库] 命中缓存: {roi_coord} -> {result['subtype']}")
        else:
            logger.warning(f"  ⚠️ [数据库] 未命中缓存: {key}")
            # 返回默认结果
            result = {
                "subtype": "undetermined",
                "subtype_confidence": 0.0,
                "invasion_layer": "unknown",
                "depth_mm": 0.0,
                "invasion_confidence": 0.0,
                "model_version": "mock_fallback"
            }

        return result

    def batch_query(self, slide_id: str, roi_list: List[tuple]) -> List[Dict]:
        """批量查询"""
        return [self.query(slide_id, coord) for coord in roi_list]


# ============= 模型注册中心（简化版） =============

class ModelRegistry:
    """
    简化版模型注册中心
    统一管理 API 调用和数据库查询
    """

    def __init__(self, config: Dict):
        """
        Args:
            config: {
                "api": {
                    "gemini": {"api_key": "...", "model": "gemini-3-pro-vision"},
                    "baichuan": {"api_key": "...", "model": "Baichuan4"}
                },
                "database": {"type": "mock"}
            }
        """
        self.config = config

        # 初始化组件
        self.api = UnifiedModelAPI(config.get("api", {}))
        self.database = DiagnosisDatabase(config.get("database", {}))
        self.prompts = PromptTemplates()

    def load_all(self):
        """加载所有组件"""
        logger.info("="*70)
        logger.info("🚀 初始化病理诊断系统")
        logger.info("="*70 + "\n")

        self.api.load()
        self.database.load()

        logger.info("="*70)
        logger.info("✅ 系统初始化完成")
        logger.info("="*70 + "\n")

    # ----------- 便捷调用方法 -----------

    def detect_rois(self, thumbnail: np.ndarray) -> List[Dict]:
        """感知导航器：检测 ROI"""
        logger.info("🔍 [Locator] 检测 ROI...")
        response = self.api.call("locator", self.prompts.LOCATOR_PROMPT, image=thumbnail)
        result = self.api.parse_json_response(response)
        return result.get("rois", [])

    def describe_patch(self, patch: np.ndarray) -> Dict:
        """语义解析员：生成形态学描述"""
        logger.info("🔬 [Describer] 生成形态学描述...")
        response = self.api.call("describer", self.prompts.DESCRIBER_PROMPT, image=patch)
        return self.api.parse_json_response(response)

    def reflect_quality(self, description: Dict, goal: str = "subtype+invasion") -> Dict:
        """审核审查员：质量反思"""
        logger.info("🤔 [Reflector] 审查描述质量...")
        prompt = self.prompts.reflector_prompt(description, goal)
        response = self.api.call("reflector", prompt)
        return self.api.parse_json_response(response)

    def generate_report(self, observations: List[Dict], diagnostics: Dict, slide_id: str) -> str:
        """报告生成器：生成病理报告"""
        logger.info("📄 [Reporter] 生成病理报告...")
        prompt = self.prompts.reporter_prompt(observations, diagnostics, slide_id)
        return self.api.call("reporter", prompt)

    def query_diagnosis(self, slide_id: str, roi_coord: tuple) -> Dict:
        """查询离线诊断结果"""
        logger.info(f"🗄️ [Database] 查询诊断结果: {roi_coord}")
        return self.database.query(slide_id, roi_coord)


# ============= 使用示例 =============

if __name__ == "__main__":
    # 配置
    config = {
        "api": {
            "gemini": {
                "api_key": "your-gemini-api-key",
                "model": "gemini-3-pro-vision"
            },
            "baichuan": {
                "api_key": "your-baichuan-api-key",
                "model": "Baichuan4",
                "api_base": "https://api.baichuan-ai.com/v1"
            }
        },
        "database": {
            "type": "mock"
        }
    }

    # 初始化
    registry = ModelRegistry(config)
    registry.load_all()

    # 测试数据
    dummy_thumbnail = np.random.randint(0, 255, (2048, 2048, 3), dtype=np.uint8)
    dummy_patch = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)

    print("\n" + "="*70)
    print("测试各个模块")
    print("="*70 + "\n")

    # 1. ROI 检测
    rois = registry.detect_rois(dummy_thumbnail)
    print(f"✅ 检测到 {len(rois)} 个 ROI\n")

    # 2. 形态学描述
    description = registry.describe_patch(dummy_patch)
    print(f"✅ 描述完整度: {description.get('completeness_score', 0):.2f}\n")

    # 3. 质量反思
    reflection = registry.reflect_quality(description)
    print(f"✅ 质量评分: {reflection.get('quality_score', 0):.2f}\n")

    # 4. 查询诊断
    diagnosis = registry.query_diagnosis("SLIDE-001", (5000, 8000))
    print(f"✅ 诊断: {diagnosis['subtype']}\n")

    # 5. 生成报告
    observations = [{"description": description, "roi_type": "tumor_region"}]
    report = registry.generate_report(observations, diagnosis, "SLIDE-001")
    print(f"✅ 报告已生成\n")
    print(report)