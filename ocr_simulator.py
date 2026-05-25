import re
import os
from PIL import Image

class SimpleOCREngine:
    """纯Python OCR模拟器 - 从文件名和图像基本信息提取特征"""
    
    def __init__(self):
        print("初始化模拟OCR引擎...")
    
    def extract_text(self, image_path):
        """从文件名和图像信息模拟OCR"""
        if not os.path.exists(image_path):
            return "文件不存在"
        
        filename = os.path.basename(image_path).lower()
        
        # 从文件名提取故障关键词
        fault_keywords = {
            'overcurrent': ['过流', '过电流', 'overcurrent', 'f0001'],
            'overheat': ['过热', '发热', '发烫', 'overheat', 'temperature'],
            'vibration': ['振动', '抖动', 'vibration', 'shake'],
            'noresponse': ['无响应', '离线', 'offline', 'noreply'],
            'stall': ['堵转', '卡死', 'stall', 'blocked']
        }
        
        detected = []
        for fault_type, keywords in fault_keywords.items():
            if any(kw in filename for kw in keywords):
                detected.append(fault_type)
        
        # 构建模拟的OCR文本
        if detected:
            ocr_text = f"检测到故障特征: {', '.join(detected)}。建议检查设备运行状态。"
        else:
            ocr_text = "未检测到明显文字信息。请确认图片是否包含故障代码或参数。"
        
        print(f"📖 模拟OCR: {ocr_text[:100]}...")
        return ocr_text
    
    def extract_parameters(self, text):
        """从文本中提取参数"""
        patterns = {
            'voltage': r'电压[：:]\s*(\d+\.?\d*)',
            'current': r'电流[：:]\s*(\d+\.?\d*)',
            'fault_code': r'[Ff](\d{4})'
        }
        
        params = {}
        for key, pattern in patterns.items():
            match = re.search(pattern, text)
            if match:
                params[key] = match.group(1)
        
        return params

# 使用模拟版本
OCREngine = SimpleOCREngine
