import os

class SimpleVLMEngine:
    """纯Python VLM模拟器 - 基于文件名和规则"""
    
    def analyze_device_photo(self, image_path):
        """分析设备照片"""
        filename = os.path.basename(image_path).lower()
        
        # 故障规则库
        fault_rules = {
            'overheat': ['热', '高温', '发烫', '烧', 'overheat'],
            'vibration': ['振', '抖', '晃动', 'vibration'],
            'oil_leak': ['漏油', '渗油', 'oil', 'leak'],
            'smoke': ['烟', '烧焦', 'smoke', 'burn'],
            'broken': ['裂', '断', '破损', 'broken']
        }
        
        detected_faults = []
        for fault, keywords in fault_rules.items():
            if any(kw in filename for kw in keywords):
                detected_faults.append(fault)
        
        if not detected_faults:
            # 尝试从文件名提取设备类型
            device_keywords = ['变频器', '电机', 'PLC', '传感器', 'inverter', 'motor']
            device = next((d for d in device_keywords if d.lower() in filename), '未知设备')
            detected_faults = ['normal']
            description = f"图片显示{device}外观无明显异常"
        else:
            description = f"检测到可能故障: {', '.join(detected_faults)}"
        
        return {
            'fault_type': detected_faults,
            'description': description,
            'confidence': 0.75,
            'suggestion': '建议现场进一步检查设备运行参数'
        }

def get_vlm_engine():
    return SimpleVLMEngine()
