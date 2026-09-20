import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from sentence_transformers import SentenceTransformer
import pickle

def load_fault_cases(file_path='data.txt'):
    """使用空行作为分隔符解析data.txt"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 按连续空行分隔（两个换行符）
    blocks = content.strip().split('\n\n')
    
    cases = []
    for block in blocks:
        if not block.strip():
            continue
        
        case = {}
        lines = block.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # 处理中文冒号
            if '：' in line:
                key, value = line.split('：', 1)
                case[key.strip()] = value.strip()
            elif ':' in line:
                key, value = line.split(':', 1)
                case[key.strip()] = value.strip()
        
        # 只添加包含'设备名称'的案例
        if case.get('设备名称'):
            cases.append(case)
    
    return cases

# 加载案例
print("正在加载故障案例...")
cases = load_fault_cases('data.txt')
print(f"✅ 成功加载 {len(cases)} 条案例")

# 显示前5条设备名称验证
print("\n前5条案例预览：")
for i, case in enumerate(cases[:5]):
    print(f"   {i+1}. {case.get('设备名称', '未知')[:40]}")

if len(cases) == 0:
    print("❌ 未找到任何案例，请检查data.txt格式")
    exit()

# 组合成检索文本
texts = []
for case in cases:
    combined = f"{case.get('设备名称', '')} {case.get('故障现象', '')} {case.get('可能原因', '')} {case.get('解决步骤', '')}"
    texts.append(combined)

# 加载嵌入模型
print("\n正在加载嵌入模型...")
model = SentenceTransformer('BAAI/bge-small-zh-v1.5')

# 生成向量
print("正在生成向量...")
embeddings = model.encode(texts, show_progress_bar=True)

# 保存
print("正在保存向量文件...")
with open('case_embeddings.pkl', 'wb') as f:
    pickle.dump({
        'texts': texts,
        'embeddings': embeddings,
        'cases': cases
    }, f)

print(f"\n✅ 成功！向量化 {len(texts)} 条案例")
print(f"   向量维度：{embeddings.shape[1]}")
print(f"   保存路径：case_embeddings.pkl")
