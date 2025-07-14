import numpy as np
import sys
from ovito.io import import_file
from ovito.modifiers import CreateBondsModifier, CoordinationAnalysisModifier, IdentifyDiamondModifier

def analyze_last_frame(file_path):
    """
    分析GPUMD生成的dump.xyz文件的最后一帧
    """
    # 导入文件并获取总帧数
    pipeline = import_file(file_path, multiple_frames=True)
    num_frames = pipeline.source.num_frames
    last_frame_index = num_frames - 1
    
    print(f"检测到 {num_frames} 帧数据，将分析最后一帧（索引 {last_frame_index})")
    
    # 计算最后一帧的基础属性
    data = pipeline.compute(last_frame_index)
    volume = data.cell.volume  # 系统体积（Å³）
    num_atoms = data.particles.count  # 原子总数
    
    # 计算质量（假设为碳体系）
    atomic_mass = 12.01  # 碳原子质量（g/mol）
    total_relative_mass = num_atoms * atomic_mass  # 系统总质量（相对质量）
    actual_mass = total_relative_mass / 6.022e23  # 实际质量（克）
    
    # 计算密度（g/cm³）
    # 注意：1 Å³ = 10⁻²⁴ cm³
    density = (actual_mass * 1e24) / volume
    
    # 添加创建键的修饰器（用于配位数分析）
    pipeline.modifiers.append(CreateBondsModifier(cutoff=1.85))
    pipeline.modifiers.append(CoordinationAnalysisModifier(cutoff=1.85))
    
    # 计算最后一帧的配位数
    data = pipeline.compute(last_frame_index)
    coord_numbers = data.particles['Coordination']
    
    # 计算sp²和sp³比例
    sp2_count = np.sum(coord_numbers == 3)
    sp3_count = np.sum(coord_numbers == 4)
    sp2_percent = (sp2_count / num_atoms) * 100
    sp3_percent = (sp3_count / num_atoms) * 100
    
    # 计算sp3/sp2比值（避免除以零）
    sp3_sp2_ratio = sp3_count / sp2_count if sp2_count > 0 else np.inf
    
    # 添加金刚石结构识别修饰器（用于结晶度分析）
    pipeline.modifiers.append(IdentifyDiamondModifier())
    
    # 计算最后一帧的结晶度
    data = pipeline.compute(last_frame_index)
    structure_types = data.particles['Structure Type']
    
    # 计算结晶度（识别为金刚石结构的原子百分比）
    crystal_atoms = np.sum((structure_types >= 1) & (structure_types <= 6))
    crystallinity = (crystal_atoms / num_atoms) * 100
    
    # 返回计算结果
    return {
        "Density (g/cm³)": density,
        "sp² Atoms (%)": sp2_percent,
        "sp³ Atoms (%)": sp3_percent,
        "sp³/sp² Ratio": sp3_sp2_ratio,
        "Crystallinity (%)": crystallinity
    }

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_dump.py <dump.xyz>")
        sys.exit(1)
    
    # 分析文件的最后一帧
    results = analyze_last_frame(sys.argv[1])
    
    # 打印结果
    print("\nSystem Properties Analysis (Last Frame):")
    print("-" * 60)
    for prop, value in results.items():
        # 根据不同属性调整格式化方式
        if "%" in prop:
            print(f"{prop}: {value:.4f}")
        elif "Ratio" in prop:
            if value == np.inf:
                print(f"{prop}: Infinity (no sp² atoms)")
            else:
                print(f"{prop}: {value:.4f}")
        else:
            print(f"{prop}: {value:.6f}")