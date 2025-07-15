import numpy as np
import sys
import os
import csv
from ovito.io import import_file
from ovito.modifiers import CreateBondsModifier, CoordinationAnalysisModifier, IdentifyDiamondModifier

AVOGADRO = 6.022e23  # 阿伏伽德罗常数
CARBON_MASS = 12.0107      # 碳原子质量(g/mol)[3](@ref)

def analyze_last_frame(file_path):
    pipeline = import_file(file_path, multiple_frames=True)
    num_frames = pipeline.source.num_frames
    last_frame_index = num_frames - 1
    
    data = pipeline.compute(last_frame_index)
    volume = data.cell.volume  # Å³
    num_atoms = data.particles.count
    
    # ==== 修正密度计算 ====
    mass_grams = (num_atoms * CARBON_MASS) / AVOGADRO
    density = mass_grams / (volume * 1e-24)  # g/cm³
    
    # 键分析（保持不变）
    pipeline.modifiers.append(CreateBondsModifier(cutoff=1.85))
    pipeline.modifiers.append(CoordinationAnalysisModifier(cutoff=1.85))
    data = pipeline.compute(last_frame_index)
    coord_numbers = data.particles['Coordination']
    
    # 结晶度分析（保持不变）
    sp2_count = np.sum(coord_numbers == 3)
    sp3_count = np.sum(coord_numbers == 4)
    sp2_percent = (sp2_count / num_atoms) * 100
    sp3_percent = (sp3_count / num_atoms) * 100
    sp3_sp2_ratio = sp3_count / sp2_count if sp2_count > 0 else np.inf
    
    pipeline.modifiers.append(IdentifyDiamondModifier())
    data = pipeline.compute(last_frame_index)
    structure_types = data.particles['Structure Type']
    crystal_atoms = np.sum((structure_types >= 1) & (structure_types <= 6))
    crystallinity = (crystal_atoms / num_atoms) * 100

    return {
        "Density": density,
        "sp2 Atoms": sp2_percent,
        "sp3 Atoms": sp3_percent,
        "sp3/sp2 Ratio": sp3_sp2_ratio,
        "Crystallinity": crystallinity
    }

# 其余代码保持不变（process_directory等函数）

def process_directory(root_dir, output_csv):
    """
    处理目录结构并输出到CSV文件
    """
    # 创建CSV文件并写入表头（使用UTF-8编码解决特殊字符问题）[2,7](@ref)
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        # 移除上标³并使用简单列名
        fieldnames = ['Folder', 'Density (g/cm3)', 'sp2 Atoms (%)', 'sp3 Atoms (%)', 'sp3/sp2 Ratio', 'Crystallinity (%)']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        # 遍历根目录下的所有子文件夹
        for folder_name in os.listdir(root_dir):
            folder_path = os.path.join(root_dir, folder_name)
            
            # 检查是否是目录
            if not os.path.isdir(folder_path):
                continue
                
            # 查找dump.xyz文件
            dump_path = os.path.join(folder_path, "dump.xyz")
            if not os.path.isfile(dump_path):
                print(f"警告: {folder_path} 中没有找到 dump.xyz 文件，跳过")
                continue
                
            try:
                # 分析文件
                print(f"\n处理文件夹: {folder_name}")
                results = analyze_last_frame(dump_path)
                
                # 准备CSV行数据
                row_data = {
                    'Folder': folder_name,
                    'Density (g/cm3)': results['Density'],
                    'sp2 Atoms (%)': results['sp2 Atoms'],
                    'sp3 Atoms (%)': results['sp3 Atoms'],
                    'sp3/sp2 Ratio': results['sp3/sp2 Ratio'],
                    'Crystallinity (%)': results['Crystallinity']
                }
                
                # 写入CSV
                writer.writerow(row_data)
                print(f"成功写入: {folder_name}")
                
            except Exception as e:
                print(f"处理 {folder_name} 时出错: {str(e)}")
                continue

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python analyze_dump.py <root_directory> <output.csv>")
        print("示例: python analyze_dump.py simulation_results analysis_output.csv")
        sys.exit(1)
    
    root_dir = sys.argv[1]
    output_csv = sys.argv[2]
    
    # 检查根目录是否存在
    if not os.path.isdir(root_dir):
        print(f"错误: 目录 '{root_dir}' 不存在")
        sys.exit(1)
    
    print(f"开始处理目录: {root_dir}")
    print(f"输出文件: {output_csv}")
    
    process_directory(root_dir, output_csv)
    
    print("\n处理完成!")
    print(f"结果已保存到: {output_csv}")