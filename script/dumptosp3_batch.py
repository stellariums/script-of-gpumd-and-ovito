###对导入的一系列dump.xyz文件的某一帧进行计算，得到对应的sp3,sp2,sp2/sp3,结晶度，密度信息
# python batch_dump.py C:\Users\20414\Desktop\y\y  .\output9.csv 9
import numpy as np
import sys
import os
import csv
from ovito.io import import_file
from ovito.modifiers import CreateBondsModifier, CoordinationAnalysisModifier, IdentifyDiamondModifier

AVOGADRO = 6.022e23  # 阿伏伽德罗常数
CARBON_MASS = 12.0107  # 碳原子质量(g/mol)

def analyze_frame(file_path, frame_index=None):
    """
    分析指定帧的数据
    """
    pipeline = import_file(file_path, multiple_frames=True)
    num_frames = pipeline.source.num_frames
    
    # 确定要分析的帧索引
    if frame_index is None:
        frame_index = num_frames - 1  # 默认最后一帧
    elif frame_index < 0:
        # 支持负数索引（例如-1表示最后一帧）
        frame_index = max(0, num_frames + frame_index)
    
    # 确保帧索引在有效范围内
    if frame_index < 0 or frame_index >= num_frames:
        raise ValueError(f"帧索引 {frame_index} 超出范围 (0-{num_frames-1})")
    
    data = pipeline.compute(frame_index)
    volume = data.cell.volume  # Å³
    num_atoms = data.particles.count
    
    # 密度计算
    mass_grams = (num_atoms * CARBON_MASS) / AVOGADRO
    density = mass_grams / (volume * 1e-24)  # g/cm³
    
    # 键分析
    pipeline.modifiers.append(CreateBondsModifier(cutoff=1.85))
    pipeline.modifiers.append(CoordinationAnalysisModifier(cutoff=1.85))
    data = pipeline.compute(frame_index)
    coord_numbers = data.particles['Coordination']
    
    # 结晶度分析
    sp2_count = np.sum(coord_numbers == 3)
    sp3_count = np.sum(coord_numbers == 4)
    sp2_percent = (sp2_count / num_atoms) * 100
    sp3_percent = (sp3_count / num_atoms) * 100
    sp3_sp2_ratio = sp3_count / sp2_count if sp2_count > 0 else np.inf
    
    pipeline.modifiers.append(IdentifyDiamondModifier())
    data = pipeline.compute(frame_index)
    structure_types = data.particles['Structure Type']
    crystal_atoms = np.sum((structure_types >= 1) & (structure_types <= 6))
    crystallinity = (crystal_atoms / num_atoms) * 100

    return {
        "Frame": frame_index,
        "Density": density,
        "sp2 Atoms": sp2_percent,
        "sp3 Atoms": sp3_percent,
        "sp3/sp2 Ratio": sp3_sp2_ratio,
        "Crystallinity": crystallinity
    }

def find_dump_files(root_dir):
    """
    递归查找所有dump.xyz文件[1,6](@ref)
    """
    dump_files = []
    # 遍历所有子目录
    for dirpath, _, filenames in os.walk(root_dir):
        # 检查当前目录是否有dump.xyz文件
        if "dump.xyz" in filenames:
            dump_path = os.path.join(dirpath, "dump.xyz")
            dump_files.append((dirpath, dump_path))
    return dump_files

def process_directory(root_dir, output_csv, frame_index=None):
    """
    处理目录结构并输出到CSV文件
    """
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        # 表头
        fieldnames = ['Folder', 'Subdirectory', 'Frame', 'Density (g/cm3)', 
                      'sp2 Atoms (%)', 'sp3 Atoms (%)', 'sp3/sp2 Ratio', 'Crystallinity (%)']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        # 查找所有dump.xyz文件[1,6](@ref)
        dump_files = find_dump_files(root_dir)
        
        if not dump_files:
            print(f"警告: 在 {root_dir} 及其子目录中未找到任何 dump.xyz 文件")
            return
        
        print(f"找到 {len(dump_files)} 个 dump.xyz 文件")
        
        # 处理每个找到的dump.xyz文件
        for folder_path, dump_path in dump_files:
            try:
                # 获取相对于根目录的子目录路径
                rel_path = os.path.relpath(folder_path, root_dir)
                print(f"\n处理目录: {rel_path}")
                
                # 分析文件
                results = analyze_frame(dump_path, frame_index)
                
                # 准备CSV行数据
                row_data = {
                    'Folder': os.path.basename(root_dir),
                    'Subdirectory': rel_path,
                    'Frame': results['Frame'],
                    'Density (g/cm3)': results['Density'],
                    'sp2 Atoms (%)': results['sp2 Atoms'],
                    'sp3 Atoms (%)': results['sp3 Atoms'],
                    'sp3/sp2 Ratio': results['sp3/sp2 Ratio'],
                    'Crystallinity (%)': results['Crystallinity']
                }
                
                # 写入CSV
                writer.writerow(row_data)
                print(f"成功分析帧 {results['Frame']} 并写入: {rel_path}")
                
            except Exception as e:
                print(f"处理 {folder_path} 时出错: {str(e)}")
                continue

if __name__ == "__main__":
    # 解析命令行参数 
    frame_index = None  # 默认分析最后一帧
    
    if len(sys.argv) < 3 or len(sys.argv) > 4:
        print("用法: python analyze_dump.py <根目录> <输出文件> [帧索引]")
        print("示例: python analyze_dump.py C:\\Users\\20414\\Desktop\\y analysis_output.csv")
        print("示例: python analyze_dump.py C:\\Users\\20414\\Desktop\\y analysis_output.csv 10 (分析第10帧)")
        print("示例: python analyze_dump.py C:\\Users\\20414\\Desktop\\y analysis_output.csv -1 (分析最后一帧)")
        sys.exit(1)
    
    root_dir = sys.argv[1]
    output_csv = sys.argv[2]
    
    # 处理可选的帧索引参数
    if len(sys.argv) == 4:
        try:
            frame_index = int(sys.argv[3])
        except ValueError:
            print("错误: 帧索引必须是整数")
            sys.exit(1)
    
    if not os.path.isdir(root_dir):
        print(f"错误: 目录 '{root_dir}' 不存在")
        sys.exit(1)
    
    print(f"开始处理目录: {root_dir}")
    print(f"输出文件: {output_csv}")
    print(f"分析帧: {'最后一帧' if frame_index is None else frame_index}")
    
    process_directory(root_dir, output_csv, frame_index)
    
    print("\n处理完成!")
    print(f"结果已保存到: {output_csv}")