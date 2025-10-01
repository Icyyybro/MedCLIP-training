#!/usr/bin/env python3
"""
MedCLIP测试运行脚本
从项目根目录运行测试，自动切换到test目录
"""

import os
import sys
import subprocess

def main():
    # 获取当前脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    test_dir = os.path.join(script_dir, 'test')
    
    # 检查test目录是否存在
    if not os.path.exists(test_dir):
        print(f"错误：test目录不存在: {test_dir}")
        sys.exit(1)
    
    # 切换到test目录
    os.chdir(test_dir)
    print(f"切换到测试目录: {test_dir}")
    
    # 运行测试脚本，传递所有命令行参数
    test_script = os.path.join(test_dir, 'test_medclip.py')
    cmd = [sys.executable, test_script] + sys.argv[1:]
    
    print(f"运行命令: {' '.join(cmd)}")
    print("="*60)
    
    # 执行测试
    try:
        result = subprocess.run(cmd, check=True)
        print("="*60)
        print("测试完成！")
        return result.returncode
    except subprocess.CalledProcessError as e:
        print("="*60)
        print(f"测试失败，退出码: {e.returncode}")
        return e.returncode
    except KeyboardInterrupt:
        print("\n测试被用户中断")
        return 1

if __name__ == "__main__":
    sys.exit(main())
