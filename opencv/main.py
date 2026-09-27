import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.helpers import print_environment_info
from examples.basic_operations import run_basic_demo
from examples.filter_enhance import run_filter_demo
from examples.feature_detection import run_feature_demo


def show_menu():
    print("\n" + "=" * 60)
    print("           OpenCV Python 开发项目 - 主菜单")
    print("=" * 60)
    print("1. 查看环境信息")
    print("2. 运行基础图像操作演示")
    print("3. 运行图像滤波与增强演示")
    print("4. 运行特征检测与匹配演示")
    print("5. 运行全部演示")
    print("0. 退出")
    print("=" * 60)


def run_all_demos():
    print("\n" + "=" * 60)
    print("开始运行全部演示...")
    print("=" * 60)
    
    try:
        run_basic_demo()
    except Exception as e:
        print(f"基础图像操作演示出错: {e}")
    
    print()
    try:
        run_filter_demo()
    except Exception as e:
        print(f"图像滤波与增强演示出错: {e}")
    
    print()
    try:
        run_feature_demo()
    except Exception as e:
        print(f"特征检测与匹配演示出错: {e}")
    
    print("\n" + "=" * 60)
    print("全部演示运行完成!")
    print("=" * 60)


def interactive_mode():
    while True:
        show_menu()
        choice = input("\n请输入选项 (0-5): ").strip()
        
        if choice == '0':
            print("感谢使用，再见!")
            break
        elif choice == '1':
            print_environment_info()
        elif choice == '2':
            run_basic_demo()
        elif choice == '3':
            run_filter_demo()
        elif choice == '4':
            run_feature_demo()
        elif choice == '5':
            run_all_demos()
        else:
            print("无效选项，请重新输入!")


def command_line_mode(args):
    if args.env:
        print_environment_info()
    
    if args.demo == 'basic':
        run_basic_demo()
    elif args.demo == 'filter':
        run_filter_demo()
    elif args.demo == 'feature':
        run_feature_demo()
    elif args.demo == 'all':
        run_all_demos()


def main():
    parser = argparse.ArgumentParser(
        description="OpenCV Python 开发项目 - 计算机视觉示例集合"
    )
    parser.add_argument(
        '--demo', type=str, choices=['basic', 'filter', 'feature', 'all'],
        help='直接运行指定演示: basic(基础), filter(滤波), feature(特征), all(全部)'
    )
    parser.add_argument(
        '--env', action='store_true',
        help='显示环境信息'
    )
    parser.add_argument(
        '--cli', action='store_true',
        help='使用命令行模式 (不显示交互菜单)'
    )
    
    args = parser.parse_args()
    
    if args.demo or args.env or args.cli:
        command_line_mode(args)
    else:
        print_environment_info()
        interactive_mode()


if __name__ == "__main__":
    main()
