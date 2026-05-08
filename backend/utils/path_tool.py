# 文件路径设置
import os

# 获取项目根目录
def get_project_root() -> str:
    current_file = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file)
    project_root = os.path.dirname(os.path.dirname(current_dir))

    return project_root

# 获取一个文件的绝对路径
def get_abs_path(relative_path : str) -> str:
    # 拼接
    return os.path.join(get_project_root(), relative_path)

