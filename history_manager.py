import csv
import json
import os
from datetime import datetime


class HistoryManager:
    def __init__(self, csv_file='history.csv'):
        self.csv_file = csv_file
        self._ensure_csv_exists()

    def _ensure_csv_exists(self):
        """确保CSV文件存在，如果不存在则创建"""
        if not os.path.exists(self.csv_file):
            with open(self.csv_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['id', 'username', 'instruction', 'code', 'timestamp', 'code_language'])

    def get_history(self, username=None):
        """读取历史记录，可选择按用户过滤"""
        history = []
        try:
            with open(self.csv_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # 转换数据类型
                    row['timestamp'] = int(row['timestamp'])
                    # 如果指定了用户名，则只返回该用户的历史记录
                    if username is None or row.get('username') == username:
                        history.append(row)
        except Exception as e:
            print(f"读取历史记录失败: {e}")
        # 按时间戳降序排序
        history.sort(key=lambda x: x['timestamp'], reverse=True)
        return history

    def add_history(self, username, instruction, code, code_language='auto'):
        """添加历史记录"""
        new_record = {
            'id': str(int(datetime.now().timestamp() * 1000)),  # 使用时间戳作为ID
            'username': username,
            'instruction': instruction,
            'code': code,
            'timestamp': int(datetime.now().timestamp()),
            'code_language': code_language
        }

        try:
            with open(self.csv_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    new_record['id'],
                    new_record['username'],
                    new_record['instruction'],
                    new_record['code'],
                    new_record['timestamp'],
                    new_record['code_language']
                ])
            return new_record
        except Exception as e:
            print(f"添加历史记录失败: {e}")
            return None

    def clear_history(self, username=None):
        """清空历史记录，可选择按用户过滤"""
        try:
            # 读取所有历史记录
            all_history = []
            if username is not None:
                with open(self.csv_file, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        if row.get('username') != username:
                            all_history.append(row)

            # 重新写入文件
            with open(self.csv_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['id', 'username', 'instruction', 'code', 'timestamp', 'code_language'])
                if username is not None:
                    for record in all_history:
                        writer.writerow([
                            record['id'],
                            record['username'],
                            record['instruction'],
                            record['code'],
                            record['timestamp'],
                            record['code_language']
                        ])
            return True
        except Exception as e:
            print(f"清空历史记录失败: {e}")
            return False


# 测试代码
if __name__ == '__main__':
    manager = HistoryManager()

    # 测试添加历史记录
    test_record = manager.add_history(
        instruction='测试指令',
        code='print("Hello, World!")',
        code_language='python'
    )
    print(f"添加的历史记录: {test_record}")

    # 测试获取历史记录
    history = manager.get_history()
    print(f"历史记录数量: {len(history)}")
    if history:
        print(f"最新的历史记录: {history[0]}")

    # 测试清空历史记录
    # manager.clear_history()
    # print("历史记录已清空")
