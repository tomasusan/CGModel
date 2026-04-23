import csv
import os
import hashlib


class UserManager:
    def __init__(self, csv_file='users.csv'):
        self.csv_file = csv_file
        self._ensure_csv_exists()

    def _ensure_csv_exists(self):
        """确保用户CSV文件存在，如果不存在则创建"""
        if not os.path.exists(self.csv_file):
            with open(self.csv_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['username', 'password_hash'])

    def _hash_password(self, password):
        """对密码进行哈希处理"""
        return hashlib.sha256(password.encode()).hexdigest()

    def register(self, username, password):
        """注册新用户"""
        # 检查用户是否已存在
        users = self.get_users()
        if any(user['username'] == username for user in users):
            return False, "用户名已存在"

        # 添加新用户
        try:
            with open(self.csv_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([username, self._hash_password(password)])
            return True, "注册成功"
        except Exception as e:
            print(f"注册失败: {e}")
            return False, "注册失败"

    def login(self, username, password):
        """用户登录"""
        users = self.get_users()
        for user in users:
            if user['username'] == username:
                if user['password_hash'] == self._hash_password(password):
                    return True, "登录成功"
                else:
                    return False, "密码错误"
        return False, "用户名不存在"

    def get_users(self):
        """获取所有用户"""
        users = []
        try:
            with open(self.csv_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    users.append(row)
        except Exception as e:
            print(f"读取用户失败: {e}")
        return users


# 测试代码
if __name__ == '__main__':
    manager = UserManager()

    # 测试注册
    success, message = manager.register('admin', 'password123')
    print(f"注册结果: {success}, {message}")

    # 测试登录
    success, message = manager.login('admin', 'password123')
    print(f"登录结果: {success}, {message}")

    # 测试错误登录
    success, message = manager.login('admin', 'wrongpassword')
    print(f"错误登录结果: {success}, {message}")
