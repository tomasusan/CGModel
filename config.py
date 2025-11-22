import os
import configparser

class Config:
    def __init__(self, filename="config.ini"):
        self.cfg = configparser.ConfigParser()
        if not os.path.exists(filename):
            raise FileNotFoundError(f"配置文件不存在: {filename}")

        self.cfg.read(filename, encoding="utf-8")

    def get(self, section, key, fallback=None):
        return self.cfg.get(section, key, fallback=fallback)

    def get_int(self, section, key, fallback=None):
        return self.cfg.getint(section, key, fallback=fallback)

    def get_float(self, section, key, fallback=None):
        return self.cfg.getfloat(section, key, fallback=fallback)

    def get_bool(self, section, key, fallback=None):
        return self.cfg.getboolean(section, key, fallback=fallback)

# 创建全局唯一配置实例
cfg = Config()
