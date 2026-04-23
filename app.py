import torch
from flask import Flask, request, jsonify, session
from flask_cors import CORS
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from history_manager import HistoryManager
from user_manager import UserManager

app = Flask(__name__)
CORS(app, supports_credentials=True)  # 允许跨域请求并支持凭证
app.secret_key = 'your-secret-key'  # 用于会话加密

# 初始化历史记录管理器和用户管理器
history_manager = HistoryManager()
user_manager = UserManager()

# # --- 1. 配置模型与适配器路径 ---
# base_model_path = "models/Qwen3-4B"
# lora_adapter_path = "lora/models/Qwen3-4B_code_alignment"
#
# # --- 2. 加载模型逻辑 ---
# print("正在加载基础模型与分词器...")
# tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
# model = AutoModelForCausalLM.from_pretrained(
#     base_model_path,
#     device_map="auto",
#     torch_dtype=torch.bfloat16,
#     trust_remote_code=True
# )
#
# print("正在挂载 LoRA 适配器...")
# model = PeftModel.from_pretrained(model, lora_adapter_path)
# model.eval()

print("Not using models")

# --- 3. Prompt 构建函数 ---
def build_prompt(in_prompt, language):
    """
    按照训练规范格式化提示词，包含语言类型
    """
    if language and language != "auto":
        return f"<|user|>\nsolve problem below, write {language} code only:\n{in_prompt}\n<|assistant|>\n"
    else:
        return f"<|user|>\nsolve problem below, write code only:\n{in_prompt}\n<|assistant|>\n"


# 辅助函数：检查用户是否登录
def is_logged_in():
    return 'username' in session


# 登录接口
@app.route('/login', methods=['POST'])
def login():
    data = request.json
    username = data.get('username', '')
    password = data.get('password', '')

    if not username or not password:
        return jsonify({"error": "用户名和密码不能为空"}), 400

    success, message = user_manager.login(username, password)
    if success:
        session['username'] = username
        return jsonify({"success": True, "message": message, "username": username})
    else:
        return jsonify({"success": False, "message": message}), 401

    # 注册接口


@app.route('/register', methods=['POST'])
def register():
    print("On Login")
    data = request.json
    username = data.get('username', '')
    password = data.get('password', '')

    if not username or not password:
        return jsonify({"error": "用户名和密码不能为空"}), 400

    success, message = user_manager.register(username, password)
    if success:
        session['username'] = username
        return jsonify({"success": True, "message": message, "username": username})
    else:
        return jsonify({"success": False, "message": message}), 400

    # 登出接口


@app.route('/logout', methods=['POST'])
def logout():
    session.pop('username', None)
    return jsonify({"success": True, "message": "登出成功"})


# 检查登录状态接口
@app.route('/check-login', methods=['GET'])
def check_login():
    if is_logged_in():
        return jsonify({"logged_in": True, "username": session['username']})
    else:
        return jsonify({"logged_in": False})

    # --- 4. API 路由设置 ---


@app.route('/generate', methods=['POST'])
def generate():
    if not is_logged_in():
        return jsonify({"error": "请先登录"}), 401

    data = request.json
    instruction = data.get('instruction', '')
    language = data.get('language', 'auto')

    if not instruction:
        return jsonify({"error": "No instruction provided"}), 400

        # 【关键步骤】处理原始提示词
    formatted_prompt = build_prompt(instruction, language)

    # 编码并移动至显卡
    # inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)

    # with torch.no_grad():
    #     outputs = model.generate(
    #         **inputs,
    #         max_new_tokens=512,
    #         temperature=0.7,
    #         top_p=0.9,
    #         do_sample=True,
    #         pad_token_id=tokenizer.eos_token_id
    #     )
    #
    #     # 解码结果
    # full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    #
    # # 技巧：截断输入部分，只展示助手生成的代码内容
    # # 注意：如果使用 skip_special_tokens=True，标记可能被移除，这里建议通过长度截断
    # generated_code = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

    generated_code = """#include<cstdio>
#include<iostream>
#include<iomanip>
#include<algorithm>
#include<fstream>
using namespace std;
int n;

struct node{
	int num;
	int time;
}peo[10000];

bool my_node(node &x,node &y){
	return x.time<y.time;
}

int main(){
	ifstream fin("water.in");
	ofstream fout("water.out");
	fin>>n;
	for(int i=1;i<=n;i++){
		fin>>peo[i].time;
		peo[i].num=i;
	}
	sort(peo+1,peo+n+1,my_node);
	for(int i=1;i<=n-1;i++)
		fout<<peo[i].num<<' '; 
	fout<<peo[n].num;
	fout<<endl;
	double sum;
	for(int i=1;i<=n;i++)
		for(int j=1;j<=i-1;j++)
			sum+=peo[j].time;
	sum/=n;
	fout<<fixed<<setprecision(2)<<sum;
		
	return 0;
}
"""

    print(f"--- 接收到请求 ---")
    print(f"用户: {session['username']}")
    print(f"原始输入: {instruction[:50]}...")
    print(f"指定语言: {language}")
    print(f"生成的代码长度: {len(generated_code)} 字符")

    # 添加到历史记录
    history_item = history_manager.add_history(session['username'], instruction, generated_code.strip(), language)

    return jsonify({"code": generated_code.strip(), "history_item": history_item})


# 获取历史记录
@app.route('/history', methods=['GET'])
def get_history():
    if not is_logged_in():
        return jsonify({"error": "请先登录"}), 401

    history = history_manager.get_history(session['username'])
    return jsonify({"history": history})


# 清空历史记录
@app.route('/history/clear', methods=['POST'])
def clear_history():
    if not is_logged_in():
        return jsonify({"error": "请先登录"}), 401

    success = history_manager.clear_history(session['username'])
    return jsonify({"success": success})


if __name__ == '__main__':
    # 这里的 5000 端口记得与 SSH 隧道映射的端口保持一致
    app.run(host='0.0.0.0', port=5000, debug=False)
