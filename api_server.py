# api_server.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import threading
import time
import os
import requests
import json
import warnings

warnings.filterwarnings('ignore')

# 导入算法模块
from algorithm_module import (
    train_algorithm, predict_best_light, save_model,
    load_model, is_trained, Q
)

app = Flask(__name__)
CORS(app)

# ========== 云平台配置 ==========
CLOUD_API_BASE = "http://www.0531yun.com"
USERNAME = "h260725zlyl"  # ⚠️ 替换为你的实际用户名
PASSWORD = "h260725zlyl"  # ⚠️ 替换为你的实际密码
DEVICE_ADDR = "21159173"  # 你的设备地址

# ========== 因子映射（从JSON解析得到） ==========
# 光照: nodeId=2, registerId=5
# CO2: nodeId=4, registerId=2
# 温度: nodeId=1, registerId=1
# 湿度: nodeId=1, registerId=2

# ========== 全局状态 ==========
training_status = {
    'is_training': False,
    'progress': 0,
    'status': 'idle',
    'message': ''
}

latest_sensor_data = {
    'co2': 0,
    'temp': 0,
    'hum': 0,
    'light': 0,
    'timestamp': ''
}

best_light_result = {
    'value': 0,
    'timestamp': ''
}


# ========== 云平台API调用函数 ==========

def get_platform_token():
    """
    获取云平台访问令牌
    接口：/api/getToken
    方法：GET
    """
    url = f"{CLOUD_API_BASE}/api/getToken"
    params = {
        "loginName": USERNAME,
        "password": PASSWORD
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            result = response.json()
            if result.get('code') == 1000:
                token = result['data']['token']
                print(f"✅ 获取Token成功")
                return token
            else:
                print(f"❌ 获取Token失败: {result.get('message')}")
                return None
        else:
            print(f"❌ HTTP请求失败: {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ 获取Token异常: {e}")
        return None


def get_realtime_data_from_cloud():
    """
    从云平台获取实时数据
    接口：/api/data/getRealTimeDataByDeviceAddr
    """
    global DEVICE_ADDR

    # 1. 获取Token
    token = get_platform_token()
    if not token:
        return None

    # 2. 请求实时数据
    url = f"{CLOUD_API_BASE}/api/data/getRealTimeDataByDeviceAddr"
    headers = {
        "authorization": token
    }
    params = {
        "deviceAddrs": str(DEVICE_ADDR)
    }

    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)

        if response.status_code == 200:
            result = response.json()
            if result.get('code') == 1000:
                data = result.get('data', {})

                # ===== 修复1：如果 data 是列表，取第一个元素 =====
                if isinstance(data, list):
                    if len(data) > 0:
                        data = data[0]
                    else:
                        print(f"❌ data列表为空")
                        return None

                # 解析数据
                sensor_data = {
                    'co2': 0.0,
                    'temp': 0.0,
                    'hum': 0.0,
                    'light': 0.0,
                    'timestamp': datetime.now().isoformat()
                }

                data_items = data.get('dataItem', [])
                for item in data_items:
                    node_id = item.get('nodeId')
                    register_item = item.get('registerItem', {})

                    # ===== 修复2：如果 registerItem 是列表，取第一个元素 =====
                    if isinstance(register_item, list):
                        if len(register_item) > 0:
                            register_item = register_item[0]
                        else:
                            continue

                    register_id = register_item.get('registerId')
                    value = register_item.get('value', 0.0)

                    if node_id == 1 and register_id == 1:
                        sensor_data['temp'] = float(value)
                        print(f"  温度: {value} ℃")
                    elif node_id == 1 and register_id == 2:
                        sensor_data['hum'] = float(value)
                        print(f"  湿度: {value} %")
                    elif node_id == 2 and register_id == 5:
                        sensor_data['light'] = float(value)
                        print(f"  光照: {value} Lux")
                    elif node_id == 4 and register_id == 2:
                        sensor_data['co2'] = float(value)
                        print(f"  CO₂: {value} ppm")

                print(f"✅ 获取实时数据成功: CO₂={sensor_data['co2']:.1f}ppm, 温度={sensor_data['temp']:.1f}℃")
                return sensor_data
            else:
                print(f"❌ 获取数据失败: {result.get('message')}")
                return None
        else:
            print(f"❌ HTTP请求失败: {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ 获取实时数据异常: {e}")
        import traceback
        traceback.print_exc()
        return None


def get_history_data_from_cloud(start_time=None, end_time=None, node_id=-1):
    """获取历史数据"""
    if start_time is None:
        end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        start_time = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d %H:%M:%S")

    token = get_platform_token()
    if not token:
        return None

    url = f"{CLOUD_API_BASE}/api/data/getHistoryData"
    headers = {"authorization": token}
    params = {
        "deviceAddr": DEVICE_ADDR,
        "nodeId": node_id,
        "startTime": start_time,
        "endTime": end_time
    }

    try:
        response = requests.get(url, headers=headers, params=params, timeout=30)
        if response.status_code == 200:
            result = response.json()
            if result.get('code') == 1000:
                history_data = result.get('data', [])
                print(f"✅ 获取历史数据成功，共 {len(history_data)} 条记录")

                if len(history_data) == 0:
                    return pd.DataFrame()

                records = []
                for record in history_data:
                    record_time = record.get('recordTimeStr', '')
                    data_items = record.get('data', [])

                    row = {'时间': record_time}
                    for item in data_items:
                        # ===== 修复：如果 item 是列表，取第一个 =====
                        if isinstance(item, list):
                            if len(item) > 0:
                                item = item[0]
                            else:
                                continue

                        reg_id = item.get('registerId')
                        value = item.get('value', 0.0)
                        node = item.get('nodeId', 0)

                        key = f"{node}_{reg_id}"
                        if key == "1_1":
                            row['温度(℃)'] = float(value)
                        elif key == "1_2":
                            row['湿度(%RH)'] = float(value)
                        elif key == "2_5":
                            row['光照(lx)'] = float(value)
                        elif key == "4_2":
                            row['CO2(ppm)'] = float(value)

                    records.append(row)

                df = pd.DataFrame(records)
                print(f"📊 转换后数据列: {df.columns.tolist()}")
                return df
            else:
                print(f"❌ 获取历史数据失败: {result.get('message')}")
                return None
        else:
            print(f"❌ HTTP请求失败: {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ 获取历史数据异常: {e}")
        return None


# ========== 本地CSV数据加载（备用） ==========
def load_local_csv_data(file_path):
    """加载本地CSV数据作为备用"""
    try:
        df = pd.read_csv(file_path)

        # 转换格式
        sensor_types = ['CO2Value', 'CurrentTemperature', 'LightLux', 'RelativeHumidity']
        df_wide = df[df['属性标识符'].isin(sensor_types)].pivot_table(
            index='时间',
            columns='属性标识符',
            values='属性值',
            aggfunc='first'
        ).reset_index()

        df_wide.columns.name = None
        df_wide = df_wide.rename(columns={
            'CO2Value': 'CO2(ppm)',
            'CurrentTemperature': '温度(℃)',
            'LightLux': '光照(lx)',
            'RelativeHumidity': '湿度(%RH)'
        })

        for col in ['CO2(ppm)', '温度(℃)', '光照(lx)', '湿度(%RH)']:
            df_wide[col] = pd.to_numeric(df_wide[col], errors='coerce')

        df_wide = df_wide.dropna()
        df_wide['时间'] = pd.to_datetime(df_wide['时间'])
        df_wide = df_wide.sort_values('时间').reset_index(drop=True)

        print(f"✅ 本地CSV加载完成！共 {len(df_wide)} 条记录")
        return df_wide
    except Exception as e:
        print(f"❌ 本地CSV加载失败: {e}")
        return None


# ========== API 接口 ==========

@app.route('/api/status', methods=['GET'])
def get_status():
    """获取系统状态"""
    return jsonify({
        'is_trained': is_trained,
        'training_status': training_status,
        'latest_sensor': latest_sensor_data,
        'best_light': best_light_result
    })


@app.route('/api/train', methods=['POST'])
def start_training():
    """启动训练"""
    global training_status

    if training_status['is_training']:
        return jsonify({'error': '训练正在进行中'}), 400

    data = request.json or {}
    episodes = data.get('episodes', 300)
    use_cloud = data.get('use_cloud', True)  # 默认使用云平台数据

    def train_thread():
        global training_status, best_light_result

        try:
            training_status['is_training'] = True
            training_status['status'] = 'training'
            training_status['progress'] = 0
            training_status['message'] = '开始训练...'

            df = None

            # 优先从云平台获取数据
            if use_cloud:
                print("📥 从云平台获取历史数据...")
                df = get_history_data_from_cloud()

            # 如果云平台数据不足，回退到本地CSV
            if df is None or len(df) < 10:
                print("⚠️ 云平台数据不足，尝试本地CSV...")
                csv_path = r"D:\OneNET_2026-06-01_全天_ESP32WIFI_20260601_124842.csv"
                if os.path.exists(csv_path):
                    df = load_local_csv_data(csv_path)
                else:
                    # 生成模拟数据用于测试
                    print("⚠️ 没有数据，生成模拟数据用于测试...")
                    df = generate_mock_data()

            if df is None or len(df) < 10:
                training_status['status'] = 'failed'
                training_status['message'] = f'数据量不足，只有 {len(df) if df is not None else 0} 条记录'
                training_status['is_training'] = False
                return

            training_status['progress'] = 30
            training_status['message'] = f'数据加载完成，共 {len(df)} 条记录'

            # 执行训练
            Q, rewards = train_algorithm(df, episodes=episodes)

            # 保存模型
            save_model("q_table.pkl")

            training_status['status'] = 'completed'
            training_status['progress'] = 100
            training_status['message'] = f'训练完成！共 {len(Q)} 个状态，{episodes} 轮'
            training_status['is_training'] = False

        except Exception as e:
            import traceback
            error_msg = traceback.format_exc()
            print(error_msg)
            training_status['status'] = 'failed'
            training_status['message'] = str(e)
            training_status['is_training'] = False

    thread = threading.Thread(target=train_thread)
    thread.start()

    return jsonify({'message': '训练已启动', 'episodes': episodes})


def generate_mock_data():
    """生成模拟数据用于测试"""
    import random
    dates = pd.date_range(start='2026-06-01', periods=500, freq='5min')
    data = {
        '时间': dates,
        'CO2(ppm)': [random.uniform(350, 900) for _ in range(500)],
        '温度(℃)': [random.uniform(20, 35) for _ in range(500)],
        '湿度(%RH)': [random.uniform(30, 80) for _ in range(500)],
        '光照(lx)': [random.uniform(50, 400) for _ in range(500)],
    }
    return pd.DataFrame(data)


@app.route('/api/predict', methods=['POST'])
def predict():
    """根据当前传感器数据预测最佳光照"""
    global best_light_result, latest_sensor_data

    data = request.json or {}

    try:
        # 如果请求中带了传感器数据，直接使用
        if data.get('co2') and data.get('temp'):
            co2 = float(data.get('co2', 0))
            temp = float(data.get('temp', 0))
            hum = float(data.get('hum', 0))
            light = float(data.get('light', 0))
            source = 'APP上传'
        else:
            # 否则从云平台获取实时数据
            print("📡 从云平台获取实时数据...")
            sensor_data = get_realtime_data_from_cloud()
            if sensor_data is None:
                return jsonify({'error': '无法获取云平台实时数据'}), 500
            co2 = sensor_data['co2']
            temp = sensor_data['temp']
            hum = sensor_data['hum']
            light = sensor_data['light']
            source = '云平台'

        # 更新最新数据
        latest_sensor_data = {
            'co2': co2,
            'temp': temp,
            'hum': hum,
            'light': light,
            'timestamp': datetime.now().isoformat()
        }

        # 预测
        if not is_trained:
            from algorithm_module import calculate_photosynthesis_potential, get_optimal_light_range
            potential = calculate_photosynthesis_potential(co2, temp)
            _, optimal = get_optimal_light_range(potential)
            result = optimal
        else:
            result = predict_best_light(co2, temp, hum, light)

        best_light_result = {
            'value': result,
            'timestamp': datetime.now().isoformat()
        }

        return jsonify({
            'best_light': result,
            'co2': co2,
            'temp': temp,
            'hum': hum,
            'current_light': light,
            'is_trained': is_trained,
            'data_source': source,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


@app.route('/api/upload_sensor', methods=['POST'])
def upload_sensor_data():
    """接收传感器上传的数据"""
    global latest_sensor_data

    data = request.json
    if not data:
        return jsonify({'error': '请提供传感器数据'}), 400

    latest_sensor_data = {
        'co2': data.get('co2', 0),
        'temp': data.get('temp', 0),
        'hum': data.get('hum', 0),
        'light': data.get('light', 0),
        'timestamp': datetime.now().isoformat()
    }

    return jsonify({
        'message': '数据接收成功',
        'received': latest_sensor_data
    })


@app.route('/api/device_info', methods=['GET'])
def get_device_info_api():
    """获取设备信息（调试用）"""
    token = get_platform_token()
    if not token:
        return jsonify({'error': '获取Token失败'}), 500

    url = f"{CLOUD_API_BASE}/api/device/getDeviceInfoByAddr"
    headers = {"authorization": token}
    params = {"deviceAddr": DEVICE_ADDR}

    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)
        if response.status_code == 200:
            result = response.json()
            return jsonify(result)
        else:
            return jsonify({'error': f'HTTP {response.status_code}'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/fetch_cloud', methods=['GET'])
def fetch_cloud_data():
    """从云平台获取实时数据并返回（供APP调用）"""
    sensor_data = get_realtime_data_from_cloud()
    if sensor_data is None:
        return jsonify({'error': '获取云平台数据失败'}), 500

    return jsonify(sensor_data)


if __name__ == '__main__':
    # 尝试加载已有模型
    try:
        load_model("q_table.pkl")
        print("✅ 已加载已有模型")
    except:
        print("⚠️ 未找到已有模型，需要训练")

    print(f"📋 设备地址: {DEVICE_ADDR}")
    print("📋 因子映射: 温度(nodeId=1,regId=1), 湿度(nodeId=1,regId=2), 光照(nodeId=2,regId=5), CO₂(nodeId=4,regId=2)")

    app.run(host='0.0.0.0', port=5000, debug=False)
