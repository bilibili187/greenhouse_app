# api_server.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from datetime import datetime, timedelta
import threading
import time
import os
import warnings
warnings.filterwarnings('ignore')

# 导入算法模块
from algorithm_module import (
    train_algorithm, predict_best_light, save_model, 
    load_model, is_trained, Q
)

app = Flask(__name__)
CORS(app)

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

# ========== 数据加载函数 ==========
def load_and_prepare_data(file_path):
    """加载并转换 OneNET 导出的 CSV 数据"""
    print(f"正在读取数据: {file_path}")
    df = pd.read_csv(file_path)
    
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
    
    print(f"数据转换完成！共 {len(df_wide)} 条完整记录")
    return df_wide

# ========== API 接口 ==========

@app.route('/api/status', methods=['GET'])
def get_status():
    return jsonify({
        'is_trained': is_trained,
        'training_status': training_status,
        'latest_sensor': latest_sensor_data,
        'best_light': best_light_result
    })

@app.route('/api/train', methods=['POST'])
def start_training():
    global training_status
    
    if training_status['is_training']:
        return jsonify({'error': '训练正在进行中'}), 400
    
    data = request.json or {}
    episodes = data.get('episodes', 300)
    
    def train_thread():
        global training_status, best_light_result
        
        try:
            training_status['is_training'] = True
            training_status['status'] = 'training'
            training_status['progress'] = 0
            training_status['message'] = '开始训练...'
            
            csv_path = r"D:\OneNET_2026-06-01_全天_ESP32WIFI_20260601_124842.csv"
            
            if not os.path.exists(csv_path):
                training_status['status'] = 'failed'
                training_status['message'] = f'数据文件不存在: {csv_path}'
                training_status['is_training'] = False
                return
            
            df = load_and_prepare_data(csv_path)
            
            if len(df) < 10:
                training_status['status'] = 'failed'
                training_status['message'] = f'数据量不足，只有 {len(df)} 条记录'
                training_status['is_training'] = False
                return
            
            training_status['progress'] = 30
            training_status['message'] = f'数据加载完成，共 {len(df)} 条记录'
            
            Q, rewards = train_algorithm(df, episodes=episodes)
            
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

@app.route('/api/predict', methods=['POST'])
def predict():
    global best_light_result, latest_sensor_data
    
    data = request.json
    if not data:
        return jsonify({'error': '请提供传感器数据'}), 400
    
    try:
        co2 = float(data.get('co2', 0))
        temp = float(data.get('temp', 0))
        hum = float(data.get('hum', 0))
        light = float(data.get('light', 0))
        
        latest_sensor_data = {
            'co2': co2,
            'temp': temp,
            'hum': hum,
            'light': light,
            'timestamp': datetime.now().isoformat()
        }
        
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
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/api/upload_sensor', methods=['POST'])
def upload_sensor_data():
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

@app.route('/api/model', methods=['GET'])
def get_model_info():
    return jsonify({
        'is_trained': is_trained,
        'q_table_size': len(Q) if Q else 0,
        'actions': [100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 330, 360, 400]
    })

if __name__ == '__main__':
    try:
        load_model("q_table.pkl")
        print("✅ 已加载已有模型")
    except:
        print("⚠️ 未找到已有模型，需要训练")
    
    app.run(host='0.0.0.0', port=5000, debug=False)
