# app_ui.py
import flet as ft
import requests
import json
from datetime import datetime
import threading
import ssl

# ========== API配置 ==========
# 注意：使用 HTTP，不是 HTTPS
API_BASE_URL = "http://192.168.1.12:5000/api"  # 改成你电脑的实际IP

# 如果遇到 SSL 错误，可以取消注释下面的代码禁用SSL验证
# import urllib3
# urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class GreenhouseApp:
    def __init__(self):
        self.sensor_data = {
            'co2': 0,
            'temp': 0,
            'hum': 0,
            'light': 0
        }
        self.best_light = 0
        self.is_trained = False
        self.training_status = 'idle'

    def main(self, page: ft.Page):
        self.page = page
        page.title = "智能温室光调控系统"
        page.theme_mode = ft.ThemeMode.LIGHT
        page.scroll = ft.ScrollMode.AUTO
        page.padding = 20
        page.horizontal_alignment = ft.CrossAxisAlignment.CENTER

        # ===== 顶部标题 =====
        page.appbar = ft.AppBar(
            title=ft.Row([
                ft.Icon(ft.Icons.WB_SUNNY, color=ft.Colors.WHITE),
                ft.Text("智能温室光调控", size=22, weight=ft.FontWeight.BOLD),
            ]),
            bgcolor=ft.Colors.GREEN_700,
            color=ft.Colors.WHITE,
            actions=[
                ft.IconButton(ft.Icons.REFRESH, tooltip="刷新", on_click=self.refresh_all),
                ft.IconButton(ft.Icons.SETTINGS, tooltip="设置"),
            ]
        )

        # ===== 状态卡片 =====
        self.status_text = ft.Text("就绪", size=14)
        self.status_card = ft.Card(
            content=ft.Container(
                content=ft.Column([
                    ft.Row([
                        ft.Icon(ft.Icons.CHECK_CIRCLE, color=ft.Colors.GREEN_500),
                        ft.Text("系统状态", size=18, weight=ft.FontWeight.BOLD),
                    ]),
                    self.status_text,
                ], spacing=10),
                padding=15,
                width=400,
            ),
            elevation=3,
        )

        # ===== 传感器数据显示 =====
        self.sensor_co2_text = ft.Text("0.0 ppm", size=16)
        self.sensor_temp_text = ft.Text("0.0 ℃", size=16)
        self.sensor_hum_text = ft.Text("0.0 %RH", size=16)
        self.sensor_light_text = ft.Text("0.0 lx", size=16)
        self.time_text = ft.Text("--", size=12, color=ft.Colors.GREY_600)

        self.sensor_card = ft.Card(
            content=ft.Container(
                content=ft.Column([
                    ft.Row([
                        ft.Icon(ft.Icons.SENSORS, color=ft.Colors.BLUE_500),
                        ft.Text("实时传感器数据", size=18, weight=ft.FontWeight.BOLD),
                    ]),
                    ft.Divider(height=10),
                    ft.Row([
                        self._make_sensor_item(ft.Icons.CO2, "CO₂", self.sensor_co2_text),
                        self._make_sensor_item(ft.Icons.THERMOSTAT, "温度", self.sensor_temp_text),
                    ], spacing=20),
                    ft.Row([
                        self._make_sensor_item(ft.Icons.WATER_DROP, "湿度", self.sensor_hum_text),
                        self._make_sensor_item(ft.Icons.LIGHT, "光照", self.sensor_light_text),
                    ], spacing=20),
                    ft.Row([
                        ft.Text("更新时间: ", size=12, color=ft.Colors.GREY_600),
                        self.time_text,
                    ]),
                ], spacing=10),
                padding=15,
                width=400,
            ),
            elevation=3,
        )

        # ===== 推荐结果显示 =====
        self.result_value = ft.Text("--", size=48, weight=ft.FontWeight.BOLD, color=ft.Colors.AMBER_700)
        self.result_note = ft.Text("", size=12, color=ft.Colors.GREY_500)

        self.result_card = ft.Card(
            content=ft.Container(
                content=ft.Column([
                    ft.Row([
                        ft.Icon(ft.Icons.LIGHTBULB, color=ft.Colors.AMBER_700),
                        ft.Text("推荐光照强度", size=18, weight=ft.FontWeight.BOLD),
                    ], alignment=ft.MainAxisAlignment.CENTER),
                    self.result_value,
                    ft.Text("勒克斯 (lx)", size=14, color=ft.Colors.GREY_600),
                    self.result_note,
                ], horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=5),
                padding=20,
                width=400,
                bgcolor=ft.Colors.AMBER_50,
            ),
            elevation=5,
        )

        # ===== 控制按钮 =====
        self.btn_train = ft.ElevatedButton(
            "🧠 训练算法",
            icon=ft.Icons.ANALYTICS,
            style=ft.ButtonStyle(
                bgcolor=ft.Colors.BLUE_600,
                color=ft.Colors.WHITE,
                padding=20,
            ),
            on_click=self.start_training,
            width=180,
        )

        self.btn_predict = ft.ElevatedButton(
            "⚡ 获取推荐",
            icon=ft.Icons.PLAY_ARROW,
            style=ft.ButtonStyle(
                bgcolor=ft.Colors.GREEN_600,
                color=ft.Colors.WHITE,
                padding=20,
            ),
            on_click=self.get_prediction,
            width=180,
        )

        # ===== 新增：从云平台获取数据按钮 =====
        self.btn_fetch_cloud = ft.ElevatedButton(
            "☁️ 获取云数据",
            icon=ft.Icons.CLOUD_DOWNLOAD,
            style=ft.ButtonStyle(
                bgcolor=ft.Colors.PURPLE_600,
                color=ft.Colors.WHITE,
                padding=20,
            ),
            on_click=self.fetch_from_cloud,
            width=180,
        )

        self.btn_upload = ft.OutlinedButton(
            "📤 模拟上传数据",
            icon=ft.Icons.UPLOAD_FILE,
            on_click=self.simulate_upload,
            width=180,
        )

        # ===== 训练进度条 =====
        self.progress_bar = ft.ProgressBar(
            width=400,
            visible=False,
            color=ft.Colors.BLUE_500,
        )
        self.progress_text = ft.Text("", size=12, color=ft.Colors.GREY_600)

        # ===== 日志区域 =====
        self.log_text = ft.Text("系统就绪", size=12, color=ft.Colors.GREY_600, max_lines=5)
        self.log_card = ft.Card(
            content=ft.Container(
                content=ft.Column([
                    ft.Row([
                        ft.Icon(ft.Icons.TERMINAL, color=ft.Colors.GREY_600),
                        ft.Text("日志", size=16, weight=ft.FontWeight.BOLD),
                    ]),
                    self.log_text,
                ], spacing=5),
                padding=15,
                width=400,
            ),
            elevation=2,
        )

        # ===== 组装页面 =====
        page.add(
            self.status_card,
            ft.Divider(height=10),
            self.sensor_card,
            ft.Divider(height=10),
            self.result_card,
            ft.Divider(height=10),
            ft.Row(
                [self.btn_train, self.btn_predict],
                alignment=ft.MainAxisAlignment.CENTER,
                spacing=20,
            ),
            ft.Row(
                [self.btn_fetch_cloud, self.btn_upload],
                alignment=ft.MainAxisAlignment.CENTER,
                spacing=20,
            ),
            ft.Divider(height=5),
            self.progress_bar,
            self.progress_text,
            ft.Divider(height=10),
            self.log_card,
        )

        # 启动时自动刷新
        self.refresh_all(None)

    def _make_sensor_item(self, icon, label, value_text):
        """创建单个传感器显示项"""
        return ft.Column([
            ft.Row([
                ft.Icon(icon, size=20, color=ft.Colors.BLUE_500),
                ft.Text(f"{label}: ", size=14, weight=ft.FontWeight.BOLD),
            ]),
            value_text,
        ], horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=2)

    def refresh_all(self, e):
        """刷新所有数据"""
        self.add_log("🔄 刷新数据...")
        self.fetch_status()
        self.fetch_sensor_data()
        self.fetch_prediction()

    def fetch_status(self):
        """获取系统状态"""
        try:
            response = requests.get(f"{API_BASE_URL}/status", timeout=3)
            if response.status_code == 200:
                data = response.json()
                self.is_trained = data.get('is_trained', False)
                self.training_status = data.get('training_status', {}).get('status', 'idle')

                if self.is_trained:
                    self.status_text.value = "✅ 已训练"
                    self.status_text.color = ft.Colors.GREEN_500
                else:
                    self.status_text.value = "⚠️ 未训练"
                    self.status_text.color = ft.Colors.ORANGE_500

                self.page.update()
        except Exception as e:
            self.add_log(f"❌ 获取状态失败: {str(e)}")

    def fetch_sensor_data(self):
        """获取传感器数据"""
        try:
            response = requests.get(f"{API_BASE_URL}/status", timeout=3)
            if response.status_code == 200:
                data = response.json()
                sensor = data.get('latest_sensor', {})
                if sensor:
                    self.sensor_data = {
                        'co2': sensor.get('co2', 0),
                        'temp': sensor.get('temp', 0),
                        'hum': sensor.get('hum', 0),
                        'light': sensor.get('light', 0),
                    }
                    self.update_sensor_display()
        except Exception as e:
            pass

    def update_sensor_display(self):
        """更新传感器显示"""
        self.sensor_co2_text.value = f"{self.sensor_data.get('co2', 0):.1f} ppm"
        self.sensor_temp_text.value = f"{self.sensor_data.get('temp', 0):.1f} ℃"
        self.sensor_hum_text.value = f"{self.sensor_data.get('hum', 0):.1f} %RH"
        self.sensor_light_text.value = f"{self.sensor_data.get('light', 0):.1f} lx"
        self.time_text.value = datetime.now().strftime("%H:%M:%S")
        self.page.update()

    def fetch_prediction(self):
        """获取推荐结果"""
        try:
            response = requests.get(f"{API_BASE_URL}/status", timeout=3)
            if response.status_code == 200:
                data = response.json()
                result = data.get('best_light', {})
                if result and result.get('value', 0) > 0:
                    self.best_light = result['value']
                    self.result_value.value = f"{self.best_light:.0f}"
                    self.result_value.color = ft.Colors.GREEN_700 if self.best_light > 100 else ft.Colors.AMBER_700
                    self.page.update()
        except Exception as e:
            pass

    def start_training(self, e):
        """启动训练"""
        self.add_log("🧠 启动训练...")
        self.btn_train.disabled = True
        self.progress_bar.visible = True
        self.progress_text.value = "训练中 0%"
        self.page.update()

        def train_thread():
            try:
                response = requests.post(
                    f"{API_BASE_URL}/train",
                    json={'episodes': 300, 'use_cloud': True},
                    timeout=10
                )
                if response.status_code == 200:
                    self.add_log("✅ 训练启动成功")
                    self.poll_training_status()
                else:
                    self.add_log(f"❌ 训练启动失败: {response.text}")
            except Exception as e:
                self.add_log(f"❌ 训练错误: {str(e)}")
            finally:
                self.btn_train.disabled = False
                self.progress_bar.visible = False
                self.page.update()

        threading.Thread(target=train_thread, daemon=True).start()

    def poll_training_status(self):
        """轮询训练状态"""
        import time
        for _ in range(120):
            try:
                response = requests.get(f"{API_BASE_URL}/status", timeout=2)
                if response.status_code == 200:
                    data = response.json()
                    status = data.get('training_status', {})
                    progress = status.get('progress', 0)
                    self.progress_text.value = f"训练中 {progress:.1f}%"
                    self.page.update()

                    if status.get('status') == 'completed':
                        self.add_log("✅ 训练完成！")
                        self.is_trained = True
                        self.status_text.value = "✅ 已训练"
                        self.status_text.color = ft.Colors.GREEN_500
                        self.page.update()
                        break
                    elif status.get('status') == 'failed':
                        self.add_log(f"❌ 训练失败: {status.get('message', '')}")
                        break
                time.sleep(1)
            except Exception as e:
                pass

    def get_prediction(self, e):
        """获取推荐"""
        self.add_log("⚡ 获取推荐...")

        try:
            response = requests.post(
                f"{API_BASE_URL}/predict",
                json={
                    'co2': self.sensor_data.get('co2', 400),
                    'temp': self.sensor_data.get('temp', 25),
                    'hum': self.sensor_data.get('hum', 60),
                    'light': self.sensor_data.get('light', 200),
                },
                timeout=10
            )

            if response.status_code == 200:
                data = response.json()
                self.best_light = data.get('best_light', 0)
                self.result_value.value = f"{self.best_light:.0f}"
                self.result_value.color = ft.Colors.GREEN_700 if self.best_light > 100 else ft.Colors.AMBER_700

                if data.get('is_trained'):
                    self.result_note.value = "基于Q-learning算法"
                else:
                    self.result_note.value = "⚠️ 基于启发式规则（未训练）"

                # 同时更新传感器数据（如果从云平台获取的）
                if data.get('data_source') == '云平台':
                    self.sensor_data = {
                        'co2': data.get('co2', 0),
                        'temp': data.get('temp', 0),
                        'hum': data.get('hum', 0),
                        'light': data.get('current_light', 0),
                    }
                    self.update_sensor_display()
                    self.add_log(f"☁️ 数据来源: 云平台")
                else:
                    self.add_log(f"📱 数据来源: APP上传")

                self.add_log(f"✅ 推荐光照: {self.best_light:.0f} lx")
                self.page.update()
            else:
                self.add_log(f"❌ 获取推荐失败: {response.text}")
        except Exception as e:
            self.add_log(f"❌ 错误: {str(e)}")

    def fetch_from_cloud(self, e):
        """从云平台获取实时数据"""
        self.add_log("☁️ 从云平台获取实时数据...")
        self.btn_fetch_cloud.disabled = True
        self.page.update()

        try:
            # 调用预测接口，不传传感器数据，服务器会从云平台获取
            response = requests.post(
                f"{API_BASE_URL}/predict",
                json={},
                timeout=10
            )

            if response.status_code == 200:
                data = response.json()
                self.sensor_data = {
                    'co2': data.get('co2', 0),
                    'temp': data.get('temp', 0),
                    'hum': data.get('hum', 0),
                    'light': data.get('current_light', 0),
                }
                self.update_sensor_display()
                self.add_log(f"✅ CO₂={data['co2']:.0f}ppm, 温度={data['temp']:.1f}℃, 光照={data['current_light']:.0f}lx")

                # 更新推荐结果
                self.best_light = data.get('best_light', 0)
                self.result_value.value = f"{self.best_light:.0f}"
                self.result_value.color = ft.Colors.GREEN_700 if self.best_light > 100 else ft.Colors.AMBER_700
                self.add_log(f"✅ 推荐光照: {self.best_light:.0f} lx")
                self.page.update()
            else:
                self.add_log(f"❌ 获取失败: {response.text}")
        except requests.exceptions.ConnectionError:
            self.add_log("❌ 连接失败，请检查API服务器是否运行")
        except Exception as e:
            self.add_log(f"❌ 错误: {str(e)}")
        finally:
            self.btn_fetch_cloud.disabled = False
            self.page.update()

    def simulate_upload(self, e):
        """模拟上传传感器数据"""
        import random

        mock_data = {
            'co2': random.uniform(350, 900),
            'temp': random.uniform(20, 35),
            'hum': random.uniform(30, 80),
            'light': random.uniform(50, 400),
        }

        self.sensor_data = mock_data
        self.update_sensor_display()
        self.add_log(f"📤 上传数据: CO₂={mock_data['co2']:.0f}ppm, 温度={mock_data['temp']:.1f}℃")

        try:
            requests.post(
                f"{API_BASE_URL}/upload_sensor",
                json=mock_data,
                timeout=3
            )
        except:
            pass

    def add_log(self, message):
        """添加日志"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.value = f"[{timestamp}] {message}\n" + self.log_text.value
        lines = self.log_text.value.split('\n')[:10]
        self.log_text.value = '\n'.join(lines)
        self.page.update()


# ========== 启动APP ==========
def main(page: ft.Page):
    """启动APP - 接收page参数"""
    app = GreenhouseApp()
    app.main(page)


if __name__ == "__main__":
    # 桌面模式
    ft.app(target=main)
    # 如果想用Web模式，取消下面这行的注释，并注释掉上面那行
    # ft.app(target=main, view=ft.AppView.WEB_BROWSER)
