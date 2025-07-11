# network_security.py
import socket
import ssl
import requests
from urllib.parse import urlparse
import ipaddress

class NetworkSecurityManager:
    def __init__(self):
        self.allowed_internal_ranges = [
            ipaddress.ip_network('192.168.0.0/16'),
            ipaddress.ip_network('172.16.0.0/12'),
            ipaddress.ip_network('10.0.0.0/8'),
            ipaddress.ip_network('127.0.0.0/8')
        ]
        self.blocked_external_apis = []
        self.offline_mode = False
    
    def set_offline_mode(self, mode=True):
        """오프라인 모드 설정"""
        self.offline_mode = mode
        print(f"🔒 오프라인 모드: {'활성화' if mode else '비활성화'}")
        
    def is_offline(self):
        """오프라인 모드 상태 확인"""
        return getattr(self, 'offline_mode', False)
        
    def is_internal_ip(self, ip_str):
        """내부 IP 대역 확인"""
        try:
            ip = ipaddress.ip_address(ip_str)
            return any(ip in network for network in self.allowed_internal_ranges)
        except ValueError:
            return False
    
    def validate_api_endpoint(self, url):
        """API 엔드포인트 유효성 검사"""
        try:
            parsed = urlparse(url)
            if parsed.hostname:
                ip = socket.gethostbyname(parsed.hostname)
                if not self.is_internal_ip(ip):
                    print(f"⚠️ 외부 API 호출 차단: {url}")
                    return False
            return True
        except Exception as e:
            print(f"❌ URL 검증 실패: {e}")
            return False
    
    def setup_firewall_rules(self):
        """방화벽 규칙 설정 (Linux 기준)"""
        firewall_commands = [
            # 외부 접속 차단
            "sudo iptables -A INPUT -s 192.168.0.0/16 -j ACCEPT",
            "sudo iptables -A INPUT -s 172.16.0.0/12 -j ACCEPT", 
            "sudo iptables -A INPUT -s 10.0.0.0/8 -j ACCEPT",
            "sudo iptables -A INPUT -j DROP",
            
            # 특정 포트만 허용
            "sudo iptables -A OUTPUT -p tcp --dport 80,443 -j ACCEPT",
            "sudo iptables -A OUTPUT -j DROP"
        ]
        
        return firewall_commands

# 기존 main.py 수정사항
class SecureRAGPipeline:
    def __init__(self):
        self.security_manager = NetworkSecurityManager()
        self.offline_mode = True
        
    def initialize_offline_llm(self):
        """오프라인 LLM 초기화 (Ollama 활용)"""
        try:
            # Ollama 로컬 서버 연결
            import ollama
            
            # 모델 로드 확인
            available_models = ollama.list()
            if 'llama2' not in [model['name'] for model in available_models['models']]:
                print("❌ 로컬 LLM 모델이 설치되지 않음")
                return False
                
            self.local_llm = ollama
            print("✅ 오프라인 LLM 초기화 완료")
            return True
            
        except ImportError:
            print("❌ Ollama가 설치되지 않음. pip install ollama")
            return False
    
    def secure_generate_response(self, prompt):
        """보안 강화된 응답 생성"""
        if self.offline_mode:
            return self.local_llm.generate(
                model='llama2',
                prompt=prompt,
                options={
                    'temperature': 0.7,
                    'num_predict': 150,
                    'stop': ['\n\n']
                }
            )['response']
        else:
            # 외부 API 사용 시 보안 검증
            if not self.security_manager.validate_api_endpoint("https://generativelanguage.googleapis.com"):
                return "보안 정책으로 인해 외부 API 사용이 제한됩니다."
            
            # 기존 Gemini API 호출
            return self.call_gemini_api(prompt)