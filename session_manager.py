import uuid
from datetime import datetime, timedelta
from typing import Dict, Optional, List
import threading
import time

class SessionManager:
    """Manages chat sessions with automatic timeout"""
    
    def __init__(self, timeout_minutes: int = 5):
        self.timeout_minutes = timeout_minutes
        self.sessions: Dict[str, Dict] = {}
        self.lock = threading.Lock()
        
        # Start cleanup thread
        self.cleanup_thread = threading.Thread(target=self._cleanup_expired_sessions, daemon=True)
        self.cleanup_thread.start()
    
    def create_session(self) -> str:
        """Create a new session and return session ID"""
        session_id = str(uuid.uuid4())
        
        with self.lock:
            self.sessions[session_id] = {
                'created_at': datetime.now(),
                'last_activity': datetime.now(),
                'expires_at': datetime.now() + timedelta(minutes=self.timeout_minutes)
            }
        
        print(f"✅ Created new session: {session_id}")
        return session_id
    
    def is_session_active(self, session_id: str) -> bool:
        """Check if a session is active (not expired)"""
        with self.lock:
            if session_id not in self.sessions:
                return False
            
            session = self.sessions[session_id]
            return datetime.now() < session['expires_at']
    
    def update_activity(self, session_id: str) -> bool:
        """Update the last activity time for a session"""
        with self.lock:
            if session_id not in self.sessions:
                return False
            
            # Check if session is expired
            if datetime.now() >= self.sessions[session_id]['expires_at']:
                return False
            
            # Update activity and extend expiration
            self.sessions[session_id]['last_activity'] = datetime.now()
            self.sessions[session_id]['expires_at'] = datetime.now() + timedelta(minutes=self.timeout_minutes)
            
            return True
    
    def get_session_info(self, session_id: str) -> Optional[Dict]:
        """Get information about a session"""
        with self.lock:
            if session_id not in self.sessions:
                return None
            
            session = self.sessions[session_id]
            
            # Check if expired
            if datetime.now() >= session['expires_at']:
                return None
            
            return {
                'session_id': session_id,
                'created_at': session['created_at'],
                'last_activity': session['last_activity'],
                'expires_at': session['expires_at'],
                'time_remaining': session['expires_at'] - datetime.now()
            }
    
    def clear_session(self, session_id: str) -> bool:
        """Manually clear a session"""
        with self.lock:
            if session_id in self.sessions:
                del self.sessions[session_id]
                print(f"🗑️ Manually cleared session: {session_id}")
                return True
            return False
    
    def get_active_sessions(self) -> List[str]:
        """Get list of active session IDs"""
        with self.lock:
            active_sessions = []
            current_time = datetime.now()
            
            for session_id, session in self.sessions.items():
                if current_time < session['expires_at']:
                    active_sessions.append(session_id)
            
            return active_sessions
    
    def _cleanup_expired_sessions(self):
        """Background thread to clean up expired sessions"""
        while True:
            try:
                time.sleep(60)  # Check every minute
                
                with self.lock:
                    current_time = datetime.now()
                    expired_sessions = []
                    
                    for session_id, session in self.sessions.items():
                        if current_time >= session['expires_at']:
                            expired_sessions.append(session_id)
                    
                    # Remove expired sessions
                    for session_id in expired_sessions:
                        del self.sessions[session_id]
                        print(f"🕐 Auto-expired session: {session_id}")
                
            except Exception as e:
                print(f"❌ Error in session cleanup: {str(e)}")
    
    def get_session_count(self) -> int:
        """Get total number of active sessions"""
        with self.lock:
            current_time = datetime.now()
            active_count = sum(
                1 for session in self.sessions.values()
                if current_time < session['expires_at']
            )
            return active_count
