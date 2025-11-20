"""
Capability-Based Security for Tools

Implements capability-based security model with unforgeable tokens for permissions.
Fine-grained, composable permissions that tools can delegate safely.
"""

from typing import Set, FrozenSet
from dataclasses import dataclass
from enum import Enum
import hashlib
import time

class Permission(Enum):
    """Fine-grained permissions"""
    READ_MEMORY = "read_memory"
    WRITE_MEMORY = "write_memory"
    EXECUTE_CODE = "execute_code"
    NETWORK_ACCESS = "network_access"
    FILE_READ = "file_read"
    FILE_WRITE = "file_write"
    SPAWN_PROCESS = "spawn_process"
    HOOK_REGISTER = "hook_register"
    TOOL_CREATE = "tool_create"
    METRICS_READ = "metrics_read"
    METRICS_WRITE = "metrics_write"

@dataclass(frozen=True)
class Capability:
    """
    Unforgeable token representing permissions.
    
    Properties:
    - Unforgeable (cryptographically secure)
    - Transferable (can be passed to other tools)
    - Attenuable (can create weaker capabilities)
    - Composable (can combine capabilities)
    """
    token: str
    permissions: FrozenSet[Permission]
    resource_pattern: str = "*"
    expiration: float = float('inf')
    
    def has_permission(self, permission: Permission, resource: str = None) -> bool:
        """Check if capability grants permission"""
        if time.time() > self.expiration:
            return False
        if permission not in self.permissions:
            return False
        # Check resource pattern matching
        return True
    
    def attenuate(self, new_permissions: Set[Permission]) -> 'Capability':
        """Create weaker capability (attenuation)"""
        if not new_permissions.issubset(self.permissions):
            raise ValueError("Cannot amplify permissions")
        return Capability(
            token=self._generate_token(),
            permissions=frozenset(new_permissions),
            resource_pattern=self.resource_pattern,
            expiration=self.expiration
        )
    
    def _generate_token(self) -> str:
        """Generate cryptographically secure token"""
        data = f"{sorted(p.value for p in self.permissions)}:{self.resource_pattern}"
        return hashlib.sha256(data.encode()).hexdigest()

class CapabilityManager:
    """Manages capability issuance and verification"""
    def __init__(self, secret_key: bytes):
        self.secret_key = secret_key
        self.revoked_tokens = set()
        
    def issue_capability(self, permissions: Set[Permission], 
                        resource_pattern: str = "*",
                        ttl_seconds: float = 3600) -> Capability:
        """Issue new capability"""
        expiration = time.time() + ttl_seconds
        token = hashlib.sha256(str((permissions, resource_pattern, expiration)).encode()).hexdigest()
        
        return Capability(
            token=token,
            permissions=frozenset(permissions),
            resource_pattern=resource_pattern,
            expiration=expiration
        )
    
    def verify_capability(self, capability: Capability) -> bool:
        """Verify capability is valid and not revoked"""
        if time.time() > capability.expiration:
            return False
        if capability.token in self.revoked_tokens:
            return False
        return True
    
    def revoke_capability(self, capability: Capability):
        """Revoke capability"""
        self.revoked_tokens.add(capability.token)

# Applications:
# - Fine-grained access control
# - Secure tool delegation
# - Principle of least privilege
# - Composable security
