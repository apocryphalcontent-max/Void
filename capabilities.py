"""
Capability-Based Security with Macaroons

Implements a capability system using Macaroons (chained HMACs) for
decentralized authorization and token attenuation.

Macaroons provide:
1. Decentralized attenuation (add restrictions without server)
2. Contextual caveats (time, location, etc.)
3. Cryptographic verification without server roundtrips
4. Delegation without exposing root keys

This is crucial for:
- Tool execution permissions
- Resource access control
- Distributed authorization
- Zero-trust security model

References:
- "Macaroons: Cookies with Contextual Caveats for Decentralized
   Authorization in the Cloud" (Birgisson et al., 2014)
- HyperDex, Google's internal use of Macaroons
"""

import hashlib
import hmac
import json
import time
from typing import List, Dict, Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
import base64
from linear_types import LinearResource


# ============================================================================
# CAPABILITY TYPES
# ============================================================================

class CapabilityType(Enum):
    """Types of capabilities"""
    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    ADMIN = "admin"
    DELETE = "delete"


# ============================================================================
# CAVEATS (RESTRICTIONS)
# ============================================================================

@dataclass
class Caveat:
    """
    A restriction on a capability.
    
    Caveats can be:
    - First-party: Verifiable by the target service
    - Third-party: Requires verification by another service
    """
    identifier: str  # Human-readable caveat description
    verification_data: Optional[bytes] = None  # For third-party caveats
    
    def to_dict(self) -> dict:
        return {
            'identifier': self.identifier,
            'verification_data': base64.b64encode(self.verification_data).decode() if self.verification_data else None
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Caveat':
        vd = data.get('verification_data')
        return cls(
            identifier=data['identifier'],
            verification_data=base64.b64decode(vd) if vd else None
        )


class CaveatVerifier:
    """
    Verifies caveats against context.
    
    Each caveat type has a verification function.
    """
    
    def __init__(self):
        self.verifiers: Dict[str, Callable[[str, dict], bool]] = {}
    
    def register(self, caveat_prefix: str, verifier: Callable[[str, dict], bool]):
        """
        Register a caveat verifier.
        
        Args:
            caveat_prefix: Caveat identifier prefix (e.g., "time <")
            verifier: Function that verifies the caveat
        """
        self.verifiers[caveat_prefix] = verifier
    
    def verify(self, caveat: Caveat, context: dict) -> bool:
        """
        Verify a caveat against context.
        
        Args:
            caveat: The caveat to verify
            context: Verification context (current time, user, etc.)
            
        Returns:
            True if caveat is satisfied
        """
        identifier = caveat.identifier
        
        # Find matching verifier
        for prefix, verifier in self.verifiers.items():
            if identifier.startswith(prefix):
                return verifier(identifier, context)
        
        # Unknown caveat type - fail closed
        return False


# ============================================================================
# MACAROON IMPLEMENTATION
# ============================================================================

@dataclass
class Macaroon:
    """
    A Macaroon capability token.
    
    Consists of:
    - location: Target service/resource
    - identifier: Token identifier
    - signature: HMAC chain signature
    - caveats: List of restrictions
    """
    location: str
    identifier: str
    signature: bytes
    caveats: List[Caveat] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        """Serialize to dictionary"""
        return {
            'location': self.location,
            'identifier': self.identifier,
            'signature': base64.b64encode(self.signature).decode(),
            'caveats': [c.to_dict() for c in self.caveats]
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Macaroon':
        """Deserialize from dictionary"""
        return cls(
            location=data['location'],
            identifier=data['identifier'],
            signature=base64.b64decode(data['signature']),
            caveats=[Caveat.from_dict(c) for c in data['caveats']]
        )
    
    def serialize(self) -> str:
        """Serialize to base64 JSON string"""
        return base64.b64encode(json.dumps(self.to_dict()).encode()).decode()
    
    @classmethod
    def deserialize(cls, serialized: str) -> 'Macaroon':
        """Deserialize from base64 JSON string"""
        data = json.loads(base64.b64decode(serialized))
        return cls.from_dict(data)


class MacaroonFactory:
    """
    Factory for creating and attenuating Macaroons.
    
    The root key is kept secret. Only the auth server knows it.
    """
    
    def __init__(self, root_key: bytes):
        """
        Initialize factory with root key.
        
        Args:
            root_key: Secret root key (must be kept secure!)
        """
        self.root_key = root_key
    
    def mint(self, location: str, identifier: str) -> Macaroon:
        """
        Mint a new Macaroon.
        
        Only the auth server should do this.
        
        Args:
            location: Target service/resource
            identifier: Token identifier
            
        Returns:
            A new Macaroon
        """
        # Initial signature is HMAC(root_key, identifier)
        signature = hmac.new(
            self.root_key,
            identifier.encode(),
            hashlib.sha256
        ).digest()
        
        return Macaroon(
            location=location,
            identifier=identifier,
            signature=signature,
            caveats=[]
        )
    
    @staticmethod
    def add_first_party_caveat(macaroon: Macaroon, caveat_id: str) -> Macaroon:
        """
        Add a first-party caveat (restriction).
        
        This can be done by the user without contacting the auth server.
        The new signature is HMAC(old_signature, caveat_id).
        
        Args:
            macaroon: Original Macaroon
            caveat_id: Caveat identifier
            
        Returns:
            Attenuated Macaroon
        """
        # Create new caveat
        caveat = Caveat(identifier=caveat_id)
        
        # Compute new signature
        new_signature = hmac.new(
            macaroon.signature,
            caveat_id.encode(),
            hashlib.sha256
        ).digest()
        
        # Create attenuated macaroon
        return Macaroon(
            location=macaroon.location,
            identifier=macaroon.identifier,
            signature=new_signature,
            caveats=macaroon.caveats + [caveat]
        )
    
    def verify(
        self,
        macaroon: Macaroon,
        verifier: CaveatVerifier,
        context: dict
    ) -> bool:
        """
        Verify a Macaroon.
        
        Checks:
        1. Signature chain is valid
        2. All caveats are satisfied
        
        Args:
            macaroon: The Macaroon to verify
            verifier: Caveat verifier
            context: Verification context
            
        Returns:
            True if Macaroon is valid
        """
        # Reconstruct signature chain
        signature = hmac.new(
            self.root_key,
            macaroon.identifier.encode(),
            hashlib.sha256
        ).digest()
        
        # Apply each caveat to signature chain
        for caveat in macaroon.caveats:
            # Verify the caveat
            if not verifier.verify(caveat, context):
                return False
            
            # Update signature
            signature = hmac.new(
                signature,
                caveat.identifier.encode(),
                hashlib.sha256
            ).digest()
        
        # Check final signature matches
        return hmac.compare_digest(signature, macaroon.signature)


# ============================================================================
# CAPABILITY TOKENS (LINEAR RESOURCES)
# ============================================================================

@dataclass
class CapabilityToken:
    """
    A capability token that must be consumed.
    
    This is a linear resource - it can only be used once.
    This prevents confused deputy attacks and ensures proper
    capability discipline.
    """
    macaroon: Macaroon
    capability_type: CapabilityType
    resource_path: str
    
    def to_linear(self) -> LinearResource['CapabilityToken']:
        """
        Convert to a linear resource.
        
        Returns:
            A linear resource wrapping this capability
        """
        return LinearResource(
            self,
            name=f"Capability({self.resource_path}, {self.capability_type.value})"
        )


class CapabilityManager:
    """
    Manages capabilities and enforces linear consumption.
    
    Tracks capability usage to prevent:
    - Capability duplication
    - Capability leaks
    - Confused deputy attacks
    """
    
    def __init__(self, factory: MacaroonFactory, verifier: CaveatVerifier):
        """
        Initialize capability manager.
        
        Args:
            factory: Macaroon factory for verification
            verifier: Caveat verifier
        """
        self.factory = factory
        self.verifier = verifier
        self.used_tokens: set[str] = set()
    
    def grant_capability(
        self,
        resource_path: str,
        capability_type: CapabilityType,
        caveats: Optional[List[str]] = None
    ) -> LinearResource[CapabilityToken]:
        """
        Grant a capability token.
        
        Args:
            resource_path: Path to resource
            capability_type: Type of capability
            caveats: Optional list of caveat identifiers
            
        Returns:
            A linear capability resource
        """
        # Mint base macaroon
        identifier = f"{resource_path}:{capability_type.value}:{time.time()}"
        macaroon = self.factory.mint(resource_path, identifier)
        
        # Add caveats
        if caveats:
            for caveat_id in caveats:
                macaroon = MacaroonFactory.add_first_party_caveat(macaroon, caveat_id)
        
        # Create token
        token = CapabilityToken(
            macaroon=macaroon,
            capability_type=capability_type,
            resource_path=resource_path
        )
        
        return token.to_linear()
    
    def verify_capability(
        self,
        token_resource: LinearResource[CapabilityToken],
        context: dict
    ) -> bool:
        """
        Verify and consume a capability token.
        
        Args:
            token_resource: Linear capability resource
            context: Verification context
            
        Returns:
            True if capability is valid
        """
        # Consume the linear resource
        token = token_resource.consume()
        
        # Check if already used
        if token.macaroon.identifier in self.used_tokens:
            raise ValueError("Capability token already used (replay attack?)")
        
        # Verify the macaroon
        valid = self.factory.verify(token.macaroon, self.verifier, context)
        
        if valid:
            # Mark as used
            self.used_tokens.add(token.macaroon.identifier)
        
        return valid


# ============================================================================
# STANDARD CAVEAT VERIFIERS
# ============================================================================

def create_standard_verifier() -> CaveatVerifier:
    """
    Create a verifier with standard caveats.
    
    Returns:
        Caveat verifier with common verifiers registered
    """
    verifier = CaveatVerifier()
    
    # Time-based caveats
    def verify_time_before(identifier: str, context: dict) -> bool:
        """Verify: time < X"""
        try:
            _, timestamp_str = identifier.split(' < ')
            timestamp = float(timestamp_str)
            return context.get('time', time.time()) < timestamp
        except:
            return False
    
    def verify_time_after(identifier: str, context: dict) -> bool:
        """Verify: time > X"""
        try:
            _, timestamp_str = identifier.split(' > ')
            timestamp = float(timestamp_str)
            return context.get('time', time.time()) > timestamp
        except:
            return False
    
    # User-based caveats
    def verify_user(identifier: str, context: dict) -> bool:
        """Verify: user = X"""
        try:
            _, user_id = identifier.split(' = ')
            return context.get('user') == user_id
        except:
            return False
    
    # IP-based caveats
    def verify_ip(identifier: str, context: dict) -> bool:
        """Verify: ip = X"""
        try:
            _, ip_addr = identifier.split(' = ')
            return context.get('ip') == ip_addr
        except:
            return False
    
    verifier.register('time <', verify_time_before)
    verifier.register('time >', verify_time_after)
    verifier.register('user =', verify_user)
    verifier.register('ip =', verify_ip)
    
    return verifier


# ============================================================================
# DEMONSTRATION AND TESTING
# ============================================================================

def _example_usage():
    """Demonstrate capability system with Macaroons"""
    
    print("Capability System with Macaroons\n" + "="*70)
    
    # Setup
    root_key = b"super-secret-root-key"
    factory = MacaroonFactory(root_key)
    verifier = create_standard_verifier()
    manager = CapabilityManager(factory, verifier)
    
    print("\n1. Grant a capability:")
    cap_resource = manager.grant_capability(
        "/database/users",
        CapabilityType.READ,
        caveats=[f"time < {time.time() + 3600}"]  # Valid for 1 hour
    )
    print(f"   Granted: {cap_resource}")
    
    print("\n2. Verify the capability:")
    context = {'time': time.time()}
    valid = manager.verify_capability(cap_resource, context)
    print(f"   Valid: {valid}")
    
    print("\n3. Try to use again (should fail - linear type):")
    try:
        valid = manager.verify_capability(cap_resource, context)
        print(f"   ✗ Should not reach here")
    except Exception as e:
        print(f"   ✓ Caught: {type(e).__name__}")
    
    print("\n4. Attenuate a capability:")
    base_macaroon = factory.mint("/api/endpoint", "token123")
    print(f"   Base: {base_macaroon.identifier}")
    
    attenuated = MacaroonFactory.add_first_party_caveat(
        base_macaroon,
        f"time < {time.time() + 60}"
    )
    print(f"   Attenuated with time caveat")
    print(f"   Caveats: {[c.identifier for c in attenuated.caveats]}")
    
    print("\n5. Verify attenuated token:")
    valid = factory.verify(attenuated, verifier, {'time': time.time()})
    print(f"   Valid: {valid}")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    _example_usage()
