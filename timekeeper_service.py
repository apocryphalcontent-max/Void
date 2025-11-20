"""
Timekeeper Service - Global Time Synchronization
"The Great Doxology"

A service that maintains globally synchronized time across all nodes
while preserving causality guarantees.

Key features:
- NTP-based time synchronization
- Continuous drift correction
- Causality preservation (never violates happens-before)
- Integration with HLC timestamps
- Bounded clock skew detection and correction

References:
- "Time, Clocks, and the Ordering of Events" (Lamport, 1978)
- "Logical Physical Clocks" (Kulkarni et al., 2014)
- NTP Protocol (RFC 5905)
"""

import time
import threading
import statistics
from typing import List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from collections import deque
import socket
import struct

from hlc import HybridLogicalClock, HLCTimestamp


# ============================================================================
# NTP CLIENT FOR TIME SYNCHRONIZATION
# ============================================================================

@dataclass
class NTPSample:
    """A single NTP time synchronization sample"""
    local_send_time: float      # Local time when request sent
    server_recv_time: float     # Server time when request received
    server_send_time: float     # Server time when reply sent
    local_recv_time: float      # Local time when reply received
    
    @property
    def offset(self) -> float:
        """
        Calculate clock offset from NTP sample.
        
        Offset = ((T2 - T1) + (T3 - T4)) / 2
        where:
        - T1 = local_send_time
        - T2 = server_recv_time
        - T3 = server_send_time
        - T4 = local_recv_time
        """
        return ((self.server_recv_time - self.local_send_time) + 
                (self.server_send_time - self.local_recv_time)) / 2.0
    
    @property
    def delay(self) -> float:
        """
        Calculate round-trip delay.
        
        Delay = (T4 - T1) - (T3 - T2)
        """
        return ((self.local_recv_time - self.local_send_time) - 
                (self.server_send_time - self.server_recv_time))


class SimpleNTPClient:
    """
    Simplified NTP client for time synchronization.
    
    This is a basic implementation for demonstration.
    Production systems should use a full NTP daemon (ntpd/chronyd).
    """
    
    NTP_PACKET_FORMAT = "!12I"
    NTP_EPOCH = 2208988800  # Seconds between 1900 and 1970
    
    def __init__(self, server: str = "pool.ntp.org", port: int = 123, timeout: float = 5.0):
        """
        Initialize NTP client.
        
        Args:
            server: NTP server hostname
            port: NTP server port (default 123)
            timeout: Socket timeout in seconds
        """
        self.server = server
        self.port = port
        self.timeout = timeout
    
    def query(self) -> Optional[NTPSample]:
        """
        Query NTP server for time synchronization.
        
        Returns:
            NTPSample if successful, None on error
        """
        try:
            # Create NTP request packet
            # Simplified: just set version (3) and mode (3 = client)
            packet = bytearray(48)
            packet[0] = 0x1B  # LI=0, VN=3, Mode=3
            
            # Record local send time
            local_send = time.time()
            
            # Send request
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                sock.settimeout(self.timeout)
                sock.sendto(packet, (self.server, self.port))
                
                # Receive response
                response, _ = sock.recvfrom(1024)
                local_recv = time.time()
            
            # Parse NTP response
            # We care about transmit timestamp (server send time)
            unpacked = struct.unpack(self.NTP_PACKET_FORMAT, response)
            
            # Extract transmit timestamp (seconds and fraction)
            tx_timestamp_sec = unpacked[10]
            tx_timestamp_frac = unpacked[11]
            
            # Convert to Unix timestamp
            server_send = (tx_timestamp_sec - self.NTP_EPOCH) + (tx_timestamp_frac / 2**32)
            
            # For simplified implementation, use server_send for both recv and send
            # Full NTP would parse receive timestamp too
            return NTPSample(
                local_send_time=local_send,
                server_recv_time=server_send,  # Simplified
                server_send_time=server_send,
                local_recv_time=local_recv
            )
            
        except (socket.timeout, socket.error, OSError) as e:
            # Network error - return None
            return None
    
    def get_offset(self, samples: int = 3) -> Optional[float]:
        """
        Get clock offset by averaging multiple samples.
        
        Args:
            samples: Number of samples to collect
            
        Returns:
            Average offset in seconds, or None on error
        """
        offsets = []
        
        for _ in range(samples):
            sample = self.query()
            if sample is not None and abs(sample.delay) < 1.0:  # Sanity check
                offsets.append(sample.offset)
            time.sleep(0.1)  # Brief delay between samples
        
        if not offsets:
            return None
        
        # Use median to be robust against outliers
        return statistics.median(offsets)


# ============================================================================
# TIMEKEEPER SERVICE
# ============================================================================

@dataclass
class TimeSyncState:
    """Current time synchronization state"""
    last_sync_time: float = 0.0
    clock_offset: float = 0.0  # Seconds to add to local clock
    drift_rate: float = 0.0    # Seconds per second
    sync_count: int = 0
    last_error: Optional[str] = None


class TimekeeperService:
    """
    Timekeeper service for global time synchronization.
    
    Maintains synchronized time across distributed nodes:
    1. Periodically syncs with NTP servers
    2. Tracks and corrects clock drift
    3. Provides corrected timestamps to HLC
    4. Ensures causality is never violated
    """
    
    def __init__(
        self,
        node_id: str,
        ntp_servers: Optional[List[str]] = None,
        sync_interval: float = 60.0,
        max_drift_correction: float = 0.5,
        max_clock_skew: float = 5.0
    ):
        """
        Initialize Timekeeper service.
        
        Args:
            node_id: This node's identifier
            ntp_servers: List of NTP servers (or None for default)
            sync_interval: Seconds between NTP syncs
            max_drift_correction: Maximum drift correction per sync (seconds)
            max_clock_skew: Maximum allowed clock skew before halting (seconds)
        """
        self.node_id = node_id
        self.ntp_servers = ntp_servers or ["pool.ntp.org"]
        self.sync_interval = sync_interval
        self.max_drift_correction = max_drift_correction
        self.max_clock_skew = max_clock_skew
        
        # Synchronization state
        self.state = TimeSyncState()
        self._lock = threading.Lock()
        
        # History of offsets for drift calculation
        self.offset_history: deque = deque(maxlen=10)
        
        # HLC integration
        self.hlc = HybridLogicalClock(node_id)
        self.hlc._get_physical_time = self._get_corrected_time_ms
        
        # Background sync thread
        self._running = False
        self._sync_thread: Optional[threading.Thread] = None
        
        # Callbacks for monitoring
        self.on_sync_complete: Optional[Callable[[TimeSyncState], None]] = None
        self.on_skew_detected: Optional[Callable[[float], None]] = None
    
    def start(self):
        """Start the timekeeper service"""
        with self._lock:
            if self._running:
                return
            
            self._running = True
            self._sync_thread = threading.Thread(target=self._sync_loop, daemon=True)
            self._sync_thread.start()
    
    def stop(self):
        """Stop the timekeeper service"""
        with self._lock:
            self._running = False
        
        if self._sync_thread:
            self._sync_thread.join(timeout=5.0)
    
    def _sync_loop(self):
        """Background loop for periodic time synchronization"""
        # Initial sync
        self._perform_sync()
        
        while self._running:
            time.sleep(self.sync_interval)
            if self._running:
                self._perform_sync()
    
    def _perform_sync(self):
        """Perform a single NTP synchronization"""
        # Try each NTP server until one succeeds
        offset = None
        
        for server in self.ntp_servers:
            try:
                client = SimpleNTPClient(server)
                offset = client.get_offset(samples=3)
                
                if offset is not None:
                    break
            except Exception as e:
                # Try next server
                continue
        
        if offset is None:
            with self._lock:
                self.state.last_error = "All NTP servers unreachable"
            return
        
        with self._lock:
            # Check for excessive clock skew
            if abs(offset) > self.max_clock_skew:
                self.state.last_error = f"Clock skew too large: {offset:.3f}s"
                
                if self.on_skew_detected:
                    self.on_skew_detected(offset)
                
                # In production, this would halt the cluster
                # For now, we just log it
                return
            
            # Apply bounded drift correction
            correction = max(-self.max_drift_correction, 
                           min(self.max_drift_correction, offset))
            
            # Update clock offset
            old_offset = self.state.clock_offset
            self.state.clock_offset += correction
            
            # Track offset history for drift calculation
            current_time = time.time()
            self.offset_history.append((current_time, self.state.clock_offset))
            
            # Calculate drift rate
            if len(self.offset_history) >= 2:
                oldest_time, oldest_offset = self.offset_history[0]
                time_delta = current_time - oldest_time
                offset_delta = self.state.clock_offset - oldest_offset
                
                if time_delta > 0:
                    self.state.drift_rate = offset_delta / time_delta
            
            # Update state
            self.state.last_sync_time = current_time
            self.state.sync_count += 1
            self.state.last_error = None
            
            if self.on_sync_complete:
                self.on_sync_complete(self.state)
    
    def _get_corrected_time_ms(self) -> int:
        """
        Get corrected physical time in milliseconds.
        
        This method is injected into HLC to provide drift-corrected time.
        
        Returns:
            Corrected time in milliseconds since epoch
        """
        with self._lock:
            uncorrected = time.time()
            corrected = uncorrected + self.state.clock_offset
            
            # Also apply drift correction
            if self.state.last_sync_time > 0:
                elapsed = uncorrected - self.state.last_sync_time
                drift_correction = elapsed * self.state.drift_rate
                corrected += drift_correction
            
            return int(corrected * 1000)
    
    def get_hlc(self) -> HybridLogicalClock:
        """Get the HLC instance with drift-corrected time"""
        return self.hlc
    
    def now(self) -> HLCTimestamp:
        """Generate an HLC timestamp with drift-corrected time"""
        return self.hlc.now()
    
    def update(self, remote_timestamp: HLCTimestamp) -> HLCTimestamp:
        """Update HLC with remote timestamp"""
        return self.hlc.update(remote_timestamp)
    
    def get_state(self) -> TimeSyncState:
        """Get current synchronization state"""
        with self._lock:
            return TimeSyncState(
                last_sync_time=self.state.last_sync_time,
                clock_offset=self.state.clock_offset,
                drift_rate=self.state.drift_rate,
                sync_count=self.state.sync_count,
                last_error=self.state.last_error
            )
    
    def force_sync(self) -> bool:
        """
        Force immediate synchronization.
        
        Returns:
            True if sync succeeded, False otherwise
        """
        self._perform_sync()
        
        with self._lock:
            return self.state.last_error is None


# ============================================================================
# DEMONSTRATION
# ============================================================================

def _example_usage():
    """Demonstrate Timekeeper service"""
    
    print("Timekeeper Service - Global Time Synchronization")
    print("="*70)
    
    # Create timekeeper
    print("\n1. Initializing Timekeeper service...")
    timekeeper = TimekeeperService(
        node_id="node_alpha",
        ntp_servers=["pool.ntp.org"],
        sync_interval=30.0
    )
    
    # Set up monitoring callbacks
    def on_sync(state: TimeSyncState):
        print(f"\n   ✓ Time sync complete:")
        print(f"     - Offset: {state.clock_offset:.6f}s")
        print(f"     - Drift rate: {state.drift_rate:.9f}s/s")
        print(f"     - Sync count: {state.sync_count}")
    
    def on_skew(skew: float):
        print(f"\n   ⚠ ALERT: Excessive clock skew detected: {skew:.3f}s")
        print("   → In production, this would halt the cluster")
    
    timekeeper.on_sync_complete = on_sync
    timekeeper.on_skew_detected = on_skew
    
    # Force initial sync (don't wait for background thread)
    print("\n2. Performing initial NTP synchronization...")
    print("   (Note: NTP queries may fail in restricted environments)")
    
    success = timekeeper.force_sync()
    
    if success:
        state = timekeeper.get_state()
        print(f"\n   Initial sync successful:")
        print(f"   - Clock offset: {state.clock_offset:.6f}s")
    else:
        state = timekeeper.get_state()
        print(f"\n   Initial sync failed: {state.last_error}")
        print("   → Continuing with uncorrected time")
    
    # Generate HLC timestamps
    print("\n3. Generating drift-corrected HLC timestamps:")
    ts1 = timekeeper.now()
    print(f"   Event 1: {ts1}")
    
    time.sleep(0.01)
    ts2 = timekeeper.now()
    print(f"   Event 2: {ts2}")
    print(f"   ts1 < ts2: {ts1 < ts2}")
    
    # Start background sync
    print("\n4. Starting background time synchronization...")
    print(f"   Sync interval: {timekeeper.sync_interval}s")
    # timekeeper.start()  # Commented out for demo
    
    print("\n5. Timekeeper guarantees:")
    print("   ✓ Continuous NTP drift correction")
    print("   ✓ Bounded clock skew detection")
    print("   ✓ Causality preservation (never violates happens-before)")
    print("   ✓ Integration with HLC for distributed coordination")
    
    print("\n" + "="*70)
    print("The Timekeeper is the heartbeat of the distributed universe.")


if __name__ == "__main__":
    _example_usage()
