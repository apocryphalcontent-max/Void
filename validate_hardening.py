#!/usr/bin/env python3
"""
Comprehensive validation script for the Void system hardening.

This script tests all new modules to ensure they're working correctly.
"""

import sys


def test_linear_types():
    """Test linear types module"""
    print("Testing linear_types.py...")
    from linear_types import LinearResource, LinearContext, AffineResource
    
    # Test basic usage
    resource = LinearResource("test_value", name="test")
    value = resource.consume()
    assert value == "test_value", "LinearResource consumption failed"
    
    # Test context manager
    with LinearContext() as ctx:
        r1 = ctx.create("value1")
        r2 = ctx.create("value2")
        v1 = r1.consume()
        v2 = r2.consume()
        assert v1 == "value1" and v2 == "value2", "LinearContext failed"
    
    # Test affine resource
    affine = AffineResource("affine_value")
    del affine  # Should not raise
    
    print("  ✓ Linear types working correctly")


def test_dependent_types():
    """Test dependent types module"""
    print("Testing dependent_types.py...")
    from dependent_types import Vector, Range, NonEmpty
    
    # Test Vector
    v = Vector[3]([1, 2, 3])
    assert len(v) == 3, "Vector length incorrect"
    
    # Test Range
    age = Range[0, 120](25)
    assert int(age) == 25, "Range value incorrect"
    
    # Test NonEmpty
    items = NonEmpty[list]([1, 2, 3])
    assert len(items) == 3, "NonEmpty length incorrect"
    
    print("  ✓ Dependent types working correctly")


def test_hlc():
    """Test Hybrid Logical Clock"""
    print("Testing hlc.py...")
    from hlc import HybridLogicalClock, LWWRegister
    
    clock_a = HybridLogicalClock("node_a")
    clock_b = HybridLogicalClock("node_b")
    
    # Test basic timestamp generation
    ts1 = clock_a.now()
    ts2 = clock_a.now()
    assert ts1 < ts2, "HLC timestamp ordering failed"
    
    # Test message passing
    ts3 = clock_b.update(ts2)
    assert ts2 < ts3, "HLC update failed"
    
    # Test LWW Register
    reg1 = LWWRegister("value1", ts1)
    reg2 = LWWRegister("value2", ts2)
    merged = reg1.merge(reg2)
    assert merged.value == "value2", "LWW merge failed"
    
    print("  ✓ HLC working correctly")


def test_gossip():
    """Test Plumtree gossip protocol"""
    print("Testing gossip.py...")
    from gossip import PlumtreeNode
    
    node = PlumtreeNode("test_node")
    node.add_peer("peer1", eager=True)
    node.add_peer("peer2", eager=False)
    
    stats = node.get_stats()
    assert stats['eager_peers'] == 1, "Eager peers count incorrect"
    assert stats['lazy_peers'] == 1, "Lazy peers count incorrect"
    
    print("  ✓ Gossip protocol working correctly")


def test_pbft():
    """Test PBFT consensus"""
    print("Testing pbft.py...")
    from pbft import PBFTNode
    
    replica_ids = ["node0", "node1", "node2", "node3"]
    node = PBFTNode("node0", replica_ids, f=1)
    
    assert node.is_primary(), "Primary check failed"
    assert node.get_primary() == "node0", "Primary ID incorrect"
    
    stats = node.get_stats()
    assert stats['view'] == 0, "Initial view incorrect"
    
    print("  ✓ PBFT working correctly")


def test_capabilities():
    """Test capability system"""
    print("Testing capabilities.py...")
    from capabilities import (
        MacaroonFactory, create_standard_verifier,
        CapabilityManager, CapabilityType
    )
    
    factory = MacaroonFactory(b"test-key")
    verifier = create_standard_verifier()
    manager = CapabilityManager(factory, verifier)
    
    # Grant capability
    cap = manager.grant_capability(
        "/test/resource",
        CapabilityType.READ
    )
    
    # Verify
    import time
    valid = manager.verify_capability(cap, {'time': time.time()})
    assert valid, "Capability verification failed"
    
    print("  ✓ Capabilities working correctly")


def test_effects():
    """Test algebraic effects"""
    print("Testing effects.py...")
    from effects import (
        TimeHandler, LogHandler, with_handlers,
        ask_current_time, log_effect
    )
    
    # Test mock time
    time_handler = TimeHandler("mock")
    time_handler.set_mock_time(1000.0)
    
    log_handler = LogHandler("capture")
    
    with with_handlers(time_handler, log_handler):
        t = ask_current_time()
        assert t == 1000.0, "Mock time failed"
        
        log_effect("INFO", "test message")
        logs = log_handler.get_logs()
        assert len(logs) == 1, "Log capture failed"
        assert logs[0] == ("INFO", "test message"), "Log content incorrect"
    
    print("  ✓ Effects system working correctly")


def test_quantum_scheduling():
    """Test quantum scheduling"""
    print("Testing quantum_scheduling.py...")
    from quantum_scheduling import IsingScheduler, Task, Node
    
    tasks = [
        Task("task1", {"cpu": 1.0}),
        Task("task2", {"cpu": 1.0})
    ]
    
    nodes = [
        Node("node1", {"cpu": 2.0}),
        Node("node2", {"cpu": 2.0})
    ]
    
    scheduler = IsingScheduler(tasks, nodes)
    schedule = scheduler.schedule()
    
    assert len(schedule) == 2, "Schedule incomplete"
    assert all(task_id in schedule for task_id in ["task1", "task2"]), "Tasks not scheduled"
    
    print("  ✓ Quantum scheduling working correctly")


def test_causal():
    """Test causal inference"""
    print("Testing causal.py...")
    from causal import CausalGraph, CausalInference
    
    graph = CausalGraph()
    graph.add_node("A")
    graph.add_node("B")
    graph.add_node("C")
    graph.add_edge("A", "B", 0.8)
    graph.add_edge("B", "C", 0.7)
    
    assert graph.is_ancestor("A", "C"), "Ancestor check failed"
    
    inference = CausalInference(graph)
    
    # Test observational query
    prob = inference.observational_probability("C", {"A": True})
    assert 0 <= prob <= 1, "Probability out of range"
    
    print("  ✓ Causal inference working correctly")


def test_system():
    """Test system orchestration"""
    print("Testing system.py...")
    from system import VoidSystem, SystemConfig
    
    config = SystemConfig(
        node_id="test-node",
        enable_gossip=False,
        enable_pbft=False,
        enable_scheduler=True,
        enable_monitoring=False
    )
    
    system = VoidSystem(config)
    success = system.initialize()
    assert success, "System initialization failed"
    
    status = system.get_status()
    assert status['state'] == 'READY', "System state incorrect"
    assert status['node_id'] == "test-node", "Node ID incorrect"
    
    system.shutdown()
    
    status = system.get_status()
    assert status['state'] == 'STOPPED', "System shutdown failed"
    
    print("  ✓ System orchestration working correctly")


def main():
    """Run all tests"""
    print("="*70)
    print("VOID SYSTEM HARDENING - COMPREHENSIVE VALIDATION")
    print("="*70)
    print()
    
    tests = [
        test_linear_types,
        test_dependent_types,
        test_hlc,
        test_gossip,
        test_pbft,
        test_capabilities,
        test_effects,
        test_quantum_scheduling,
        test_causal,
        test_system
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"  ✗ Test failed: {e}")
            import traceback
            traceback.print_exc()
    
    print()
    print("="*70)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("="*70)
    
    if failed > 0:
        sys.exit(1)
    else:
        print("\n✅ All tests passed! System hardening complete.")
        sys.exit(0)


if __name__ == "__main__":
    main()
