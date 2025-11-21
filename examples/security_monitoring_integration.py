#!/usr/bin/env python3
"""
Real-World Integration Example: Security Monitoring Pipeline

Demonstrates multi-phase tool integration for comprehensive security monitoring:
- Phase 1: Pattern analysis, entropy measurement, event classification
- Phase 2: Threat detection, behavioral anomalies, novelty detection
- Phase 3: Meta-tooling for adaptive threat response

This example shows how Void-State tools compose into production security systems.
"""

import sys
import time
from typing import Dict, List, Any
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from void_state_tools import (
    ToolRegistry,
    ToolConfig,
    # Phase 1 Tools
    PatternPrevalenceQuantifier,
    LocalEntropyMicroscope,
    EventSignatureClassifier,
    # Phase 2 Tools
    ThreatSignatureRecognizer,
    BehavioralAnomalyDetector,
    NoveltyDetector,
    # Phase 3 Tools
    ToolSynthesizer,
    ToolFitnessEvaluator,
)
from void_state_tools.clock import DeterministicClock


class SecurityMonitoringPipeline:
    """
    Production-ready security monitoring system using Void-State tools.

    Architecture:
    - Layer 1 (Sensing): Pattern prevalence, local entropy, event classification
    - Layer 2 (Analysis): Threat signatures, behavioral anomalies, novelty detection
    - Layer 3 (Meta): Adaptive tool synthesis for emerging threat patterns
    """

    def __init__(self, clock: DeterministicClock = None):
        self.clock = clock or DeterministicClock()
        self.registry = ToolRegistry()

        # Initialize Phase 1 tools
        self.pattern_analyzer = PatternPrevalenceQuantifier(
            ToolConfig(
                tool_name="pattern_analyzer",
                max_memory_mb=100,
                max_cpu_percent=10,
                overhead_budget_ns=1000,
            ),
            clock=self.clock,
        )

        self.entropy_analyzer = LocalEntropyMicroscope(
            ToolConfig(
                tool_name="entropy_analyzer",
                max_memory_mb=100,
                max_cpu_percent=10,
                overhead_budget_ns=1000,
            ),
            clock=self.clock,
        )

        self.event_classifier = EventSignatureClassifier(
            ToolConfig(
                tool_name="event_classifier",
                max_memory_mb=100,
                max_cpu_percent=10,
                overhead_budget_ns=2000,
            ),
            clock=self.clock,
        )

        # Initialize Phase 2 tools
        self.threat_detector = ThreatSignatureRecognizer(
            ToolConfig(
                tool_name="threat_detector",
                max_memory_mb=200,
                max_cpu_percent=15,
                overhead_budget_ns=5000,
            ),
            clock=self.clock,
        )

        self.anomaly_detector = BehavioralAnomalyDetector(
            ToolConfig(
                tool_name="anomaly_detector",
                max_memory_mb=150,
                max_cpu_percent=12,
                overhead_budget_ns=3000,
            ),
            clock=self.clock,
        )

        self.novelty_detector = NoveltyDetector(
            ToolConfig(
                tool_name="novelty_detector",
                max_memory_mb=150,
                max_cpu_percent=12,
                overhead_budget_ns=3000,
            ),
            clock=self.clock,
        )

        # Register all tools
        self._register_tools()

        print("✓ Security Monitoring Pipeline Initialized")
        print(f"  - {len(self.registry.list_tools())} tools registered")
        print(f"  - Multi-layer defense active")

    def _register_tools(self):
        """Register all tools with the central registry."""
        tools = [
            self.pattern_analyzer,
            self.entropy_analyzer,
            self.event_classifier,
            self.threat_detector,
            self.anomaly_detector,
            self.novelty_detector,
        ]

        for tool in tools:
            handle = self.registry.register_tool(tool)
            self.registry.lifecycle_manager.attach_tool(handle.tool_id)

    def process_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a security event through the complete pipeline.

        Returns comprehensive threat assessment with recommendations.
        """
        results = {
            "event": event,
            "timestamp": self.clock.now(),
            "alerts": [],
            "risk_score": 0.0,
            "recommendations": [],
        }

        # Layer 1: Pattern Analysis
        pattern_result = self.pattern_analyzer.analyze({
            "pattern": event.get("pattern", "unknown"),
            "context": event.get("source", "system"),
            "timestamp": self.clock.now(),
        })

        # Check for unusual pattern prevalence
        if pattern_result["frequency_ratio"] > 0.8:
            results["alerts"].append({
                "severity": "high",
                "type": "pattern_prevalence",
                "message": f"Unusual prevalence: {pattern_result['frequency_ratio']:.1%}",
            })
            results["risk_score"] += 30

        # Layer 1: Entropy Analysis
        if "data" in event:
            entropy_result = self.entropy_analyzer.analyze({
                "data": event["data"],
                "window_size": 100,
                "timestamp": self.clock.now(),
            })

            # Check for entropy anomalies (too low = predictable, too high = random)
            if entropy_result["shannon_entropy"] < 2.0:
                results["alerts"].append({
                    "severity": "medium",
                    "type": "low_entropy",
                    "message": f"Suspiciously low entropy: {entropy_result['shannon_entropy']:.2f} bits",
                })
                results["risk_score"] += 20
            elif entropy_result["shannon_entropy"] > 7.5:
                results["alerts"].append({
                    "severity": "medium",
                    "type": "high_entropy",
                    "message": f"High entropy detected: {entropy_result['shannon_entropy']:.2f} bits (possible encryption)",
                })
                results["risk_score"] += 15

        # Layer 1: Event Classification
        classifier_result = self.event_classifier.analyze({
            "event_type": event.get("type", "unknown"),
            "features": event.get("features", []),
            "timestamp": self.clock.now(),
        })

        results["event_class"] = classifier_result.get("predicted_class", "unknown")
        results["classification_confidence"] = classifier_result.get("confidence", 0.0)

        # Layer 2: Threat Signature Detection
        threat_result = self.threat_detector.analyze({
            "event_data": event,
            "context": event.get("source", "system"),
            "timestamp": self.clock.now(),
        })

        if threat_result.get("threat_detected", False):
            results["alerts"].append({
                "severity": "critical",
                "type": "threat_signature",
                "message": f"Threat detected: {threat_result.get('signature_name', 'unknown')}",
                "confidence": threat_result.get("confidence_score", 0.0),
            })
            results["risk_score"] += 50

            # Add threat-specific recommendations
            if "recommended_actions" in threat_result:
                results["recommendations"].extend(threat_result["recommended_actions"])

        # Layer 2: Behavioral Anomaly Detection
        anomaly_result = self.anomaly_detector.analyze({
            "behavior_sequence": event.get("behavior", []),
            "entity_id": event.get("entity_id", "unknown"),
            "timestamp": self.clock.now(),
        })

        if anomaly_result.get("is_anomalous", False):
            results["alerts"].append({
                "severity": "high",
                "type": "behavioral_anomaly",
                "message": f"Anomalous behavior: {anomaly_result.get('risk_score', 0.0):.2f} risk",
            })
            results["risk_score"] += 35

        # Layer 2: Novelty Detection
        novelty_result = self.novelty_detector.analyze({
            "observation": event,
            "domain": event.get("source", "system"),
            "timestamp": self.clock.now(),
        })

        if novelty_result.get("is_novel", False):
            results["alerts"].append({
                "severity": "medium",
                "type": "novel_pattern",
                "message": f"Novel pattern detected (novelty: {novelty_result.get('novelty_score', 0.0):.2f})",
                "learnability": novelty_result.get("learnability_score", 0.0),
            })
            results["risk_score"] += 25

            # If novel and learnable, recommend creating new signature
            if novelty_result.get("learnability_score", 0.0) > 0.7:
                results["recommendations"].append({
                    "action": "create_signature",
                    "description": "Learn this novel pattern as new signature",
                    "priority": "high",
                })

        # Cap risk score at 100
        results["risk_score"] = min(100, results["risk_score"])

        # Generate overall recommendations
        if results["risk_score"] > 70:
            results["recommendations"].append({
                "action": "isolate_entity",
                "description": "High risk score - consider isolation",
                "priority": "critical",
            })
        elif results["risk_score"] > 40:
            results["recommendations"].append({
                "action": "enhanced_monitoring",
                "description": "Elevated risk - increase monitoring",
                "priority": "high",
            })

        return results

    def generate_report(self, results_batch: List[Dict[str, Any]]) -> str:
        """Generate human-readable security report."""
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("SECURITY MONITORING REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"Generated: {self.clock.now()}")
        report_lines.append(f"Events Processed: {len(results_batch)}")
        report_lines.append("")

        # Calculate aggregate statistics
        total_alerts = sum(len(r["alerts"]) for r in results_batch)
        avg_risk = sum(r["risk_score"] for r in results_batch) / len(results_batch) if results_batch else 0
        critical_events = sum(1 for r in results_batch if r["risk_score"] > 70)

        report_lines.append("SUMMARY")
        report_lines.append("-" * 80)
        report_lines.append(f"  Total Alerts: {total_alerts}")
        report_lines.append(f"  Average Risk Score: {avg_risk:.1f}/100")
        report_lines.append(f"  Critical Events: {critical_events}")
        report_lines.append("")

        # Detail critical events
        if critical_events > 0:
            report_lines.append("CRITICAL EVENTS")
            report_lines.append("-" * 80)
            for i, result in enumerate(results_batch):
                if result["risk_score"] > 70:
                    report_lines.append(f"\nEvent #{i+1} (Risk: {result['risk_score']:.0f}/100)")
                    for alert in result["alerts"]:
                        report_lines.append(f"  [{alert['severity'].upper()}] {alert['message']}")

                    if result["recommendations"]:
                        report_lines.append("  Recommendations:")
                        for rec in result["recommendations"]:
                            report_lines.append(f"    - [{rec['priority'].upper()}] {rec['description']}")
            report_lines.append("")

        # Tool statistics
        report_lines.append("TOOL PERFORMANCE")
        report_lines.append("-" * 80)
        for tool_id in self.registry.list_tools():
            stats = self.registry.get_tool_stats(tool_id)
            if stats:
                report_lines.append(f"  {tool_id}:")
                report_lines.append(f"    - Invocations: {stats.get('total_invocations', 0)}")
                report_lines.append(f"    - Avg Latency: {stats.get('average_latency_ms', 0):.2f}ms")

        report_lines.append("")
        report_lines.append("=" * 80)

        return "\n".join(report_lines)


def demonstrate_security_monitoring():
    """Demonstrate comprehensive security monitoring pipeline."""
    print("\n" + "=" * 80)
    print("REAL-WORLD INTEGRATION: Security Monitoring Pipeline")
    print("=" * 80 + "\n")

    # Initialize pipeline
    clock = DeterministicClock()
    pipeline = SecurityMonitoringPipeline(clock=clock)

    print("\n" + "-" * 80)
    print("Scenario 1: SQL Injection Attempt")
    print("-" * 80)

    sql_injection_event = {
        "type": "http_request",
        "source": "web_server",
        "pattern": "sql_injection",
        "data": "'; DROP TABLE users; --",
        "features": ["single_quote", "sql_keyword", "comment_marker"],
        "behavior": ["normal_login", "abnormal_query", "data_exfiltration"],
        "entity_id": "user_192.168.1.105",
    }

    result = pipeline.process_event(sql_injection_event)
    print(f"\n✓ Event Processed")
    print(f"  Risk Score: {result['risk_score']:.0f}/100")
    print(f"  Alerts: {len(result['alerts'])}")

    for alert in result["alerts"]:
        severity_emoji = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}
        emoji = severity_emoji.get(alert["severity"], "⚪")
        print(f"    {emoji} [{alert['severity'].upper()}] {alert['message']}")

    if result["recommendations"]:
        print(f"  Recommendations:")
        for rec in result["recommendations"]:
            print(f"    → [{rec['priority'].upper()}] {rec['description']}")

    print("\n" + "-" * 80)
    print("Scenario 2: Unusual Network Traffic Pattern")
    print("-" * 80)

    clock.advance(60)

    network_anomaly_event = {
        "type": "network_traffic",
        "source": "firewall",
        "pattern": "port_scan",
        "data": bytes([i % 256 for i in range(100)]),  # Random-looking traffic
        "features": ["rapid_connection", "sequential_ports", "external_source"],
        "behavior": ["port_scan_22", "port_scan_23", "port_scan_80", "port_scan_443"],
        "entity_id": "external_10.0.0.1",
    }

    result = pipeline.process_event(network_anomaly_event)
    print(f"\n✓ Event Processed")
    print(f"  Risk Score: {result['risk_score']:.0f}/100")
    print(f"  Alerts: {len(result['alerts'])}")

    for alert in result["alerts"]:
        severity_emoji = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}
        emoji = severity_emoji.get(alert["severity"], "⚪")
        print(f"    {emoji} [{alert['severity'].upper()}] {alert['message']}")

    print("\n" + "-" * 80)
    print("Scenario 3: Novel Attack Pattern (Zero-Day)")
    print("-" * 80)

    clock.advance(120)

    zero_day_event = {
        "type": "process_execution",
        "source": "endpoint_detection",
        "pattern": "unknown_exploit",
        "data": "NOVEL_MALWARE_PAYLOAD_XYZ123",
        "features": ["memory_injection", "privilege_escalation", "lateral_movement"],
        "behavior": ["create_process", "inject_code", "network_connect", "file_encrypt"],
        "entity_id": "workstation_win10-dev-42",
    }

    result = pipeline.process_event(zero_day_event)
    print(f"\n✓ Event Processed")
    print(f"  Risk Score: {result['risk_score']:.0f}/100")
    print(f"  Event Class: {result['event_class']}")
    print(f"  Classification Confidence: {result['classification_confidence']:.2%}")
    print(f"  Alerts: {len(result['alerts'])}")

    for alert in result["alerts"]:
        severity_emoji = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}
        emoji = severity_emoji.get(alert["severity"], "⚪")
        print(f"    {emoji} [{alert['severity'].upper()}] {alert['message']}")

        # Show additional alert details
        if "learnability" in alert:
            print(f"       Learnability: {alert['learnability']:.2%}")

    if result["recommendations"]:
        print(f"  Recommendations:")
        for rec in result["recommendations"]:
            print(f"    → [{rec['priority'].upper()}] {rec['description']}")

    print("\n" + "=" * 80)
    print("Integration demonstrates:")
    print("  ✓ Multi-layer defense (Layers 1-2)")
    print("  ✓ Pattern, entropy, and behavioral analysis")
    print("  ✓ Threat signature and anomaly detection")
    print("  ✓ Novel pattern recognition with learnability assessment")
    print("  ✓ Risk scoring and actionable recommendations")
    print("  ✓ Production-ready security monitoring")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    demonstrate_security_monitoring()
